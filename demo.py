"""End-to-end demo of the agentic email assistant.

Acts:
  1. The inbox            — what is waiting in the mock mailbox
  2. Autonomous run       — triage -> summarize -> draft -> follow-up sweep
                            (every outgoing action is queued, nothing is sent)
  3. Human review         — approve one reply, reject another with feedback
  4. The learning moment  — the drafter re-drafts the rejected reply, now
                            honoring the feedback stored in long-term memory
  5. Telemetry            — metrics report + run report artifact

The demo runs in an isolated temporary data directory, so repository state
(queue, memory, calendar) is untouched and every run starts fresh.

Usage:
  python demo.py             # live mode: real Claude calls (needs ANTHROPIC_API_KEY)
  python demo.py --offline   # scripted mode: canned agent outputs, no API key
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from email_assistant.config import Settings
from email_assistant.hitl import PendingActionQueue, execute_action, record_feedback
from email_assistant.memory.long_term import MemoryStore
from email_assistant.observability import MetricsRegistry, setup_logging
from email_assistant.orchestrator import Orchestrator, write_report
from email_assistant.schemas import (
    ActionType,
    FollowUpKind,
    FollowUpProposal,
    Intent,
    MemoryKind,
    PendingAction,
    Priority,
    ReplyDraft,
    Route,
    ThreadSummary,
    Tone,
    TriageResult,
)
from email_assistant.tools.calendar import MockCalendarProvider
from email_assistant.tools.mail import MockMailProvider

console = Console()
REPO_ROOT = Path(__file__).resolve().parent

REJECTION_NOTE = (
    "Too formal and too long. Keep replies to Marta to two short sentences, "
    "no pleasantries."
)


# ---------------------------------------------------------------------------
# Offline mode: a scripted LLM with realistic canned outputs
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Implements the LLM protocol with canned, schema-valid outputs."""

    def __init__(self, canned: list[BaseModel]) -> None:
        self._canned = list(canned)
        self.metrics = MetricsRegistry()

    def structured(self, *, output_type, system, user, model=None,
                   max_tokens=16000, name="llm_call"):
        self.metrics.incr(f"{name}_calls")
        for i, obj in enumerate(self._canned):
            if isinstance(obj, output_type):
                return self._canned.pop(i)
        raise RuntimeError(f"scripted demo has no canned {output_type.__name__} left")


def _triage(tid, intent, priority, route, labels, rationale):
    return TriageResult(
        thread_id=tid, intent=intent, priority=priority, labels=labels,
        route=route, confidence=0.93, rationale=rationale,
    )


def build_scripted_llm() -> ScriptedLLM:
    """Canned outputs in processing order (unread threads, newest first)."""
    return ScriptedLLM(
        [
            # -- triage: t-002, t-001, [t-004/t-006 hit the rule pre-filter],
            #    t-009, t-003, t-007, t-008
            _triage("t-002", Intent.question, Priority.high, Route.needs_reply,
                    ["client", "api", "rate-limits"],
                    "Client asks for a burst raise and a pricing contact."),
            _triage("t-001", Intent.request, Priority.urgent, Route.needs_reply,
                    ["incident", "postmortem", "deadline"],
                    "Manager requests the postmortem timeline by EOD."),
            _triage("t-009", Intent.request, Priority.high, Route.needs_reply,
                    ["onboarding", "hr"],
                    "HR needs desk, training slot and the Snowflake request."),
            _triage("t-003", Intent.meeting, Priority.normal, Route.needs_reply,
                    ["scheduling", "roadmap"],
                    "Meeting proposal needs a confirmation."),
            _triage("t-007", Intent.fyi, Priority.low, Route.no_action,
                    ["facilities"], "Office closure notice, keep and label."),
            _triage("t-008", Intent.billing, Priority.high, Route.needs_reply,
                    ["invoice", "vendor"],
                    "Overdue vendor invoice needs a response."),
            # -- summaries (multi-message needs_reply threads: t-002, t-009)
            ThreadSummary(
                thread_id="t-002",
                tl_dr=("Dana (Northwind Labs) hit 429s from burst traffic; the Pro "
                       "limit is 1000 req/min per key. She now asks for a temporary "
                       "burst raise for quarter-end and a pricing contact."),
                key_points=["429s appear around 14:00 CET during the sync job",
                            "Key ID provided: NWL-PROD-07"],
                open_questions=["Can burst limits be raised for the backfill?",
                                "Who owns pricing for a permanent raise?"],
                action_items=["Adam: review traffic on NWL-PROD-07",
                              "Adam: route the pricing question to Priya Nair"],
            ),
            ThreadSummary(
                thread_id="t-009",
                tl_dr=("Onboarding for Elena Sousa (starts Aug 3) is mostly settled; "
                       "desk, security training and the Snowflake request are still "
                       "on Adam."),
                key_points=["Jonas confirmed as week-1 buddy",
                            "Snowflake approvals take ~2 weeks"],
                open_questions=["Desk allocation and security training slot?"],
                action_items=["Adam: file the Snowflake request by Friday",
                              "Adam: confirm desk and training slot to Sofia"],
            ),
            # -- drafts (popped in order: t-002, t-001, t-009, t-003, t-008)
            ReplyDraft(
                thread_id="t-002", to=["dana.reyes@northwindlabs.com"],
                subject="Re: Question about API rate limits", tone=Tone.neutral,
                body=("Hi Dana,\n\nThanks for the key ID — I'll review the traffic "
                      "on NWL-PROD-07 and get back to you. A temporary burst raise "
                      "for the quarter-end backfill is doable via a support ticket; "
                      "for permanent pricing changes Priya Nair is the right "
                      "contact and I'm happy to loop her in.\n\nBest, Adam"),
                rationale=("Grounded in org facts: burst raises go via support "
                           "ticket, pricing is owned by Priya Nair."),
            ),
            ReplyDraft(
                thread_id="t-001", to=["marta.lindqvist@acme.io"],
                subject="Re: URGENT: Postmortem for yesterday's checkout outage",
                tone=Tone.formal,
                body=("Dear Marta,\n\nThank you for bringing this to my attention. "
                      "I would like to confirm that I have received your message "
                      "regarding the postmortem for yesterday's checkout outage. I "
                      "fully understand the importance and urgency of this matter, "
                      "and I want to assure you that it has my complete attention.\n\n"
                      "I will take ownership of the timeline section as requested. I "
                      "am currently reviewing the incident logs and will begin "
                      "assembling the timeline immediately. I anticipate being able "
                      "to confirm my progress with you before lunch, as requested.\n\n"
                      "Please do not hesitate to reach out if there is anything "
                      "further you need from me.\n\nKind regards,\nAdam Foldvari"),
                rationale="Manager escalation — formal acknowledgement.",
            ),
            ReplyDraft(
                thread_id="t-009", to=["sofia.brandt@acme.io"],
                subject="Re: Onboarding plan — Elena Sousa", tone=Tone.neutral,
                body=("Hi Sofia,\n\nConfirming the remaining items: I'll book a desk "
                      "for Elena and reserve a security training slot in week 1. The "
                      "Snowflake read request goes into the IT portal today, given "
                      "the ~2-week approval time Jonas flagged.\n\nBest, Adam"),
                rationale="Answers the two open items and commits to the request.",
            ),
            ReplyDraft(
                thread_id="t-003", to=["priya.nair@acme.io"],
                subject="Re: Q3 roadmap sync — Thursday 14:00?", tone=Tone.casual,
                body=("Hi Priya,\n\nThursday 14:00 works — booked 45 minutes. Agenda "
                      "looks good; I'll bring the current rate-limit tier numbers "
                      "for the revamp discussion.\n\nBest, Adam"),
                rationale="Afternoon slot matches the user's meeting preference.",
            ),
            ReplyDraft(
                thread_id="t-008", to=["billing@datacloud.example"],
                subject="Re: Invoice #2219 — 11 days overdue", tone=Tone.formal,
                body=("Hello,\n\nThanks for the reminder about invoice #2219. I'm "
                      "forwarding it to finance@acme.io with the PO number — they "
                      "handle vendor payments and will confirm the payment date "
                      "directly.\n\nBest, Adam"),
                rationale="Org fact: vendor invoices are paid by finance@acme.io.",
            ),
            # -- follow-up sweep: t-005 (we sent the last message, no reply)
            FollowUpProposal(
                thread_id="t-005", kind=FollowUpKind.reminder,
                title="Chase DataCloud renewal quote",
                note="No reply since Jul 10; finance needs the quote before Jul 25.",
                due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
            ),
            # -- Act 4: the re-draft after the rejection (honors the feedback)
            ReplyDraft(
                thread_id="t-001", to=["marta.lindqvist@acme.io"],
                subject="Re: URGENT: Postmortem for yesterday's checkout outage",
                tone=Tone.neutral,
                body=("Marta — on it, I own the timeline section. You'll have my "
                      "confirmation before lunch and the full timeline by EOD.\n\n"
                      "Adam"),
                rationale=("Per stored feedback: two short sentences, no "
                           "pleasantries."),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Demo flow
# ---------------------------------------------------------------------------


def act(number: int, title: str) -> None:
    console.print(f"\n[bold cyan]{'=' * 66}[/bold cyan]")
    console.print(f"[bold cyan]Act {number} — {title}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 66}[/bold cyan]\n")


def find_action(queue: PendingActionQueue, action_type: ActionType,
                thread_id: str) -> PendingAction | None:
    for action in queue.pending():
        if action.type == action_type and action.payload.thread_id == thread_id:
            return action
    return None


def show_draft(title: str, draft: ReplyDraft) -> None:
    console.print(Panel(
        f"[bold]to:[/bold] {', '.join(draft.to)}\n"
        f"[bold]subject:[/bold] {draft.subject}\n"
        f"[bold]tone:[/bold] {draft.tone.value}\n\n{draft.body}",
        title=title,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="Run with scripted agent outputs (no API key needed).")
    args = parser.parse_args()

    # Keep the show clean; full JSON logs are demonstrated in docs/samples/.
    setup_logging(level=os.environ.get("ASSISTANT_LOG_LEVEL", "WARNING"))

    # Isolated workspace: repo data/ is never touched.
    workspace = Path(tempfile.mkdtemp(prefix="email-assistant-demo-"))
    (workspace / "fixtures").mkdir()
    (workspace / "memory").mkdir()
    shutil.copy(REPO_ROOT / "data" / "fixtures" / "inbox.json",
                workspace / "fixtures" / "inbox.json")
    shutil.copy(REPO_ROOT / "data" / "memory" / "memory.seed.json",
                workspace / "memory" / "memory.seed.json")

    settings = Settings(data_dir=workspace)
    if args.offline:
        llm = build_scripted_llm()
        mode = "OFFLINE (scripted outputs, no API calls)"
    else:
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY is not set.[/red] "
                          "Put it in .env, or run:  python demo.py --offline")
            return 1
        from email_assistant.llm import ClaudeClient
        llm = ClaudeClient(settings=settings)
        mode = f"LIVE ({settings.model})"

    mail = MockMailProvider(settings.fixtures_path,
                            user_email=settings.user_email,
                            user_name=settings.user_name)
    calendar = MockCalendarProvider(settings.calendar_path)
    memory = MemoryStore(settings.memory_path, seed_path=settings.memory_seed_path)
    queue = PendingActionQueue(settings.actions_path)
    orchestrator = Orchestrator(llm=llm, mail=mail, memory=memory,
                                queue=queue, settings=settings)

    console.print(Panel(
        f"Agentic email assistant — end-to-end demo\nMode: {mode}\n"
        f"Workspace: {workspace}",
        title="demo",
    ))

    # ---- Act 1: the inbox ---------------------------------------------------
    act(1, "The inbox")
    table = Table()
    for column in ("thread", "from", "subject"):
        table.add_column(column)
    for thread in mail.fetch_unread():
        table.add_row(thread.id, thread.last_message.sender, thread.subject)
    console.print(table)
    console.print(f"{len(mail.fetch_unread())} unread threads. The vendor thread "
                  "t-005 is read (we sent the last message) — watch the "
                  "follow-up sweep catch it.")

    # ---- Act 2: autonomous run ----------------------------------------------
    act(2, "Autonomous run — triage, summarize, draft, schedule")
    report = orchestrator.run()
    table = Table(title=f"Triage decisions (run {report['run_id']})")
    for column in ("thread", "intent", "priority", "route", "labels"):
        table.add_column(column)
    for ctx in orchestrator.state.contexts.values():
        if ctx.triage:
            table.add_row(ctx.thread_id, ctx.triage.intent.value,
                          ctx.triage.priority.value, ctx.triage.route.value,
                          ", ".join(ctx.triage.labels))
    console.print(table)
    console.print(f"\n[bold]{report['pending_actions']} outgoing actions queued — "
                  "nothing was sent.[/bold] The newsletter and the spam were "
                  "archived by the deterministic rule pre-filter (zero API calls).")

    # ---- Act 3: human review -------------------------------------------------
    act(3, "Human review — approve, reject with feedback")

    approve_reply = find_action(queue, ActionType.send_reply, "t-002")
    if approve_reply:
        show_draft("Draft for the client (t-002) — APPROVE",
                   approve_reply.payload)
        queue.approve(approve_reply.id)
        console.print(f"[green]{execute_action(queue.get(approve_reply.id), mail, calendar)}[/green]")
        queue.mark_executed(approve_reply.id)

    reject_reply = (find_action(queue, ActionType.send_reply, "t-001")
                    or next((a for a in queue.pending()
                             if a.type == ActionType.send_reply), None))
    rejected_draft = None
    if reject_reply:
        rejected_draft = reject_reply.payload
        show_draft("Draft for the manager (t-001) — REJECT", rejected_draft)
        queue.reject(reject_reply.id, REJECTION_NOTE)
        record_feedback(memory, queue.get(reject_reply.id), "rejected",
                        REJECTION_NOTE)
        console.print(f'[yellow]Rejected with note: "{REJECTION_NOTE}"[/yellow]')

    followup = find_action(queue, ActionType.create_followup, "t-005")
    if followup:
        payload = followup.payload
        console.print(Panel(
            f"[bold]{payload.kind.value}[/bold]: {payload.title}\n"
            f"due: {payload.due_at:%Y-%m-%d %H:%M} UTC\n{payload.note}",
            title="Follow-up for the silent vendor (t-005) — APPROVE",
        ))
        queue.approve(followup.id)
        console.print(f"[green]{execute_action(queue.get(followup.id), mail, calendar)}[/green]")
        queue.mark_executed(followup.id)

    console.print("\nRemaining drafts left pending — the reviewer decides later.")

    # ---- Act 4: the learning moment -----------------------------------------
    act(4, "The learning moment — memory changes behavior")
    feedback = memory.recall(kind=MemoryKind.feedback)
    console.print(Panel(feedback[-1].value, title="Stored feedback record"))
    console.print("Re-running the DraftAgent for the same thread — the feedback "
                  "record is now injected into its prompt:\n")
    new_draft = orchestrator.drafter.draft(mail.get_thread("t-001"), memory)
    show_draft("New draft for t-001 (after feedback)", new_draft)
    if rejected_draft:
        console.print(f"Old draft: {len(rejected_draft.body)} chars -> "
                      f"new draft: {len(new_draft.body)} chars.")

    # ---- Act 5: telemetry -----------------------------------------------------
    act(5, "Telemetry — what happened, measured")
    console.print(orchestrator.metrics.render_report())
    report_path = write_report(report, settings.runs_dir)
    console.print(f"\nRun report artifact: {report_path}")
    console.print("Structured JSON logs (run_id -> span_id correlation) are "
                  "committed in docs/samples/.")

    console.print(Panel(
        "1. Every outgoing action waited for human approval — nothing was sent "
        "autonomously.\n"
        "2. The rejection became a feedback memory record, and the very next "
        "draft honored it.\n"
        "3. Every step was traced (run_id/span_id), measured and reported.",
        title="What this demo proved",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
