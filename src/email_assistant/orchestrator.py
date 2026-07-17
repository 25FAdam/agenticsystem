"""Orchestrator: the run loop that coordinates the worker agents.

One run = fetch unread threads -> triage each -> dispatch by route
(needs_reply -> summarize + draft; follow_up -> scheduler proposal;
archive / no_action -> mail actions) -> sweep for threads awaiting a reply.

All outgoing actions go into the PendingActionQueue — nothing is sent or
scheduled without human review. Per-thread failures are logged and counted
but never abort the run.
"""

from __future__ import annotations

from datetime import datetime

from .agents import (
    DraftAgent,
    SchedulerAgent,
    SummarizerAgent,
    TriageAgent,
    find_awaiting_response,
)
from .config import Settings, get_settings
from .hitl import PendingActionQueue
from .llm import LLM, LLMError
from .memory.long_term import MemoryStore
from .memory.short_term import RunState
from .observability import MetricsRegistry, bind_run, get_logger
from .schemas import ActionType, EmailThread, Route
from .tools.mail import MailProvider


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        mail: MailProvider,
        memory: MemoryStore,
        queue: PendingActionQueue,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.mail = mail
        self.memory = memory
        self.queue = queue
        self.metrics: MetricsRegistry = getattr(llm, "metrics", None) or MetricsRegistry()

        self.triage = TriageAgent(llm, self._settings)
        self.summarizer = SummarizerAgent(llm, self._settings)
        self.drafter = DraftAgent(llm, self._settings)
        self.scheduler = SchedulerAgent(llm, self._settings)

        self.state = RunState()
        self._log = get_logger(component="orchestrator")

    # -- run loop -----------------------------------------------------------------

    def run(self, max_threads: int | None = None, now: datetime | None = None) -> dict:
        bind_run(self.state.run_id)
        threads = self.mail.fetch_unread()
        if max_threads is not None:
            threads = threads[:max_threads]
        self._log.info("run_started", unread=len(threads))

        for thread in threads:
            try:
                self._process_thread(thread)
            except LLMError as exc:
                self.metrics.error("thread_failed")
                self._log.error("thread_failed", thread_id=thread.id, error=str(exc))

        self._followup_sweep(now)

        report = {
            "run_id": self.state.run_id,
            "unread_processed": len(threads),
            "pending_actions": len(self.queue.pending()),
            "counters": dict(self.metrics.counters),
        }
        self._log.info("run_finished", **{k: v for k, v in report.items() if k != "counters"})
        return report

    # -- per-thread processing -------------------------------------------------------

    def _process_thread(self, thread: EmailThread) -> None:
        ctx = self.state.context_for(thread.id)
        triage = self.triage.triage(thread)
        ctx.triage = triage
        self.metrics.incr("threads_processed")

        if triage.labels:
            self.mail.apply_labels(thread.id, triage.labels)

        if triage.route == Route.archive:
            self.mail.archive(thread.id)
            self.metrics.incr("threads_archived")
            return

        if triage.route == Route.no_action:
            self.mail.mark_read(thread.id)
            self.metrics.incr("threads_no_action")
            return

        if triage.route == Route.needs_reply:
            if not self.queue.has_action_for(thread.id, ActionType.send_reply):
                summary = None
                if len(thread.messages) > 1:
                    summary = self.summarizer.summarize(thread)
                    ctx.summary = summary
                draft = self.drafter.draft(thread, self.memory, summary)
                ctx.draft = draft
                self.queue.add(ActionType.send_reply, draft)
                self.metrics.incr("replies_drafted")
            else:
                self._log.info("draft_skipped_duplicate", thread_id=thread.id)
            self.mail.mark_read(thread.id)
            return

        # Route.follow_up
        if not self.queue.has_action_for(thread.id, ActionType.create_followup):
            proposal = self.scheduler.propose_followup(thread, reason=triage.rationale)
            ctx.followup = proposal
            self.queue.add(ActionType.create_followup, proposal)
            self.metrics.incr("followups_proposed")
        else:
            self._log.info("followup_skipped_duplicate", thread_id=thread.id)
        self.mail.mark_read(thread.id)

    # -- awaiting-reply sweep -----------------------------------------------------------

    def _followup_sweep(self, now: datetime | None = None) -> None:
        waiting = find_awaiting_response(
            self.mail.list_threads(), self._settings.user_email, now=now
        )
        for thread in waiting:
            if self.queue.has_action_for(thread.id, ActionType.create_followup):
                continue
            try:
                proposal = self.scheduler.propose_followup(
                    thread,
                    reason=f"awaiting a reply since {thread.last_message.date:%Y-%m-%d}",
                    now=now,
                )
            except LLMError as exc:
                self.metrics.error("followup_failed")
                self._log.error("followup_failed", thread_id=thread.id, error=str(exc))
                continue
            self.state.context_for(thread.id).followup = proposal
            self.queue.add(ActionType.create_followup, proposal)
            self.metrics.incr("followups_proposed")
