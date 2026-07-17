"""Command-line interface.

Commands:
  run     — process the inbox (triage -> summarize -> draft -> follow-ups)
  review  — approve / edit / reject pending actions; approved ones execute
  inbox   — list unread threads
  actions — list all recorded actions and their statuses
  memory  — inspect long-term memory records
"""

from __future__ import annotations

from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .hitl import PendingActionQueue, execute_action, record_feedback
from .memory.long_term import MemoryStore
from .observability import setup_logging
from .schemas import FollowUpProposal, MemoryKind, PendingAction, ReplyDraft
from .tools.calendar import MockCalendarProvider
from .tools.mail import MockMailProvider

app = typer.Typer(
    help="Agentic email assistant (assessment prototype).", no_args_is_help=True
)
console = Console()


def _build(settings: Settings):
    mail = MockMailProvider(
        settings.fixtures_path,
        user_email=settings.user_email,
        user_name=settings.user_name,
    )
    calendar = MockCalendarProvider(settings.calendar_path)
    memory = MemoryStore(settings.memory_path, seed_path=settings.memory_seed_path)
    queue = PendingActionQueue(settings.actions_path)
    return mail, calendar, memory, queue


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    max_threads: int = typer.Option(
        None, "--max-threads", help="Process at most N unread threads."
    ),
) -> None:
    """Process the inbox and queue outgoing actions for review."""
    settings = Settings()
    setup_logging(settings.log_level)
    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY is not set — put it in .env[/red]")
        raise typer.Exit(1)

    from .llm import ClaudeClient
    from .orchestrator import Orchestrator

    mail, _calendar, memory, queue = _build(settings)
    llm = ClaudeClient(settings=settings)
    orchestrator = Orchestrator(
        llm=llm, mail=mail, memory=memory, queue=queue, settings=settings
    )
    report = orchestrator.run(max_threads=max_threads)

    triage_table = Table(title=f"Triage — run {report['run_id']}")
    for column in ("thread", "intent", "priority", "route", "labels"):
        triage_table.add_column(column)
    for ctx in orchestrator.state.contexts.values():
        if ctx.triage is None:
            continue
        triage_table.add_row(
            ctx.thread_id,
            ctx.triage.intent.value,
            ctx.triage.priority.value,
            ctx.triage.route.value,
            ", ".join(ctx.triage.labels),
        )
    console.print(triage_table)

    pending = queue.pending()
    console.print(
        f"\n[bold]{len(pending)} action(s) awaiting review[/bold] — run "
        "[cyan]assistant review[/cyan] to approve, edit or reject them.\n"
    )
    console.print(llm.metrics.render_report())


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


def _show_action(index: int, total: int, action: PendingAction) -> None:
    payload = action.payload
    if isinstance(payload, ReplyDraft):
        body = (
            f"[bold]to:[/bold] {', '.join(payload.to)}\n"
            f"[bold]subject:[/bold] {payload.subject}\n"
            f"[bold]tone:[/bold] {payload.tone.value}\n\n"
            f"{payload.body}\n\n"
            f"[dim]rationale: {payload.rationale}[/dim]"
        )
        title = f"[{index}/{total}] send_reply — thread {payload.thread_id}"
    else:
        body = (
            f"[bold]kind:[/bold] {payload.kind.value}\n"
            f"[bold]title:[/bold] {payload.title}\n"
            f"[bold]due:[/bold] {payload.due_at:%Y-%m-%d %H:%M} UTC\n"
            f"[bold]note:[/bold] {payload.note or '(none)'}"
        )
        title = f"[{index}/{total}] create_followup — thread {payload.thread_id}"
    console.print(Panel(body, title=title))


def _edit_payload(action: PendingAction) -> tuple[ReplyDraft | FollowUpProposal, str]:
    payload = action.payload
    if isinstance(payload, ReplyDraft):
        edited = typer.edit(payload.body)
        new_body = edited.strip() if edited and edited.strip() else payload.body
        note = typer.prompt(
            "What changed / why? (stored as feedback for future drafts)", default=""
        )
        return payload.model_copy(update={"body": new_body}), note

    due_raw = typer.prompt(
        f"New due date (ISO 8601, empty keeps {payload.due_at:%Y-%m-%d %H:%M})",
        default="",
    )
    title = typer.prompt("New title (empty keeps current)", default="")
    updates: dict = {}
    if due_raw.strip():
        updates["due_at"] = datetime.fromisoformat(due_raw.strip())
    if title.strip():
        updates["title"] = title.strip()
    note = typer.prompt("Why the change? (stored as feedback)", default="")
    return payload.model_copy(update=updates), note


def _execute(queue: PendingActionQueue, action_id: str, mail, calendar) -> None:
    action = queue.get(action_id)
    try:
        description = execute_action(action, mail, calendar)
    except Exception as exc:  # noqa: BLE001 — surface any tool failure to the user
        queue.mark_failed(action_id, str(exc))
        console.print(f"[red]Execution failed: {exc}[/red]")
        return
    queue.mark_executed(action_id)
    console.print(f"[green]Executed: {description}[/green]")


@app.command()
def review() -> None:
    """Review pending actions: approve [a], edit [e], reject [r], skip [s]."""
    settings = Settings()
    setup_logging(settings.log_level)
    mail, calendar, memory, queue = _build(settings)

    pending = queue.pending()
    if not pending:
        console.print("No pending actions.")
        return

    for index, action in enumerate(pending, start=1):
        _show_action(index, len(pending), action)
        choice = (
            typer.prompt("approve [a] / edit [e] / reject [r] / skip [s]", default="s")
            .strip()
            .lower()[:1]
        )
        if choice == "a":
            queue.approve(action.id)
            _execute(queue, action.id, mail, calendar)
        elif choice == "e":
            new_payload, note = _edit_payload(action)
            queue.edit(action.id, new_payload, note)
            record_feedback(memory, queue.get(action.id), "edited", note)
            _execute(queue, action.id, mail, calendar)
        elif choice == "r":
            note = typer.prompt(
                "Reason? (stored as feedback; empty to skip)", default=""
            )
            queue.reject(action.id, note)
            record_feedback(memory, queue.get(action.id), "rejected", note)
            console.print("[yellow]Rejected.[/yellow]")
        else:
            console.print("Skipped.")

    console.print("\nReview finished.")


# ---------------------------------------------------------------------------
# inspection commands
# ---------------------------------------------------------------------------


@app.command()
def inbox() -> None:
    """List unread threads in the mock inbox."""
    settings = Settings()
    mail, _, _, _ = _build(settings)
    table = Table(title="Unread threads")
    for column in ("thread", "last message", "from", "subject", "labels"):
        table.add_column(column)
    for thread in mail.fetch_unread():
        last = thread.last_message
        table.add_row(
            thread.id,
            f"{last.date:%m-%d %H:%M}",
            last.sender,
            thread.subject,
            ", ".join(thread.labels),
        )
    console.print(table)


@app.command()
def actions() -> None:
    """List all recorded actions and their statuses."""
    settings = Settings()
    queue = PendingActionQueue(settings.actions_path)
    items = queue.all()
    if not items:
        console.print("No actions recorded.")
        return
    table = Table(title="Actions")
    for column in ("id", "type", "thread", "status", "created", "note"):
        table.add_column(column)
    for action in items:
        table.add_row(
            action.id[:8],
            action.type.value,
            action.payload.thread_id,
            action.status.value,
            f"{action.created_at:%m-%d %H:%M}",
            action.user_note,
        )
    console.print(table)


@app.command()
def memory(
    kind: str = typer.Option(None, "--kind", help="preference|contact|org_fact|feedback"),
) -> None:
    """Inspect long-term memory records."""
    settings = Settings()
    store = MemoryStore(settings.memory_path, seed_path=settings.memory_seed_path)
    records = store.recall(kind=MemoryKind(kind) if kind else None)
    if not records:
        console.print("No memory records.")
        return
    table = Table(title=f"Memory ({len(records)} records)")
    for column in ("kind", "key", "value", "source", "updated"):
        table.add_column(column)
    for record in records:
        value = record.value if len(record.value) <= 80 else record.value[:77] + "..."
        table.add_row(
            record.kind.value,
            record.key,
            value,
            record.source,
            f"{record.updated_at:%m-%d %H:%M}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
