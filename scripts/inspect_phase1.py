"""Manual inspection of the Phase 0-1 building blocks.

Runs fully offline — no API key needed. Exercises the mock inbox, the memory
store, the calendar mock and the observability layer, printing what each one
returns.

Usage:  python scripts/inspect_phase1.py
"""

from datetime import datetime, timedelta, timezone

from email_assistant.config import get_settings
from email_assistant.memory.long_term import MemoryStore
from email_assistant.observability import (
    MetricsRegistry,
    bind_run,
    get_logger,
    new_id,
    setup_logging,
)
from email_assistant.schemas import FollowUpKind, FollowUpProposal, MemoryKind, ReplyDraft
from email_assistant.tools.calendar import MockCalendarProvider
from email_assistant.tools.mail import MockMailProvider


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    settings = get_settings()

    # --- 0. Config + logging -------------------------------------------------
    section("0) Config + structured logging (goes to stderr as JSON)")
    print(f"model:            {settings.model}")
    print(f"triage model:     {settings.effective_triage_model}")
    print(f"user:             {settings.user_name} <{settings.user_email}>")
    print(f"fixtures:         {settings.fixtures_path}")
    print(f"memory (runtime): {settings.memory_path}  [gitignored]")

    setup_logging(level="INFO")
    run_id = new_id("run")
    bind_run(run_id)
    log = get_logger(component="inspect")
    log.info("inspection_started")  # this JSON line goes to stderr

    # --- 1. Mock inbox ---------------------------------------------------------
    section("1) Mock inbox — unread threads (newest first)")
    mail = MockMailProvider(
        settings.fixtures_path,
        user_email=settings.user_email,
        user_name=settings.user_name,
    )
    for thread in mail.fetch_unread():
        last = thread.last_message
        print(
            f"  {thread.id}  {last.date:%m-%d %H:%M}  "
            f"{last.sender:<35}  {thread.subject[:45]}"
        )
    print(f"\n  unread: {len(mail.fetch_unread())} / total: {len(mail.list_threads())}")
    t5 = mail.get_thread("t-005")
    print(
        f"  t-005 not listed: we sent the last message ({t5.last_message.date:%b %d}),"
        " awaiting reply -> follow-up candidate"
    )

    # --- 2. Mail mutations -------------------------------------------------------
    section("2) Mail actions — reply / labels / archive (in-memory only)")
    draft = ReplyDraft(
        thread_id="t-002",
        to=["dana.reyes@northwindlabs.com"],
        subject="Re: Question about API rate limits",
        body="Hi Dana,\n\nBurst raise for the backfill is doable...\n\nBest, Adam",
    )
    sent = mail.send_reply(draft)
    print(f"  reply sent:    {sent.id} -> {sent.to[0]}")
    print(f"  t-002 messages: {len(mail.get_thread('t-002').messages)} (was 3)")

    mail.apply_labels("t-002", ["api", "client"])
    print(f"  t-002 labels:  {mail.get_thread('t-002').labels} (no duplicates)")

    mail.archive("t-004")
    print(f"  t-004 archived, unread now: {len(mail.fetch_unread())}")
    print("  (fixture file untouched — a new provider would start fresh)")

    # --- 3. Long-term memory ------------------------------------------------------
    section("3) Memory store — seeded search + upsert")
    store = MemoryStore(settings.memory_path, seed_path=settings.memory_seed_path)
    print(f"  records loaded: {len(store)}")

    for query in ("what are the API rate limits?", "who is Dana?"):
        results = store.search(query, limit=2)
        print(f"\n  search({query!r}):")
        for r in results:
            print(f"    [{r.kind.value}] {r.key}: {r.value[:70]}...")

    store.remember(
        MemoryKind.feedback,
        "draft_feedback_t-002",
        "User shortened the draft — prefers 3-sentence replies to clients.",
        source="human_review",
    )
    print(f"\n  feedback saved, records now: {len(store)} (re-run: stays, upsert)")

    # --- 4. Calendar ---------------------------------------------------------------
    section("4) Calendar mock — follow-up from proposal")
    calendar = MockCalendarProvider()
    event = calendar.create_item(
        FollowUpProposal(
            thread_id="t-005",
            kind=FollowUpKind.reminder,
            title="Chase DataCloud quote",
            note="No reply since Jul 10; finance needs it by Jul 25.",
            due_at=datetime.now(timezone.utc) + timedelta(days=3),
        )
    )
    print(f"  created: {event.id}  [{event.kind.value}]  {event.title}  due {event.due_at:%b %d}")

    # --- 5. Metrics -----------------------------------------------------------------
    section("5) Metrics — what a run report will look like")
    metrics = MetricsRegistry()
    metrics.incr("emails_processed", 8)
    metrics.incr("replies_drafted", 3)
    metrics.observe("triage_latency", 140.0)
    metrics.observe("triage_latency", 95.0)
    metrics.add_tokens(input_tokens=4200, output_tokens=910)
    print(metrics.render_report())

    log.info("inspection_finished", unread_left=len(mail.fetch_unread()))
    print(f"\nDone. run_id={run_id} — the same id appears in every JSON log line.")


if __name__ == "__main__":
    main()
