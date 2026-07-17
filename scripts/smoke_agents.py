"""Live smoke test of all four agents — makes ~4 real Claude API calls.

Requires ANTHROPIC_API_KEY (in .env). Flow:
  1. Triage t-004 (newsletter) + t-006 (spam)  -> rule pre-filter, NO API call
  2. Triage t-002 (client question)            -> LLM classification
  3. Summarize t-009 (4-message onboarding)    -> LLM
  4. Draft reply for t-002 with seeded memory  -> LLM (memory-grounded)
  5. Follow-up proposal for t-005 (vendor)     -> LLM

Prints each schema-validated result plus the aggregated metrics report.

Usage:  python scripts/smoke_agents.py
"""

import sys

from email_assistant.agents import (
    DraftAgent,
    SchedulerAgent,
    SummarizerAgent,
    TriageAgent,
    find_awaiting_response,
)
from email_assistant.config import get_settings
from email_assistant.llm import ClaudeClient
from email_assistant.memory.long_term import MemoryStore
from email_assistant.observability import bind_run, get_logger, new_id, setup_logging
from email_assistant.tools.mail import MockMailProvider


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set (put it in .env) — skipping live calls.")
        return 1

    setup_logging(level="INFO")
    run_id = new_id("run")
    bind_run(run_id)
    get_logger(component="smoke_agents").info("smoke_started", model=settings.model)

    mail = MockMailProvider(settings.fixtures_path, user_email=settings.user_email)
    memory = MemoryStore(settings.memory_path, seed_path=settings.memory_seed_path)
    llm = ClaudeClient(settings=settings)

    triage = TriageAgent(llm, settings)
    summarizer = SummarizerAgent(llm, settings)
    drafter = DraftAgent(llm, settings)
    scheduler = SchedulerAgent(llm, settings)

    section("1) Rule pre-filter (no API call): t-004 newsletter, t-006 spam")
    for thread_id in ("t-004", "t-006"):
        result = triage.triage(mail.get_thread(thread_id))
        print(
            f"  {thread_id}: intent={result.intent.value} route={result.route.value} "
            f"({result.rationale})"
        )

    section("2) LLM triage: t-002 (client question about rate limits)")
    triage_result = triage.triage(mail.get_thread("t-002"))
    print(triage_result.model_dump_json(indent=2))

    section("3) Summary: t-009 (4-message onboarding thread)")
    summary = summarizer.summarize(mail.get_thread("t-009"))
    print(summary.model_dump_json(indent=2))

    section("4) Memory-grounded draft: t-002")
    draft = drafter.draft(mail.get_thread("t-002"), memory)
    print(f"to:      {draft.to}")
    print(f"subject: {draft.subject}")
    print(f"tone:    {draft.tone.value}")
    print(f"--- body ---\n{draft.body}\n--- end body ---")
    print(f"rationale: {draft.rationale}")

    section("5) Follow-up proposal: awaiting-reply detection + scheduler")
    waiting = find_awaiting_response(mail.list_threads(), settings.user_email)
    print(f"awaiting-reply threads: {[t.id for t in waiting]}")
    if waiting:
        proposal = scheduler.propose_followup(
            waiting[0], reason="vendor has not replied; finance deadline July 25"
        )
        print(proposal.model_dump_json(indent=2))

    section("Metrics")
    print(llm.metrics.render_report())
    print(f"\nDone. run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
