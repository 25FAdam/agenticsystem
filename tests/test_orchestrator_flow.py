"""Full run-loop integration test: FakeLLM + mock providers, no network."""

from datetime import datetime, timezone

import pytest

from email_assistant.config import Settings
from email_assistant.hitl import PendingActionQueue
from email_assistant.orchestrator import Orchestrator
from email_assistant.schemas import (
    ActionType,
    FollowUpKind,
    FollowUpProposal,
    Intent,
    Priority,
    ReplyDraft,
    Route,
    ThreadSummary,
    TriageResult,
)
from tests.conftest import FakeLLM

NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)


def triage(thread_id, intent, route, priority=Priority.normal, labels=()):
    return TriageResult(
        thread_id=thread_id,
        intent=intent,
        priority=priority,
        labels=list(labels),
        route=route,
        confidence=0.9,
        rationale="canned",
    )


def draft(thread_id):
    return ReplyDraft(thread_id=thread_id, to=[], subject="Re: x", body="canned body")


def followup(thread_id):
    return FollowUpProposal(
        thread_id=thread_id,
        kind=FollowUpKind.reminder,
        title="canned follow-up",
        due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def orchestrator(mail, seeded_store, tmp_path):
    # Unread threads arrive newest-first: t-002, t-001, [t-004 rule],
    # [t-006 rule], t-009, t-003, t-007, t-008. Canned triage results must
    # follow that order for the LLM-classified ones.
    fake = FakeLLM(
        [
            triage("t-002", Intent.question, Route.needs_reply, Priority.high, ["client"]),
            triage("t-001", Intent.request, Route.needs_reply, Priority.urgent),
            triage("t-009", Intent.fyi, Route.no_action),
            triage("t-003", Intent.meeting, Route.follow_up),
            triage("t-007", Intent.fyi, Route.no_action, Priority.low),
            triage("t-008", Intent.billing, Route.needs_reply, Priority.high),
            # t-002 has 3 messages -> summarized before drafting
            ThreadSummary(thread_id="t-002", tl_dr="canned summary"),
            # drafts pop in processing order: t-002, t-001, t-008
            draft("t-002"),
            draft("t-001"),
            draft("t-008"),
            # follow-ups: t-003 (route) + t-005 (awaiting-reply sweep)
            followup("t-003"),
            followup("t-005"),
        ]
    )
    queue = PendingActionQueue(tmp_path / "actions.json")
    return Orchestrator(
        llm=fake, mail=mail, memory=seeded_store, queue=queue, settings=Settings()
    )


def test_full_run(orchestrator, mail):
    report = orchestrator.run(now=NOW)

    assert report["unread_processed"] == 8
    assert report["pending_actions"] == 5

    counters = report["counters"]
    assert counters["threads_processed"] == 8
    assert counters["threads_archived"] == 2  # newsletter + spam via rules
    assert counters["threads_no_action"] == 2
    assert counters["replies_drafted"] == 3
    assert counters["followups_proposed"] == 2

    # queue contents
    by_type = {}
    for action in orchestrator.queue.all():
        by_type.setdefault(action.type, []).append(action.payload.thread_id)
    assert sorted(by_type[ActionType.send_reply]) == ["t-001", "t-002", "t-008"]
    assert sorted(by_type[ActionType.create_followup]) == ["t-003", "t-005"]

    # mail side effects
    assert mail.get_thread("t-004").archived
    assert "newsletter" in mail.get_thread("t-004").labels
    assert mail.get_thread("t-006").archived
    assert not mail.get_thread("t-007").unread
    assert "client" in mail.get_thread("t-002").labels

    # short-term state captured the pipeline artifacts
    ctx = orchestrator.state.contexts["t-002"]
    assert ctx.triage is not None
    assert ctx.summary is not None
    assert ctx.draft is not None
    # single-message thread skipped summarization
    assert orchestrator.state.contexts["t-001"].summary is None

    # step tracing: every agent invocation became an AgentStep span
    agents = [s.agent for s in orchestrator.steps]
    assert agents.count("triage") == 8
    assert agents.count("summarizer") == 1
    assert agents.count("drafter") == 3
    assert agents.count("scheduler") == 2
    assert all(s.run_id == orchestrator.state.run_id for s in orchestrator.steps)
    assert all(not s.error for s in orchestrator.steps)
    drafter_steps = [s for s in orchestrator.steps if s.agent == "drafter"]
    assert all("draft to" in s.output_summary for s in drafter_steps)
    assert len(report["steps"]) == 14


def test_second_run_is_idempotent(orchestrator, mail):
    orchestrator.run(now=NOW)
    before = len(orchestrator.queue.all())

    # FakeLLM has no canned responses left: any LLM call would raise.
    report = orchestrator.run(now=NOW)

    assert report["unread_processed"] == 0
    assert len(orchestrator.queue.all()) == before  # sweep deduped t-005


def test_thread_failure_does_not_abort_run(mail, seeded_store, tmp_path):
    from email_assistant.llm import LLMParseError

    class FailingLLM(FakeLLM):
        def structured(self, **kwargs):
            raise LLMParseError("boom")

    orchestrator = Orchestrator(
        llm=FailingLLM(),
        mail=mail,
        memory=seeded_store,
        queue=PendingActionQueue(tmp_path / "actions.json"),
        settings=Settings(),
    )
    report = orchestrator.run(now=NOW)

    # Rule-triaged threads (newsletter, spam) still succeed without the LLM.
    assert report["counters"]["threads_archived"] == 2
    assert report["counters"]["threads_processed"] == 2
    assert orchestrator.metrics.errors["thread_failed"] == 6
    assert report["pending_actions"] == 0

    # failed invocations are captured as error steps:
    # 6 LLM triages + 1 scheduler call from the awaiting-reply sweep
    error_steps = [s for s in orchestrator.steps if s.error]
    assert len(error_steps) == 7
    assert all("LLMParseError" in s.error for s in error_steps)
    assert orchestrator.metrics.errors["followup_failed"] == 1


def test_write_report_creates_artifact(tmp_path):
    from email_assistant.orchestrator import write_report

    report = {"run_id": "run_test123", "counters": {"threads_processed": 2}}
    path = write_report(report, tmp_path / "runs")

    assert path.name == "run_test123.json"
    import json

    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["counters"]["threads_processed"] == 2
