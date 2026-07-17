from datetime import datetime, timezone

from email_assistant.agents import (
    DraftAgent,
    SchedulerAgent,
    SummarizerAgent,
    TriageAgent,
    find_awaiting_response,
)
from email_assistant.config import Settings
from email_assistant.schemas import (
    FollowUpKind,
    FollowUpProposal,
    Intent,
    MemoryKind,
    Priority,
    ReplyDraft,
    Route,
    ThreadSummary,
    TriageResult,
)
from tests.conftest import USER_EMAIL, FakeLLM

NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)


def canned_triage(thread_id="t-001"):
    return TriageResult(
        thread_id=thread_id,
        intent=Intent.request,
        priority=Priority.urgent,
        labels=["incident"],
        route=Route.needs_reply,
        confidence=0.95,
        rationale="canned",
    )


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


class TestTriageAgent:
    def test_newsletter_is_classified_by_rule_without_llm(self, mail):
        fake = FakeLLM()
        result = TriageAgent(fake).triage(mail.get_thread("t-004"))

        assert result.intent == Intent.newsletter
        assert result.route == Route.archive
        assert result.confidence == 1.0
        assert fake.calls == []

    def test_spam_is_classified_by_rule_without_llm(self, mail):
        fake = FakeLLM()
        result = TriageAgent(fake).triage(mail.get_thread("t-006"))

        assert result.intent == Intent.spam
        assert result.route == Route.archive
        assert fake.calls == []

    def test_normal_email_goes_to_llm(self, mail):
        fake = FakeLLM([canned_triage()])
        result = TriageAgent(fake).triage(mail.get_thread("t-001"))

        assert result.route == Route.needs_reply
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["name"] == "triage"
        assert "postmortem" in call["user"]
        assert USER_EMAIL in call["system"]

    def test_uses_dedicated_triage_model_when_configured(self, mail):
        fake = FakeLLM([canned_triage()])
        settings = Settings(triage_model="claude-haiku-4-5")
        TriageAgent(fake, settings=settings).triage(mail.get_thread("t-001"))

        assert fake.calls[0]["model"] == "claude-haiku-4-5"

    def test_wrong_thread_id_from_llm_is_corrected(self, mail):
        fake = FakeLLM([canned_triage(thread_id="t-999")])
        result = TriageAgent(fake).triage(mail.get_thread("t-001"))
        assert result.thread_id == "t-001"


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class TestSummarizerAgent:
    def test_prompt_contains_all_messages(self, mail):
        fake = FakeLLM([ThreadSummary(thread_id="t-009", tl_dr="canned")])
        SummarizerAgent(fake).summarize(mail.get_thread("t-009"))

        prompt = fake.calls[0]["user"]
        # one detail from each of the four messages
        assert "Elena Sousa signed" in prompt
        assert "standard dev laptop" in prompt
        assert "Snowflake read approvals" in prompt
        assert "desk allocation" in prompt

    def test_thread_id_is_corrected(self, mail):
        fake = FakeLLM([ThreadSummary(thread_id="wrong", tl_dr="canned")])
        result = SummarizerAgent(fake).summarize(mail.get_thread("t-009"))
        assert result.thread_id == "t-009"


# ---------------------------------------------------------------------------
# Drafter
# ---------------------------------------------------------------------------


def canned_draft(to=None):
    return ReplyDraft(
        thread_id="t-002",
        to=to if to is not None else [],
        subject="Re: Question about API rate limits",
        body="canned body",
    )


class TestDraftAgent:
    def test_memory_context_is_injected_into_prompt(self, mail, seeded_store):
        fake = FakeLLM([canned_draft()])
        DraftAgent(fake).draft(mail.get_thread("t-002"), seeded_store)

        prompt = fake.calls[0]["user"]
        assert "MEMORY CONTEXT:" in prompt
        assert "friendly-professional" in prompt  # tone preference
        assert "Northwind Labs" in prompt  # contact fact about Dana
        assert "1000 requests/min" in prompt  # related org fact via search

    def test_feedback_records_are_included(self, mail, seeded_store):
        seeded_store.remember(
            MemoryKind.feedback,
            "draft_feedback_t-002",
            "Keep client replies under four sentences.",
            source="human_review",
        )
        fake = FakeLLM([canned_draft()])
        DraftAgent(fake).draft(mail.get_thread("t-002"), seeded_store)

        assert "Keep client replies under four sentences." in fake.calls[0]["user"]

    def test_empty_recipients_default_to_last_sender(self, mail, seeded_store):
        fake = FakeLLM([canned_draft(to=[])])
        result = DraftAgent(fake).draft(mail.get_thread("t-002"), seeded_store)
        assert result.to == ["dana.reyes@northwindlabs.com"]

    def test_summary_block_is_included_when_given(self, mail, seeded_store):
        summary = ThreadSummary(
            thread_id="t-002", tl_dr="Client asks about burst limits."
        )
        fake = FakeLLM([canned_draft()])
        DraftAgent(fake).draft(mail.get_thread("t-002"), seeded_store, summary=summary)

        prompt = fake.calls[0]["user"]
        assert "THREAD SUMMARY:" in prompt
        assert "Client asks about burst limits." in prompt


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_find_awaiting_response_detects_vendor_thread(self, mail):
        waiting = find_awaiting_response(mail.list_threads(), USER_EMAIL, now=NOW)
        assert [t.id for t in waiting] == ["t-005"]

    def test_find_awaiting_response_respects_min_age(self, mail):
        waiting = find_awaiting_response(
            mail.list_threads(), USER_EMAIL, now=NOW, min_age_days=10
        )
        assert waiting == []

    def test_find_awaiting_response_skips_archived(self, mail):
        mail.archive("t-005")
        waiting = find_awaiting_response(mail.list_threads(), USER_EMAIL, now=NOW)
        assert waiting == []

    def test_propose_followup(self, mail):
        proposal = FollowUpProposal(
            thread_id="t-005",
            kind=FollowUpKind.reminder,
            title="Chase DataCloud quote",
            due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
        )
        fake = FakeLLM([proposal])
        result = SchedulerAgent(fake).propose_followup(
            mail.get_thread("t-005"), reason="no reply for 7 days", now=NOW
        )

        assert result.kind == FollowUpKind.reminder
        prompt = fake.calls[0]["user"]
        assert "no reply for 7 days" in prompt
        assert "DataCloud license renewal" in prompt
        assert NOW.isoformat() in prompt
