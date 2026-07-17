import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from email_assistant.schemas import (
    ActionStatus,
    ActionType,
    EmailThread,
    FollowUpKind,
    FollowUpProposal,
    Intent,
    PendingAction,
    Priority,
    ReplyDraft,
    Route,
    TriageResult,
)
from tests.conftest import INBOX


def test_inbox_fixture_validates_against_schema():
    raw = json.loads(INBOX.read_text(encoding="utf-8"))
    threads = [EmailThread.model_validate(t) for t in raw["threads"]]

    assert len(threads) >= 8
    ids = [t.id for t in threads]
    assert len(ids) == len(set(ids)), "thread ids must be unique"
    for thread in threads:
        assert thread.messages, thread.id
        assert all(m.thread_id == thread.id for m in thread.messages)


def test_last_message_is_latest_by_date():
    raw = json.loads(INBOX.read_text(encoding="utf-8"))
    thread = EmailThread.model_validate(
        next(t for t in raw["threads"] if t["id"] == "t-009")
    )
    assert thread.last_message.id == "m-009-4"


def test_triage_confidence_is_bounded():
    with pytest.raises(ValidationError):
        TriageResult(
            thread_id="t-001",
            intent=Intent.request,
            priority=Priority.urgent,
            route=Route.needs_reply,
            confidence=1.5,
        )


def test_pending_action_reply_payload_roundtrip():
    draft = ReplyDraft(
        thread_id="t-002",
        to=["dana.reyes@northwindlabs.com"],
        subject="Re: Question about API rate limits",
        body="Hi Dana, ...",
    )
    action = PendingAction(type=ActionType.send_reply, payload=draft)
    assert action.status == ActionStatus.pending

    restored = PendingAction.model_validate(action.model_dump(mode="json"))
    assert isinstance(restored.payload, ReplyDraft)
    assert restored.payload.thread_id == "t-002"


def test_pending_action_followup_payload_roundtrip():
    proposal = FollowUpProposal(
        thread_id="t-005",
        kind=FollowUpKind.reminder,
        title="Chase DataCloud quote",
        due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )
    action = PendingAction(type=ActionType.create_followup, payload=proposal)

    restored = PendingAction.model_validate(action.model_dump(mode="json"))
    assert isinstance(restored.payload, FollowUpProposal)
    assert restored.payload.kind == FollowUpKind.reminder
