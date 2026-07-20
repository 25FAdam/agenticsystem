from datetime import datetime, timezone

import pytest

from email_assistant.hitl import PendingActionQueue, execute_action, record_feedback
from email_assistant.schemas import (
    ActionStatus,
    ActionType,
    FollowUpKind,
    FollowUpProposal,
    MemoryKind,
    ReplyDraft,
)
from email_assistant.tools.calendar import MockCalendarProvider

DRAFT = ReplyDraft(
    thread_id="t-002",
    to=["dana.reyes@northwindlabs.com"],
    subject="Re: Question about API rate limits",
    body="Hi Dana, ...",
)

PROPOSAL = FollowUpProposal(
    thread_id="t-005",
    kind=FollowUpKind.reminder,
    title="Chase DataCloud quote",
    due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
)


@pytest.fixture
def queue(tmp_path):
    return PendingActionQueue(tmp_path / "actions.json")


def test_add_persists_across_instances(tmp_path, queue):
    action = queue.add(ActionType.send_reply, DRAFT)

    reloaded = PendingActionQueue(tmp_path / "actions.json")
    restored = reloaded.get(action.id)
    assert restored.status == ActionStatus.pending
    assert isinstance(restored.payload, ReplyDraft)


def test_approve_sets_status_and_decided_at(queue):
    action = queue.add(ActionType.send_reply, DRAFT)
    queue.approve(action.id)

    action = queue.get(action.id)
    assert action.status == ActionStatus.approved
    assert action.decided_at is not None


def test_reject_stores_note(queue):
    action = queue.add(ActionType.send_reply, DRAFT)
    queue.reject(action.id, note="too formal")
    assert queue.get(action.id).user_note == "too formal"
    assert queue.pending() == []


def test_edit_replaces_payload(queue):
    action = queue.add(ActionType.send_reply, DRAFT)
    new_payload = DRAFT.model_copy(update={"body": "Shorter body."})
    queue.edit(action.id, new_payload, note="shortened")

    edited = queue.get(action.id)
    assert edited.status == ActionStatus.edited
    assert edited.payload.body == "Shorter body."


def test_invalid_transitions_raise(queue):
    action = queue.add(ActionType.send_reply, DRAFT)
    queue.approve(action.id)

    with pytest.raises(ValueError):
        queue.approve(action.id)  # already approved
    with pytest.raises(ValueError):
        queue.reject(action.id)  # only pending can be rejected

    other = queue.add(ActionType.create_followup, PROPOSAL)
    with pytest.raises(ValueError):
        queue.mark_executed(other.id)  # pending cannot execute


def test_has_action_for_dedupes_active_but_not_rejected(queue):
    action = queue.add(ActionType.send_reply, DRAFT)
    assert queue.has_action_for("t-002", ActionType.send_reply)
    assert not queue.has_action_for("t-002", ActionType.create_followup)

    queue.reject(action.id)
    assert not queue.has_action_for("t-002", ActionType.send_reply)


def test_execute_send_reply(queue, mail):
    action = queue.add(ActionType.send_reply, DRAFT)
    queue.approve(action.id)

    description = execute_action(queue.get(action.id), mail, MockCalendarProvider())
    queue.mark_executed(action.id)

    assert "reply sent" in description
    assert len(mail.sent) == 1
    assert queue.get(action.id).status == ActionStatus.executed


def test_execute_create_followup(queue, mail):
    calendar = MockCalendarProvider()
    action = queue.add(ActionType.create_followup, PROPOSAL)
    queue.approve(action.id)

    description = execute_action(queue.get(action.id), mail, calendar)

    assert "reminder created" in description
    assert len(calendar.list_items()) == 1


def test_record_feedback_writes_memory(queue, store):
    action = queue.add(ActionType.send_reply, DRAFT)
    queue.reject(action.id, "too formal")

    record = record_feedback(store, queue.get(action.id), "rejected", "too formal")

    assert record is not None
    assert record.kind == MemoryKind.feedback
    assert "too formal" in record.value
    assert store.recall(kind=MemoryKind.feedback)


def test_record_feedback_ignores_approvals(queue, store):
    action = queue.add(ActionType.send_reply, DRAFT)
    assert record_feedback(store, action, "approved") is None
    assert store.recall(kind=MemoryKind.feedback) == []
