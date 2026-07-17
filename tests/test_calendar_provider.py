from datetime import datetime, timezone

from email_assistant.schemas import FollowUpKind, FollowUpProposal
from email_assistant.tools.calendar import MockCalendarProvider

PROPOSAL = FollowUpProposal(
    thread_id="t-005",
    kind=FollowUpKind.reminder,
    title="Chase DataCloud quote",
    note="No reply since July 10; finance needs it by July 25.",
    due_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
)


def test_create_item_from_proposal():
    calendar = MockCalendarProvider()
    event = calendar.create_item(PROPOSAL)

    assert event.thread_id == "t-005"
    assert event.kind == FollowUpKind.reminder
    assert event.due_at == PROPOSAL.due_at
    assert calendar.list_items() == [event]


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "calendar.json"
    MockCalendarProvider(path).create_item(PROPOSAL)

    reloaded = MockCalendarProvider(path)
    items = reloaded.list_items()
    assert len(items) == 1
    assert items[0].title == "Chase DataCloud quote"
