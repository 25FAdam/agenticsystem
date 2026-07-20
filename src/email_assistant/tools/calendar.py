"""Calendar/task tool: provider interface + mock implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..observability import new_id
from ..schemas import CalendarEvent, FollowUpProposal


class CalendarProvider(Protocol):
    def create_item(self, proposal: FollowUpProposal) -> CalendarEvent: ...

    def list_items(self) -> list[CalendarEvent]: ...


class MockCalendarProvider:
    """In-memory calendar, optionally persisted to a JSON file."""

    def __init__(self, persist_path: Path | str | None = None) -> None:
        self._path = Path(persist_path) if persist_path else None
        self._items: list[CalendarEvent] = []
        if self._path and self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = [CalendarEvent.model_validate(i) for i in raw["items"]]

    def create_item(self, proposal: FollowUpProposal) -> CalendarEvent:
        event = CalendarEvent(
            id=new_id("cal"),
            thread_id=proposal.thread_id,
            kind=proposal.kind,
            title=proposal.title,
            note=proposal.note,
            due_at=proposal.due_at,
        )
        self._items.append(event)
        self._save()
        return event

    def list_items(self) -> list[CalendarEvent]:
        return list(self._items)

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [i.model_dump(mode="json") for i in self._items]}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
