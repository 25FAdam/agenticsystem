"""Mail tool: provider interface + mock implementation.

The orchestrator and agents only ever see the MailProvider protocol, so a real
Gmail / MS Graph adapter is a drop-in replacement for the mock.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, Sequence

from ..observability import new_id
from ..schemas import EmailMessage, EmailThread, ReplyDraft


class MailProvider(Protocol):
    def list_threads(self) -> list[EmailThread]: ...

    def fetch_unread(self) -> list[EmailThread]: ...

    def get_thread(self, thread_id: str) -> EmailThread: ...

    def send_reply(self, draft: ReplyDraft) -> EmailMessage: ...

    def apply_labels(self, thread_id: str, labels: Sequence[str]) -> None: ...

    def archive(self, thread_id: str) -> None: ...

    def mark_read(self, thread_id: str) -> None: ...


class MockMailProvider:
    """In-memory mailbox seeded from a JSON fixture.

    Never writes back to the fixture file, so every instantiation starts from
    the same inbox state.
    """

    def __init__(self, fixture_path: Path | str, user_email: str, user_name: str = "") -> None:
        raw = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        self._threads: dict[str, EmailThread] = {
            t["id"]: EmailThread.model_validate(t) for t in raw["threads"]
        }
        self.user_email = user_email
        self.user_name = user_name
        self.sent: list[EmailMessage] = []

    # -- reads ---------------------------------------------------------------

    def list_threads(self) -> list[EmailThread]:
        return list(self._threads.values())

    def fetch_unread(self) -> list[EmailThread]:
        unread = [t for t in self._threads.values() if t.unread and not t.archived]
        return sorted(unread, key=lambda t: t.last_message.date, reverse=True)

    def get_thread(self, thread_id: str) -> EmailThread:
        if thread_id not in self._threads:
            raise KeyError(f"unknown thread: {thread_id}")
        return self._threads[thread_id]

    # -- writes --------------------------------------------------------------

    def send_reply(self, draft: ReplyDraft) -> EmailMessage:
        thread = self.get_thread(draft.thread_id)
        # A reply is always the newest message in its thread, even when fixture
        # dates are ahead of the wall clock.
        sent_at = max(
            datetime.now(timezone.utc),
            thread.last_message.date + timedelta(seconds=1),
        )
        message = EmailMessage(
            id=new_id("msg"),
            thread_id=draft.thread_id,
            sender=self.user_email,
            to=list(draft.to),
            date=sent_at,
            subject=draft.subject,
            body=draft.body,
        )
        thread.messages.append(message)
        self.sent.append(message)
        return message

    def apply_labels(self, thread_id: str, labels: Sequence[str]) -> None:
        thread = self.get_thread(thread_id)
        for label in labels:
            if label not in thread.labels:
                thread.labels.append(label)

    def archive(self, thread_id: str) -> None:
        thread = self.get_thread(thread_id)
        thread.archived = True
        thread.unread = False

    def mark_read(self, thread_id: str) -> None:
        self.get_thread(thread_id).unread = False
