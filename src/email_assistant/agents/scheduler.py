"""Scheduler agent: propose follow-ups and reminders.

Two parts:
- ``find_awaiting_response`` — deterministic detection of threads where the
  user sent the last message and has been waiting too long (no LLM needed);
- ``SchedulerAgent`` — turns a thread + reason into a concrete, schema-valid
  FollowUpProposal (kind, due time, title, note).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from ..schemas import EmailThread, FollowUpProposal
from .base import Agent, render_thread

SCHEDULER_SYSTEM_TEMPLATE = """\
You schedule follow-ups and reminders for {user_name} <{user_email}> ("the user").

Given a thread and a reason, propose exactly ONE follow-up:
- kind: "reminder" for chase-ups and tasks, "event" for calendar meetings.
- due_at: ISO 8601 with timezone. Prefer working hours (09:00-17:00 UTC).
  Respect deadlines mentioned in the thread; for chase-ups with no deadline
  use 2-3 business days from the current time.
- title: short imperative, e.g. "Chase DataCloud quote".
- note: 1-2 sentences of context so the user remembers why this exists.
thread_id: echo the given thread_id exactly."""


def find_awaiting_response(
    threads: Iterable[EmailThread],
    user_email: str,
    now: datetime | None = None,
    min_age_days: int = 3,
) -> list[EmailThread]:
    """Threads where the user sent the last message >= min_age_days ago."""
    now = now or datetime.now(timezone.utc)
    cutoff = timedelta(days=min_age_days)
    waiting = [
        t
        for t in threads
        if not t.archived
        and t.last_message.sender.lower() == user_email.lower()
        and (now - t.last_message.date) >= cutoff
    ]
    return sorted(waiting, key=lambda t: t.last_message.date)


class SchedulerAgent(Agent):
    name = "scheduler"

    def propose_followup(
        self,
        thread: EmailThread,
        reason: str = "",
        now: datetime | None = None,
    ) -> FollowUpProposal:
        now = now or datetime.now(timezone.utc)
        system = SCHEDULER_SYSTEM_TEMPLATE.format(
            user_name=self._settings.user_name,
            user_email=self._settings.user_email,
        )
        user = (
            f"current time: {now.isoformat()}\n"
            f"reason for the follow-up: {reason or 'unspecified'}\n\n"
            f"THREAD:\n{render_thread(thread)}"
        )
        result = self._call(FollowUpProposal, system, user)
        result = self._fix_thread_id(result, thread)
        self._log.info(
            "followup_proposed",
            thread_id=thread.id,
            kind=result.kind.value,
            due_at=result.due_at.isoformat(),
            title=result.title,
        )
        return result
