"""Human-in-the-loop layer.

Every outgoing action (send a reply, create a follow-up) lands in the
PendingActionQueue instead of executing directly. A human approves, edits or
rejects each one; only approved/edited actions are executed. Edit/reject
decisions are written back into long-term memory as feedback records, which
the DraftAgent reads on later runs — that is the behavior-evolution loop.

The queue persists to JSON so `assistant run` and `assistant review` work
across separate processes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .memory.long_term import MemoryStore
from .observability import get_logger
from .schemas import (
    ActionStatus,
    ActionType,
    FollowUpProposal,
    MemoryKind,
    MemoryRecord,
    PendingAction,
    ReplyDraft,
)
from .tools.calendar import CalendarProvider
from .tools.mail import MailProvider

# Statuses that block adding a duplicate action for the same thread. A
# rejected action does NOT block: re-running after a rejection should produce
# a fresh proposal that incorporates the stored feedback.
ACTIVE_STATUSES = frozenset(
    {
        ActionStatus.pending,
        ActionStatus.approved,
        ActionStatus.edited,
        ActionStatus.executed,
    }
)


class PendingActionQueue:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._actions: list[PendingAction] = []
        self._log = get_logger(component="hitl")
        if self._path and self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._actions = [PendingAction.model_validate(a) for a in raw["actions"]]

    # -- adding / reading ------------------------------------------------------

    def add(
        self, action_type: ActionType, payload: ReplyDraft | FollowUpProposal
    ) -> PendingAction:
        action = PendingAction(type=action_type, payload=payload)
        self._actions.append(action)
        self._save()
        self._log.info(
            "action_queued",
            action_id=action.id,
            type=action_type.value,
            thread_id=payload.thread_id,
        )
        return action

    def get(self, action_id: str) -> PendingAction:
        for action in self._actions:
            if action.id == action_id:
                return action
        raise KeyError(f"unknown action: {action_id}")

    def all(self) -> list[PendingAction]:
        return list(self._actions)

    def pending(self) -> list[PendingAction]:
        return [a for a in self._actions if a.status == ActionStatus.pending]

    def has_action_for(self, thread_id: str, action_type: ActionType) -> bool:
        return any(
            a.type == action_type
            and a.payload.thread_id == thread_id
            and a.status in ACTIVE_STATUSES
            for a in self._actions
        )

    # -- decisions --------------------------------------------------------------

    def approve(self, action_id: str) -> PendingAction:
        return self._transition(action_id, {ActionStatus.pending}, ActionStatus.approved)

    def edit(
        self,
        action_id: str,
        new_payload: ReplyDraft | FollowUpProposal,
        note: str = "",
    ) -> PendingAction:
        action = self._require_status(action_id, {ActionStatus.pending})
        action.payload = new_payload
        return self._decide(action, ActionStatus.edited, note)

    def reject(self, action_id: str, note: str = "") -> PendingAction:
        action = self._require_status(action_id, {ActionStatus.pending})
        return self._decide(action, ActionStatus.rejected, note)

    def mark_executed(self, action_id: str) -> PendingAction:
        return self._transition(
            action_id, {ActionStatus.approved, ActionStatus.edited}, ActionStatus.executed
        )

    def mark_failed(self, action_id: str, note: str = "") -> PendingAction:
        action = self._require_status(
            action_id, {ActionStatus.approved, ActionStatus.edited}
        )
        action.status = ActionStatus.failed
        if note:
            action.user_note = (action.user_note + " | " + note).strip(" |")
        self._save()
        return action

    # -- internals ----------------------------------------------------------------

    def _require_status(
        self, action_id: str, allowed: set[ActionStatus]
    ) -> PendingAction:
        action = self.get(action_id)
        if action.status not in allowed:
            raise ValueError(
                f"action {action_id} is {action.status.value}; "
                f"expected one of {sorted(s.value for s in allowed)}"
            )
        return action

    def _transition(
        self, action_id: str, allowed: set[ActionStatus], new_status: ActionStatus
    ) -> PendingAction:
        action = self._require_status(action_id, allowed)
        action.status = new_status
        action.decided_at = datetime.now(timezone.utc)
        self._save()
        self._log.info("action_" + new_status.value, action_id=action.id)
        return action

    def _decide(
        self, action: PendingAction, status: ActionStatus, note: str
    ) -> PendingAction:
        action.status = status
        action.user_note = note
        action.decided_at = datetime.now(timezone.utc)
        self._save()
        self._log.info("action_" + status.value, action_id=action.id, note=note)
        return action

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"actions": [a.model_dump(mode="json") for a in self._actions]}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def execute_action(
    action: PendingAction, mail: MailProvider, calendar: CalendarProvider
) -> str:
    """Execute an approved/edited action through the tools; returns a summary."""
    if action.type == ActionType.send_reply:
        assert isinstance(action.payload, ReplyDraft)
        message = mail.send_reply(action.payload)
        return f"reply sent to {', '.join(message.to)} (message {message.id})"
    assert isinstance(action.payload, FollowUpProposal)
    event = calendar.create_item(action.payload)
    return f"{event.kind.value} created for {event.due_at:%Y-%m-%d %H:%M} UTC ({event.id})"


def record_feedback(
    memory: MemoryStore, action: PendingAction, decision: str, note: str = ""
) -> MemoryRecord | None:
    """Persist an edit/reject decision as a feedback memory record.

    These records are injected into the DraftAgent prompt on later runs, so
    human corrections change future behavior.
    """
    if decision not in ("edited", "rejected"):
        return None
    thread_id = action.payload.thread_id
    value = f"{action.type.value} for thread {thread_id} was {decision} by the user."
    if note:
        value += f" User note: {note}"
    key = f"{action.type.value}:{thread_id}:{action.id[:8]}"
    return memory.remember(MemoryKind.feedback, key, value, source="human_review")
