"""Short-term memory: per-run working state held by the orchestrator.

Lives for a single processing run; snapshotted into logs for debuggability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..observability import new_id
from ..schemas import FollowUpProposal, ReplyDraft, ThreadSummary, TriageResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ThreadContext(BaseModel):
    """Everything the run has learned about one thread so far."""

    thread_id: str
    triage: TriageResult | None = None
    summary: ThreadSummary | None = None
    draft: ReplyDraft | None = None
    followup: FollowUpProposal | None = None


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    started_at: datetime = Field(default_factory=_utcnow)
    contexts: dict[str, ThreadContext] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def context_for(self, thread_id: str) -> ThreadContext:
        if thread_id not in self.contexts:
            self.contexts[thread_id] = ThreadContext(thread_id=thread_id)
        return self.contexts[thread_id]

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot for logging."""
        return self.model_dump(mode="json")
