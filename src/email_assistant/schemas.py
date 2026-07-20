"""All typed contracts of the system.

Every agent/tool boundary passes one of these Pydantic models — no free-form
dicts anywhere. LLM calls are constrained to these shapes via structured
outputs / strict tool schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Mail domain
# ---------------------------------------------------------------------------


class EmailMessage(BaseModel):
    id: str
    thread_id: str
    sender: str
    to: list[str]
    cc: list[str] = Field(default_factory=list)
    date: datetime
    subject: str
    body: str


class EmailThread(BaseModel):
    id: str
    subject: str
    participants: list[str]
    messages: list[EmailMessage] = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    unread: bool = True
    archived: bool = False

    @property
    def last_message(self) -> EmailMessage:
        return max(self.messages, key=lambda m: m.date)


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    question = "question"
    request = "request"
    meeting = "meeting"
    fyi = "fyi"
    newsletter = "newsletter"
    billing = "billing"
    spam = "spam"
    other = "other"


class Priority(str, Enum):
    urgent = "urgent"
    high = "high"
    normal = "normal"
    low = "low"


class Route(str, Enum):
    needs_reply = "needs_reply"
    follow_up = "follow_up"
    archive = "archive"
    no_action = "no_action"


class TriageResult(BaseModel):
    thread_id: str
    intent: Intent
    priority: Priority
    labels: list[str] = Field(default_factory=list)
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Summarization / drafting / scheduling
# ---------------------------------------------------------------------------


class ThreadSummary(BaseModel):
    thread_id: str
    tl_dr: str
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class Tone(str, Enum):
    formal = "formal"
    neutral = "neutral"
    casual = "casual"


class ReplyDraft(BaseModel):
    thread_id: str
    to: list[str]
    subject: str
    body: str
    tone: Tone = Tone.neutral
    rationale: str = ""


class FollowUpKind(str, Enum):
    reminder = "reminder"
    event = "event"


class FollowUpProposal(BaseModel):
    thread_id: str
    kind: FollowUpKind
    title: str
    note: str = ""
    due_at: datetime


class CalendarEvent(BaseModel):
    id: str
    thread_id: str
    kind: FollowUpKind
    title: str
    note: str = ""
    due_at: datetime
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    send_reply = "send_reply"
    create_followup = "create_followup"


class ActionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    edited = "edited"
    rejected = "rejected"
    executed = "executed"
    failed = "failed"


class PendingAction(BaseModel):
    id: str = Field(default_factory=_new_uuid)
    type: ActionType
    payload: ReplyDraft | FollowUpProposal
    status: ActionStatus = ActionStatus.pending
    created_at: datetime = Field(default_factory=_utcnow)
    decided_at: datetime | None = None
    user_note: str = ""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryKind(str, Enum):
    preference = "preference"
    contact = "contact"
    org_fact = "org_fact"
    feedback = "feedback"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=_new_uuid)
    kind: MemoryKind
    key: str
    value: str
    source: str = "agent"
    updated_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class AgentStep(BaseModel):
    """One traced step of a run: an agent invocation or a tool call."""

    run_id: str
    step_id: str = Field(default_factory=_new_uuid)
    agent: str
    tool: str = ""
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)
