"""Draft agent: write contextual replies in the user's voice.

The prompt is grounded in long-term memory: tone preferences, facts about the
contact, related org facts, and past feedback records (which is how the
assistant's drafting behavior evolves over time).
"""

from __future__ import annotations

from ..memory.long_term import MemoryStore
from ..schemas import EmailThread, MemoryKind, ReplyDraft, ThreadSummary
from .base import Agent, render_thread

DRAFTER_SYSTEM_TEMPLATE = """\
You draft email replies on behalf of {user_name} <{user_email}> ("the user").

Rules:
- Write in the user's voice, honoring the preferences in MEMORY CONTEXT.
  Feedback records describe past human corrections — they override defaults.
- Ground every factual claim in the thread or MEMORY CONTEXT. If information
  is missing, the draft should say the user will follow up on that point —
  never invent facts, numbers, dates or commitments.
- Answer every question from the last message, briefly.
- to: the people who need the reply (usually the sender of the last message).
- subject: the thread subject, prefixed with "Re: " unless already present.
- tone: choose formal/neutral/casual to fit the relationship and preferences.
- rationale: one sentence on the choices you made, for the human reviewer."""


class DraftAgent(Agent):
    name = "drafter"

    def draft(
        self,
        thread: EmailThread,
        memory: MemoryStore,
        summary: ThreadSummary | None = None,
    ) -> ReplyDraft:
        system = DRAFTER_SYSTEM_TEMPLATE.format(
            user_name=self._settings.user_name,
            user_email=self._settings.user_email,
        )

        parts = [f"MEMORY CONTEXT:\n{self._memory_context(thread, memory)}"]
        if summary is not None:
            parts.append(
                "THREAD SUMMARY:\n"
                f"tl_dr: {summary.tl_dr}\n"
                f"open questions: {'; '.join(summary.open_questions) or '(none)'}\n"
                f"action items: {'; '.join(summary.action_items) or '(none)'}"
            )
        parts.append(f"THREAD:\n{render_thread(thread)}")
        parts.append("Write the reply now.")

        result = self._call(ReplyDraft, system, "\n\n".join(parts))
        result = self._fix_thread_id(result, thread)
        if not result.to:
            result.to = [self._reply_recipient(thread)]
        self._log.info(
            "draft_done",
            thread_id=thread.id,
            to=result.to,
            tone=result.tone.value,
            body_chars=len(result.body),
        )
        return result

    # -- helpers ---------------------------------------------------------------

    def _reply_recipient(self, thread: EmailThread) -> str:
        last = thread.last_message
        if last.sender.lower() != self._settings.user_email.lower():
            return last.sender
        others = [
            p for p in thread.participants
            if p.lower() != self._settings.user_email.lower()
        ]
        return others[0] if others else last.sender

    def _memory_context(self, thread: EmailThread, memory: MemoryStore) -> str:
        recipient = self._reply_recipient(thread)
        records: list = []
        records += memory.recall(kind=MemoryKind.preference)
        records += memory.recall(kind=MemoryKind.contact, key=recipient)
        records += memory.search(
            f"{thread.subject} {thread.last_message.body[:300]}", limit=4
        )
        feedback = sorted(
            memory.recall(kind=MemoryKind.feedback), key=lambda r: r.updated_at
        )
        records += feedback[-3:]

        seen: set[str] = set()
        lines: list[str] = []
        for record in records:
            if record.id in seen:
                continue
            seen.add(record.id)
            lines.append(f"[{record.kind.value}] {record.key}: {record.value}")
        return "\n".join(lines) if lines else "(no stored memory)"
