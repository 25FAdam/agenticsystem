"""Summarizer agent: condense a thread into a structured summary."""

from __future__ import annotations

from ..schemas import EmailThread, ThreadSummary
from .base import Agent, render_thread

SUMMARIZER_SYSTEM_TEMPLATE = """\
You summarize email threads for {user_name} <{user_email}> ("the user").

Produce:
- tl_dr: 1-2 sentences capturing what the thread is about and where it stands.
- key_points: the facts that matter, briefly.
- open_questions: unresolved questions, with who is being asked.
- action_items: concrete tasks with owner and deadline when known,
  e.g. "Adam: file the Snowflake access request by Friday".

Be faithful to the thread — do not invent details. thread_id: echo exactly."""


class SummarizerAgent(Agent):
    name = "summarizer"

    def summarize(self, thread: EmailThread) -> ThreadSummary:
        system = SUMMARIZER_SYSTEM_TEMPLATE.format(
            user_name=self._settings.user_name,
            user_email=self._settings.user_email,
        )
        result = self._call(ThreadSummary, system, render_thread(thread))
        result = self._fix_thread_id(result, thread)
        self._log.info(
            "summary_done",
            thread_id=thread.id,
            key_points=len(result.key_points),
            action_items=len(result.action_items),
        )
        return result
