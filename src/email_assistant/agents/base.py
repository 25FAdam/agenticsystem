"""Shared plumbing for worker agents: prompt assembly + structured LLM calls."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings
from ..llm import LLM
from ..observability import get_logger
from ..schemas import EmailThread

T = TypeVar("T", bound=BaseModel)


def render_thread(thread: EmailThread, max_body_chars: int = 4000) -> str:
    """Render a thread into a plain-text block for prompts (oldest first)."""
    lines = [
        f"thread_id: {thread.id}",
        f"subject: {thread.subject}",
        f"participants: {', '.join(thread.participants)}",
        f"labels: {', '.join(thread.labels) or '(none)'}",
        "",
    ]
    for message in sorted(thread.messages, key=lambda m: m.date):
        body = message.body.strip()
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + " [...truncated]"
        lines += [
            f"--- message | {message.date:%Y-%m-%d %H:%M} UTC ---",
            f"from: {message.sender}",
            f"to: {', '.join(message.to)}",
            body,
            "",
        ]
    return "\n".join(lines)


class Agent:
    """Base class: holds the LLM handle, settings and a component logger."""

    name = "agent"

    def __init__(self, llm: LLM, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings or get_settings()
        self._log = get_logger(component=self.name)

    def _call(
        self,
        output_type: type[T],
        system: str,
        user: str,
        model: str | None = None,
    ) -> T:
        return self._llm.structured(
            output_type=output_type,
            system=system,
            user=user,
            model=model or self._settings.model,
            name=self.name,
        )

    def _fix_thread_id(self, result: T, thread: EmailThread) -> T:
        """Defensive: force the echoed thread_id to match the actual thread."""
        if getattr(result, "thread_id", thread.id) != thread.id:
            self._log.warning(
                "thread_id_mismatch_corrected",
                expected=thread.id,
                got=result.thread_id,  # type: ignore[attr-defined]
            )
            result.thread_id = thread.id  # type: ignore[attr-defined]
        return result
