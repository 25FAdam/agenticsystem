"""Deterministic rule pre-filter for triage.

Obvious bulk mail is classified without an LLM call: cheaper, faster, and
fully predictable. Anything the rules don't catch falls through to the LLM
classifier in TriageAgent. Rules are deliberately demo-grade — a production
system would use sender reputation / list-id headers.
"""

from __future__ import annotations

from ..schemas import EmailThread, Intent, Priority, Route, TriageResult

NEWSLETTER_SENDER_PREFIXES = ("newsletter@", "digest@", "news@", "noreply@", "no-reply@")
NEWSLETTER_MARKERS = ("unsubscribe", "read online", "view in browser")

SPAM_MARKERS = (
    "you have been selected",
    "claim within",
    "exclusive rewards",
    "your wallet",
    "congratulations",
    "act now",
    "winner",
)


def rule_prefilter(thread: EmailThread) -> TriageResult | None:
    """Return a TriageResult for obvious newsletter/spam threads, else None."""
    last = thread.last_message
    sender = last.sender.lower()
    text = f"{last.subject}\n{last.body}".lower()

    if any(sender.startswith(p) for p in NEWSLETTER_SENDER_PREFIXES) and any(
        marker in text for marker in NEWSLETTER_MARKERS
    ):
        return TriageResult(
            thread_id=thread.id,
            intent=Intent.newsletter,
            priority=Priority.low,
            labels=["newsletter"],
            route=Route.archive,
            confidence=1.0,
            rationale="rule: bulk sender with newsletter markers",
        )

    spam_hits = sum(1 for marker in SPAM_MARKERS if marker in text)
    if spam_hits >= 2:
        return TriageResult(
            thread_id=thread.id,
            intent=Intent.spam,
            priority=Priority.low,
            labels=["spam"],
            route=Route.archive,
            confidence=1.0,
            rationale=f"rule: {spam_hits} spam markers matched",
        )

    return None
