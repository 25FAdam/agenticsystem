"""Triage agent: classify intent + priority, label, and route each thread."""

from __future__ import annotations

from ..schemas import EmailThread, TriageResult
from ..tools.classifier import rule_prefilter
from .base import Agent, render_thread

TRIAGE_SYSTEM_TEMPLATE = """\
You are the triage component of an email assistant for {user_name} <{user_email}> ("the user").

Classify the thread based on its LAST message; earlier messages are context.

intent — what the last message is:
- question: someone asks the user for information
- request: someone asks the user to do something
- meeting: scheduling or rescheduling a meeting
- fyi: informational, no response expected
- newsletter: bulk mailing / digest
- billing: invoices, payments, quotes
- spam: unsolicited junk
- other: anything else

priority:
- urgent: explicit same-day deadline, active incident, or direct manager escalation
- high: client-facing or time-sensitive within days
- normal: routine work items
- low: no time pressure

route — what should happen next:
- needs_reply: the user is expected to respond; a reply draft will be prepared
- follow_up: no reply needed now, but a reminder or calendar entry should be scheduled
- archive: nothing worth keeping in the inbox (newsletters, spam)
- no_action: keep and label only (e.g. FYI notices)

labels: 1-4 short lowercase tags describing the thread.
thread_id: echo the given thread_id exactly.
confidence: 0-1, your certainty in the chosen route.
rationale: one short sentence explaining the routing decision."""


class TriageAgent(Agent):
    name = "triage"

    def triage(self, thread: EmailThread) -> TriageResult:
        rule_result = rule_prefilter(thread)
        if rule_result is not None:
            self._log.info(
                "triage_rule_hit",
                thread_id=thread.id,
                intent=rule_result.intent.value,
                route=rule_result.route.value,
            )
            return rule_result

        system = TRIAGE_SYSTEM_TEMPLATE.format(
            user_name=self._settings.user_name,
            user_email=self._settings.user_email,
        )
        result = self._call(
            TriageResult,
            system,
            render_thread(thread),
            model=self._settings.effective_triage_model,
        )
        result = self._fix_thread_id(result, thread)
        self._log.info(
            "triage_done",
            thread_id=thread.id,
            intent=result.intent.value,
            priority=result.priority.value,
            route=result.route.value,
            confidence=result.confidence,
        )
        return result
