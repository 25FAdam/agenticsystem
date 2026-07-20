"""Orchestrator: the run loop that coordinates the worker agents.

One run = fetch unread threads -> triage each -> dispatch by route
(needs_reply -> summarize + draft; follow_up -> scheduler proposal;
archive / no_action -> mail actions) -> sweep for threads awaiting a reply.

All outgoing actions go into the PendingActionQueue — nothing is sent or
scheduled without human review. Per-thread failures are logged and counted
but never abort the run.

Observability: every agent invocation is recorded as an AgentStep (span)
carrying latency, token deltas and any error; nested log events (llm_call)
share the step's span_id, and everything in a run shares one run_id. The
full run report (counters, metrics, steps) can be written to a JSON artifact
via ``write_report``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from .agents import (
    DraftAgent,
    SchedulerAgent,
    SummarizerAgent,
    TriageAgent,
    find_awaiting_response,
)
from .config import Settings, get_settings
from .hitl import PendingActionQueue
from .llm import LLM, LLMError
from .memory.long_term import MemoryStore
from .memory.short_term import RunState
from .observability import MetricsRegistry, bind_run, bound_span, get_logger
from .schemas import ActionType, AgentStep, EmailThread, Route

T = TypeVar("T")


def write_report(report: dict, runs_dir: Path) -> Path:
    """Persist a run report as a JSON artifact; returns the file path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        mail,
        memory: MemoryStore,
        queue: PendingActionQueue,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.mail = mail
        self.memory = memory
        self.queue = queue
        self.metrics: MetricsRegistry = getattr(llm, "metrics", None) or MetricsRegistry()

        self.triage = TriageAgent(llm, self._settings)
        self.summarizer = SummarizerAgent(llm, self._settings)
        self.drafter = DraftAgent(llm, self._settings)
        self.scheduler = SchedulerAgent(llm, self._settings)

        self.state = RunState()
        self.steps: list[AgentStep] = []
        self._log = get_logger(component="orchestrator")

    # -- run loop -----------------------------------------------------------------

    def run(self, max_threads: int | None = None, now: datetime | None = None) -> dict:
        bind_run(self.state.run_id)
        threads = self.mail.fetch_unread()
        if max_threads is not None:
            threads = threads[:max_threads]
        self._log.info("run_started", unread=len(threads))

        for thread in threads:
            try:
                self._process_thread(thread)
            except LLMError as exc:
                self.metrics.error("thread_failed")
                self._log.error("thread_failed", thread_id=thread.id, error=str(exc))

        self._followup_sweep(now)

        report = {
            "run_id": self.state.run_id,
            "unread_processed": len(threads),
            "pending_actions": len(self.queue.pending()),
            "counters": dict(self.metrics.counters),
            "metrics": self.metrics.as_dict(),
            "steps": [step.model_dump(mode="json") for step in self.steps],
        }
        self._log.info(
            "run_finished",
            unread_processed=report["unread_processed"],
            pending_actions=report["pending_actions"],
            steps=len(self.steps),
        )
        return report

    # -- step tracing --------------------------------------------------------------

    def _step(
        self,
        agent: str,
        thread_id: str,
        tool: str,
        fn: Callable[[], T],
        describe: Callable[[T], str] | None = None,
    ) -> T:
        """Run one agent invocation as a traced span (AgentStep)."""
        step = AgentStep(
            run_id=self.state.run_id,
            agent=agent,
            tool=tool,
            input_summary=f"thread {thread_id}",
        )
        tokens_before = (self.metrics.input_tokens, self.metrics.output_tokens)
        started = time.perf_counter()
        try:
            with bound_span(step.step_id):
                result = fn()
            step.output_summary = describe(result) if describe else type(result).__name__
            return result
        except Exception as exc:
            step.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            step.input_tokens = self.metrics.input_tokens - tokens_before[0]
            step.output_tokens = self.metrics.output_tokens - tokens_before[1]
            self.steps.append(step)
            event: dict = {
                "step_id": step.step_id,
                "agent": agent,
                "tool": tool,
                "thread_id": thread_id,
                "latency_ms": step.latency_ms,
                "input_tokens": step.input_tokens,
                "output_tokens": step.output_tokens,
            }
            if step.error:
                event["error"] = step.error
            self._log.info("agent_step", **event)

    # -- per-thread processing -------------------------------------------------------

    def _process_thread(self, thread: EmailThread) -> None:
        ctx = self.state.context_for(thread.id)
        triage = self._step(
            "triage",
            thread.id,
            "classifier",
            lambda: self.triage.triage(thread),
            lambda r: f"{r.intent.value}/{r.priority.value} -> {r.route.value}",
        )
        ctx.triage = triage
        self.metrics.incr("threads_processed")

        if triage.labels:
            self.mail.apply_labels(thread.id, triage.labels)

        if triage.route == Route.archive:
            self.mail.archive(thread.id)
            self.metrics.incr("threads_archived")
            return

        if triage.route == Route.no_action:
            self.mail.mark_read(thread.id)
            self.metrics.incr("threads_no_action")
            return

        if triage.route == Route.needs_reply:
            if not self.queue.has_action_for(thread.id, ActionType.send_reply):
                summary = None
                if len(thread.messages) > 1:
                    summary = self._step(
                        "summarizer",
                        thread.id,
                        "llm",
                        lambda: self.summarizer.summarize(thread),
                        lambda r: (
                            f"{len(r.key_points)} key points, "
                            f"{len(r.action_items)} action items"
                        ),
                    )
                    ctx.summary = summary
                draft = self._step(
                    "drafter",
                    thread.id,
                    "llm+memory",
                    lambda: self.drafter.draft(thread, self.memory, summary),
                    lambda r: f"draft to {', '.join(r.to)} ({len(r.body)} chars)",
                )
                ctx.draft = draft
                self.queue.add(ActionType.send_reply, draft)
                self.metrics.incr("replies_drafted")
            else:
                self._log.info("draft_skipped_duplicate", thread_id=thread.id)
            self.mail.mark_read(thread.id)
            return

        # Route.follow_up
        if not self.queue.has_action_for(thread.id, ActionType.create_followup):
            proposal = self._step(
                "scheduler",
                thread.id,
                "llm",
                lambda: self.scheduler.propose_followup(thread, reason=triage.rationale),
                lambda r: f"{r.kind.value} due {r.due_at:%Y-%m-%d %H:%M}",
            )
            ctx.followup = proposal
            self.queue.add(ActionType.create_followup, proposal)
            self.metrics.incr("followups_proposed")
        else:
            self._log.info("followup_skipped_duplicate", thread_id=thread.id)
        self.mail.mark_read(thread.id)

    # -- awaiting-reply sweep -----------------------------------------------------------

    def _followup_sweep(self, now: datetime | None = None) -> None:
        waiting = find_awaiting_response(
            self.mail.list_threads(), self._settings.user_email, now=now
        )
        for thread in waiting:
            if self.queue.has_action_for(thread.id, ActionType.create_followup):
                continue
            try:
                proposal = self._step(
                    "scheduler",
                    thread.id,
                    "llm",
                    lambda: self.scheduler.propose_followup(
                        thread,
                        reason=f"awaiting a reply since {thread.last_message.date:%Y-%m-%d}",
                        now=now,
                    ),
                    lambda r: f"{r.kind.value} due {r.due_at:%Y-%m-%d %H:%M}",
                )
            except LLMError as exc:
                self.metrics.error("followup_failed")
                self._log.error("followup_failed", thread_id=thread.id, error=str(exc))
                continue
            self.state.context_for(thread.id).followup = proposal
            self.queue.add(ActionType.create_followup, proposal)
            self.metrics.incr("followups_proposed")
