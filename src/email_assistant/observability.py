"""Structured logging, trace IDs and in-process metrics.

Every agent step logs a JSON event bound to the current run_id; the
MetricsRegistry aggregates counters, latencies, token usage and errors into a
per-run report.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections import Counter, defaultdict
from typing import Any

import structlog


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # Resolve sys.stderr at logger-creation time (not configure time), so
        # stream redirection in CLI test runners cannot leave the global
        # config pointing at a closed stream.
        logger_factory=lambda *args: structlog.PrintLogger(sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(**bind: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger().bind(**bind)


def bind_run(run_id: str) -> None:
    """Attach run_id to all subsequent log events on this context."""
    structlog.contextvars.bind_contextvars(run_id=run_id)


class MetricsRegistry:
    """Per-run counters, latency observations, token usage and errors."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.latencies_ms: dict[str, list[float]] = defaultdict(list)
        self.input_tokens = 0
        self.output_tokens = 0
        self.errors: Counter[str] = Counter()

    def incr(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def observe(self, name: str, ms: float) -> None:
        self.latencies_ms[name].append(float(ms))

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def error(self, name: str) -> None:
        self.errors[name] += 1

    def as_dict(self) -> dict[str, Any]:
        latencies = {
            name: {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 1),
                "max_ms": round(max(values), 1),
            }
            for name, values in self.latencies_ms.items()
            if values
        }
        return {
            "counters": dict(self.counters),
            "latencies": latencies,
            "tokens": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "errors": dict(self.errors),
        }

    def render_report(self) -> str:
        data = self.as_dict()
        lines = ["=== Run metrics ===", "-- counters --"]
        for name, value in sorted(data["counters"].items()):
            lines.append(f"  {name}: {value}")
        lines.append("-- latencies --")
        for name, stats in sorted(data["latencies"].items()):
            lines.append(
                f"  {name}: n={stats['count']} avg={stats['avg_ms']}ms max={stats['max_ms']}ms"
            )
        lines.append("-- tokens --")
        lines.append(f"  input_tokens: {data['tokens']['input_tokens']}")
        lines.append(f"  output_tokens: {data['tokens']['output_tokens']}")
        lines.append("-- errors --")
        if data["errors"]:
            for name, value in sorted(data["errors"].items()):
                lines.append(f"  {name}: {value}")
        else:
            lines.append("  none")
        return "\n".join(lines)
