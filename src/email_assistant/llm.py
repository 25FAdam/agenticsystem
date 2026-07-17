"""LLM layer: a thin, observable wrapper around the Anthropic SDK.

Design goals:
- every call returns a schema-validated Pydantic object (structured outputs
  via ``messages.parse``) — agents never touch raw JSON;
- one automatic retry on parse failure, logged and counted as an error metric;
- token usage and latency are recorded into the shared MetricsRegistry;
- the system prompt block carries a cache_control marker so repeated agent
  calls with the same prompt hit the prompt cache (short prompts below the
  model's cacheable minimum are silently uncached — harmless);
- agents depend on the ``LLM`` protocol, so tests inject a fake without
  touching the network.
"""

from __future__ import annotations

import time
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from .config import Settings, get_settings
from .observability import MetricsRegistry, get_logger

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_TOKENS = 16000


class LLMError(Exception):
    """Base error for LLM-layer failures (API errors, refusals)."""


class LLMParseError(LLMError):
    """The model did not produce a valid instance of the requested schema."""


class LLM(Protocol):
    """What agents see. ClaudeClient implements it; tests use a fake."""

    def structured(
        self,
        *,
        output_type: type[T],
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        name: str = "llm_call",
    ) -> T: ...


class ClaudeClient:
    def __init__(
        self,
        settings: Settings | None = None,
        metrics: MetricsRegistry | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if client is not None:
            self._client = client
        elif self._settings.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        else:
            # Fall back to the SDK's own credential resolution (env, profile).
            self._client = anthropic.Anthropic()
        self.metrics = metrics or MetricsRegistry()
        self._log = get_logger(component="llm")

    def structured(
        self,
        *,
        output_type: type[T],
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        name: str = "llm_call",
    ) -> T:
        """One schema-validated call; retries once on parse failure."""
        model = model or self._settings.model

        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                response = self._client.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": user}],
                    output_format=output_type,
                )
            except anthropic.APIError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                self.metrics.error(f"{name}_api_error")
                self._log.error(
                    "llm_api_error",
                    name=name,
                    model=model,
                    attempt=attempt,
                    latency_ms=round(latency_ms, 1),
                    error=type(exc).__name__,
                )
                raise LLMError(f"{name}: API call failed ({type(exc).__name__})") from exc

            latency_ms = (time.perf_counter() - started) * 1000
            self._record_call(name, model, response, latency_ms, attempt)

            if response.stop_reason == "refusal":
                self.metrics.error(f"{name}_refusal")
                raise LLMError(f"{name}: the model refused the request")

            parsed = getattr(response, "parsed_output", None)
            if parsed is not None:
                return parsed

            self.metrics.error(f"{name}_parse_failure")
            self._log.warning(
                "llm_parse_failure",
                name=name,
                model=model,
                attempt=attempt,
                stop_reason=response.stop_reason,
            )

        raise LLMParseError(
            f"{name}: no valid {output_type.__name__} after 2 attempts"
        )

    def _record_call(self, name, model, response, latency_ms, attempt) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = output_tokens = 0
        if usage is not None:
            # Total prompt size = uncached remainder + cache reads + cache writes.
            input_tokens = (
                (usage.input_tokens or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            )
            output_tokens = usage.output_tokens or 0

        self.metrics.incr(f"{name}_calls")
        self.metrics.observe(f"{name}_latency", latency_ms)
        self.metrics.add_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
        self._log.info(
            "llm_call",
            name=name,
            model=model,
            attempt=attempt,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=response.stop_reason,
        )
