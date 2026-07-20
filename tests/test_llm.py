import types

import anthropic
import pytest
from pydantic import BaseModel

from email_assistant.config import Settings
from email_assistant.llm import ClaudeClient, LLMError, LLMParseError
from email_assistant.observability import MetricsRegistry


class Verdict(BaseModel):
    label: str
    confidence: float


VERDICT = Verdict(label="ok", confidence=0.9)


def make_response(parsed, stop_reason="end_turn", input_tokens=100, output_tokens=20):
    return types.SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model="claude-opus-4-8",
        usage=types.SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class StubAnthropic:
    """Stands in for anthropic.Anthropic; replays canned parse() results."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []
        self.messages = types.SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class BoomAPIError(anthropic.APIError):
    def __init__(self):
        Exception.__init__(self, "boom")


def make_client(stub, metrics):
    return ClaudeClient(settings=Settings(), metrics=metrics, client=stub)


@pytest.fixture
def metrics():
    return MetricsRegistry()


def test_structured_returns_parsed_model_and_records_metrics(metrics):
    stub = StubAnthropic([make_response(VERDICT)])
    client = make_client(stub, metrics)

    result = client.structured(
        output_type=Verdict, system="sys", user="usr", name="triage"
    )

    assert result == VERDICT
    assert len(stub.calls) == 1
    data = metrics.as_dict()
    assert data["counters"]["triage_calls"] == 1
    assert data["tokens"] == {"input_tokens": 100, "output_tokens": 20}
    assert data["errors"] == {}


def test_request_shape(metrics):
    stub = StubAnthropic([make_response(VERDICT)])
    make_client(stub, metrics).structured(output_type=Verdict, system="sys", user="usr")

    kwargs = stub.calls[0]
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["output_format"] is Verdict
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["messages"] == [{"role": "user", "content": "usr"}]


def test_model_override(metrics):
    stub = StubAnthropic([make_response(VERDICT)])
    make_client(stub, metrics).structured(
        output_type=Verdict, system="s", user="u", model="claude-haiku-4-5"
    )
    assert stub.calls[0]["model"] == "claude-haiku-4-5"


def test_retries_once_on_parse_failure(metrics):
    stub = StubAnthropic([make_response(None), make_response(VERDICT)])
    client = make_client(stub, metrics)

    result = client.structured(
        output_type=Verdict, system="s", user="u", name="triage"
    )

    assert result == VERDICT
    assert len(stub.calls) == 2
    assert metrics.errors["triage_parse_failure"] == 1


def test_raises_after_two_parse_failures(metrics):
    stub = StubAnthropic([make_response(None), make_response(None)])
    client = make_client(stub, metrics)

    with pytest.raises(LLMParseError):
        client.structured(output_type=Verdict, system="s", user="u", name="triage")
    assert metrics.errors["triage_parse_failure"] == 2


def test_refusal_raises_without_retry(metrics):
    stub = StubAnthropic([make_response(None, stop_reason="refusal")])
    client = make_client(stub, metrics)

    with pytest.raises(LLMError, match="refused"):
        client.structured(output_type=Verdict, system="s", user="u", name="triage")
    assert len(stub.calls) == 1
    assert metrics.errors["triage_refusal"] == 1


def test_api_error_is_wrapped(metrics):
    stub = StubAnthropic([BoomAPIError()])
    client = make_client(stub, metrics)

    with pytest.raises(LLMError, match="API call failed"):
        client.structured(output_type=Verdict, system="s", user="u", name="triage")
    assert metrics.errors["triage_api_error"] == 1
