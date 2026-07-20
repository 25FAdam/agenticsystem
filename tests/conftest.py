from pathlib import Path

import pytest
from pydantic import BaseModel

from email_assistant.memory.long_term import MemoryStore
from email_assistant.tools.mail import MockMailProvider

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "fixtures" / "inbox.json"
SEED = ROOT / "data" / "memory" / "memory.seed.json"

USER_EMAIL = "adam.foldvari@testmail.com"


class FakeLLM:
    """Schema-faithful stand-in for ClaudeClient.

    Returns canned Pydantic objects matched by output type and records every
    call, so tests can assert on prompts without touching the network.
    """

    def __init__(self, canned=()) -> None:
        self._canned = list(canned)
        self.calls: list[dict] = []

    def structured(
        self,
        *,
        output_type: type[BaseModel],
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 16000,
        name: str = "llm_call",
    ):
        self.calls.append(
            {
                "output_type": output_type,
                "system": system,
                "user": user,
                "model": model,
                "name": name,
            }
        )
        for i, obj in enumerate(self._canned):
            if isinstance(obj, output_type):
                return self._canned.pop(i)
        raise AssertionError(f"FakeLLM has no canned {output_type.__name__}")


@pytest.fixture
def mail() -> MockMailProvider:
    return MockMailProvider(INBOX, user_email=USER_EMAIL, user_name="Adam Foldvari")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json")


@pytest.fixture
def seeded_store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json", seed_path=SEED)
