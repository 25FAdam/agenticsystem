from pathlib import Path

import pytest

from email_assistant.memory.long_term import MemoryStore
from email_assistant.tools.mail import MockMailProvider

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "fixtures" / "inbox.json"
SEED = ROOT / "data" / "memory" / "memory.seed.json"

USER_EMAIL = "adam.farkas@acme.io"


@pytest.fixture
def mail() -> MockMailProvider:
    return MockMailProvider(INBOX, user_email=USER_EMAIL, user_name="Adam Farkas")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json")


@pytest.fixture
def seeded_store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json", seed_path=SEED)
