"""CLI tests for the non-LLM commands (review / actions / inbox / memory)."""

import json
import shutil

import pytest
from typer.testing import CliRunner

from email_assistant.cli import app
from email_assistant.hitl import PendingActionQueue
from email_assistant.memory.long_term import MemoryStore
from email_assistant.schemas import ActionStatus, ActionType, MemoryKind, ReplyDraft
from tests.conftest import INBOX

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path):
    """A self-contained data dir the CLI is pointed at via ASSISTANT_DATA_DIR."""
    (tmp_path / "fixtures").mkdir()
    shutil.copy(INBOX, tmp_path / "fixtures" / "inbox.json")
    return tmp_path


def env(data_dir):
    return {"ASSISTANT_DATA_DIR": str(data_dir)}


def seed_queue(data_dir):
    queue = PendingActionQueue(data_dir / "actions.json")
    draft = ReplyDraft(
        thread_id="t-002",
        to=["dana.reyes@northwindlabs.com"],
        subject="Re: Question about API rate limits",
        body="Hi Dana, short canned reply.",
    )
    return queue.add(ActionType.send_reply, draft)


def test_actions_empty(data_dir):
    result = runner.invoke(app, ["actions"], env=env(data_dir))
    assert result.exit_code == 0
    assert "No actions recorded." in result.output


def test_inbox_lists_unread(data_dir):
    result = runner.invoke(app, ["inbox"], env=env(data_dir))
    assert result.exit_code == 0
    assert "t-001" in result.output
    assert "t-005" not in result.output  # read thread


def test_review_approve_executes_action(data_dir):
    action = seed_queue(data_dir)

    result = runner.invoke(app, ["review"], input="a\n", env=env(data_dir))

    assert result.exit_code == 0
    assert "Executed: reply sent" in result.output
    reloaded = PendingActionQueue(data_dir / "actions.json")
    assert reloaded.get(action.id).status == ActionStatus.executed


def test_review_reject_records_feedback(data_dir):
    action = seed_queue(data_dir)

    result = runner.invoke(app, ["review"], input="r\ntoo formal\n", env=env(data_dir))

    assert result.exit_code == 0
    assert "Rejected." in result.output
    reloaded = PendingActionQueue(data_dir / "actions.json")
    assert reloaded.get(action.id).status == ActionStatus.rejected
    assert reloaded.get(action.id).user_note == "too formal"

    store = MemoryStore(data_dir / "memory" / "memory.json")
    feedback = store.recall(kind=MemoryKind.feedback)
    assert len(feedback) == 1
    assert "too formal" in feedback[0].value


def test_review_with_empty_queue(data_dir):
    result = runner.invoke(app, ["review"], env=env(data_dir))
    assert result.exit_code == 0
    assert "No pending actions." in result.output


def test_run_without_api_key_exits(data_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Redirect env-file lookup away from the repo's .env
    result = runner.invoke(
        app,
        ["run"],
        env={**env(data_dir), "ANTHROPIC_API_KEY": ""},
    )
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
