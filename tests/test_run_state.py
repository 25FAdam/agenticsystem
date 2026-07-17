import json

from email_assistant.memory.short_term import RunState
from email_assistant.schemas import Intent, Priority, Route, TriageResult


def test_context_for_creates_once():
    state = RunState()
    ctx1 = state.context_for("t-001")
    ctx2 = state.context_for("t-001")
    assert ctx1 is ctx2
    assert set(state.contexts) == {"t-001"}


def test_snapshot_is_json_serializable():
    state = RunState()
    state.context_for("t-001").triage = TriageResult(
        thread_id="t-001",
        intent=Intent.request,
        priority=Priority.urgent,
        route=Route.needs_reply,
        confidence=0.95,
    )
    state.add_note("processed t-001")

    snapshot = state.snapshot()
    assert json.dumps(snapshot)
    assert snapshot["contexts"]["t-001"]["triage"]["priority"] == "urgent"
