"""Streamlit UI for the agentic email assistant.

A thin visual layer over the exact same components the CLI uses: the
orchestrator, the pending-action queue, the memory store and the mock
providers. Queue, memory and calendar persist to data/; the mock inbox state
lives in the browser session.

Run:
    streamlit run app.py

Works without an API key in Offline (scripted) mode — same canned outputs as
`demo.py --offline`, flowing through the real orchestrator and review code.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from demo import build_scripted_llm
from email_assistant.config import Settings
from email_assistant.hitl import PendingActionQueue, execute_action, record_feedback
from email_assistant.memory.long_term import MemoryStore
from email_assistant.observability import setup_logging
from email_assistant.orchestrator import Orchestrator, write_report
from email_assistant.schemas import ActionStatus, MemoryKind, ReplyDraft
from email_assistant.tools.calendar import MockCalendarProvider
from email_assistant.tools.mail import MockMailProvider

st.set_page_config(page_title="Agentic Email Assistant", page_icon="📬", layout="wide")

SETTINGS = Settings()


# ---------------------------------------------------------------------------
# Session wiring
# ---------------------------------------------------------------------------


def init_session() -> None:
    if "mail" in st.session_state:
        return
    setup_logging(level="WARNING")
    st.session_state.mail = MockMailProvider(
        SETTINGS.fixtures_path,
        user_email=SETTINGS.user_email,
        user_name=SETTINGS.user_name,
    )
    st.session_state.calendar = MockCalendarProvider(SETTINGS.calendar_path)
    st.session_state.memory = MemoryStore(
        SETTINGS.memory_path, seed_path=SETTINGS.memory_seed_path
    )
    st.session_state.queue = PendingActionQueue(SETTINGS.actions_path)
    st.session_state.last_report = None
    st.session_state.last_contexts = {}
    st.session_state.last_metrics = ""
    st.session_state.flash = ""


def reset_workspace() -> None:
    for path in (SETTINGS.actions_path, SETTINGS.calendar_path, SETTINGS.memory_path):
        Path(path).unlink(missing_ok=True)
    st.session_state.clear()


def run_inbox(offline: bool) -> None:
    if offline:
        llm = build_scripted_llm()
    else:
        from email_assistant.llm import ClaudeClient

        llm = ClaudeClient(settings=SETTINGS)
    orchestrator = Orchestrator(
        llm=llm,
        mail=st.session_state.mail,
        memory=st.session_state.memory,
        queue=st.session_state.queue,
        settings=SETTINGS,
    )
    report = orchestrator.run()
    st.session_state.last_report = report
    st.session_state.last_contexts = dict(orchestrator.state.contexts)
    st.session_state.last_metrics = orchestrator.metrics.render_report()
    write_report(report, SETTINGS.runs_dir)


init_session()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📬 Email Assistant")
    has_key = bool(SETTINGS.anthropic_api_key)
    mode = st.radio(
        "Mode",
        ["Live (Claude API)", "Offline (scripted)"],
        index=0 if has_key else 1,
        help="Offline replays realistic canned agent outputs through the real "
        "orchestrator — no API key needed.",
    )
    offline = mode.startswith("Offline")
    if not offline and not has_key:
        st.error("ANTHROPIC_API_KEY is not set — add it to .env or switch to Offline.")
    st.caption(f"Model: `{SETTINGS.model}`")
    st.caption(f"User: {SETTINGS.user_name} <{SETTINGS.user_email}>")
    st.divider()
    st.caption(
        "Queue, memory and calendar persist to `data/`. The mock inbox state "
        "lives in this browser session."
    )
    if st.button(
        "🔄 Reset workspace",
        help="Clears the action queue and calendar, re-seeds memory, restores "
        "the fixture inbox.",
    ):
        reset_workspace()
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_inbox, tab_run, tab_review, tab_memory, tab_telemetry = st.tabs(
    ["📥 Inbox", "▶️ Run", "✅ Review", "🧠 Memory", "📈 Telemetry"]
)

mail = st.session_state.mail
queue = st.session_state.queue
memory = st.session_state.memory
calendar = st.session_state.calendar

# ---- Inbox -----------------------------------------------------------------

with tab_inbox:
    threads = mail.list_threads()
    st.subheader(f"Mailbox — {sum(t.unread for t in threads)} unread of {len(threads)}")
    st.dataframe(
        [
            {
                "thread": t.id,
                "unread": "🔵" if t.unread else "",
                "archived": "🗄️" if t.archived else "",
                "from": t.last_message.sender,
                "subject": t.subject,
                "labels": ", ".join(t.labels),
                "last message": f"{t.last_message.date:%m-%d %H:%M}",
            }
            for t in sorted(threads, key=lambda t: t.last_message.date, reverse=True)
        ],
        width="stretch",
        hide_index=True,
    )

    selected = st.selectbox("Open thread", [t.id for t in threads])
    thread = mail.get_thread(selected)
    st.markdown(f"#### {thread.subject}")
    for message in sorted(thread.messages, key=lambda m: m.date):
        who = "🟢 me" if message.sender == SETTINGS.user_email else f"👤 {message.sender}"
        with st.expander(f"{who} — {message.date:%Y-%m-%d %H:%M} UTC", expanded=True):
            st.text(message.body)

# ---- Run --------------------------------------------------------------------

with tab_run:
    st.subheader("Process the inbox")
    st.write(
        "Triage every unread thread, summarize and draft where a reply is "
        "needed, propose follow-ups — **all outgoing actions are queued for "
        "review, nothing is sent.**"
    )
    disabled = (not offline) and (not has_key)
    if st.button("▶️ Process inbox", type="primary", disabled=disabled):
        with st.spinner("Agents working… (triage → summarize → draft → schedule)"):
            run_inbox(offline)
        st.rerun()

    if st.session_state.last_report:
        report = st.session_state.last_report
        counters = report["counters"]
        cols = st.columns(5)
        cols[0].metric("Threads processed", counters.get("threads_processed", 0))
        cols[1].metric("Replies drafted", counters.get("replies_drafted", 0))
        cols[2].metric("Follow-ups", counters.get("followups_proposed", 0))
        cols[3].metric("Archived", counters.get("threads_archived", 0))
        cols[4].metric("Awaiting review", report["pending_actions"])

        st.markdown(f"##### Triage decisions — run `{report['run_id']}`")
        st.dataframe(
            [
                {
                    "thread": ctx.thread_id,
                    "intent": ctx.triage.intent.value,
                    "priority": ctx.triage.priority.value,
                    "route": ctx.triage.route.value,
                    "labels": ", ".join(ctx.triage.labels),
                    "rationale": ctx.triage.rationale,
                }
                for ctx in st.session_state.last_contexts.values()
                if ctx.triage
            ],
            width="stretch",
            hide_index=True,
        )
        st.info("Head to the **Review** tab to approve, edit or reject the drafts.")
    else:
        st.caption("No run in this session yet.")

# ---- Review ------------------------------------------------------------------

with tab_review:
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = ""

    pending = queue.pending()
    st.subheader(f"Pending actions ({len(pending)})")
    if not pending:
        st.caption("Nothing to review — run the inbox first.")

    for action in pending:
        payload = action.payload
        with st.container(border=True):
            if isinstance(payload, ReplyDraft):
                st.markdown(
                    f"**✉️ Reply — thread `{payload.thread_id}`** → "
                    f"{', '.join(payload.to)}"
                )
                st.caption(f"{payload.subject} · tone: {payload.tone.value}")
                body = st.text_area(
                    "Draft body — edit before sending if needed",
                    payload.body,
                    key=f"body_{action.id}",
                    height=170,
                )
                edited = body.strip() != payload.body.strip()
                if payload.rationale:
                    st.caption(f"agent rationale: {payload.rationale}")
            else:
                st.markdown(
                    f"**⏰ Follow-up — thread `{payload.thread_id}`**: {payload.title}"
                )
                st.caption(
                    f"{payload.kind.value} · due {payload.due_at:%Y-%m-%d %H:%M} UTC"
                )
                if payload.note:
                    st.write(payload.note)
                edited = False

            note = st.text_input(
                "Note — stored as feedback memory on edit/reject",
                key=f"note_{action.id}",
                placeholder="e.g. Too formal — keep it to two short sentences",
            )
            approve_label = "✅ Send edited version" if edited else "✅ Approve & execute"
            col_ok, col_reject, _ = st.columns([1, 1, 2])
            if col_ok.button(approve_label, key=f"ok_{action.id}"):
                if edited:
                    new_payload = payload.model_copy(update={"body": body.strip()})
                    queue.edit(action.id, new_payload, note)
                    record_feedback(memory, queue.get(action.id), "edited", note)
                else:
                    queue.approve(action.id)
                try:
                    description = execute_action(queue.get(action.id), mail, calendar)
                    queue.mark_executed(action.id)
                    st.session_state.flash = f"Executed: {description}"
                except Exception as exc:  # noqa: BLE001
                    queue.mark_failed(action.id, str(exc))
                    st.session_state.flash = f"Execution failed: {exc}"
                st.rerun()
            if col_reject.button("🗑️ Reject", key=f"rej_{action.id}"):
                queue.reject(action.id, note)
                record_feedback(memory, queue.get(action.id), "rejected", note)
                st.session_state.flash = (
                    "Rejected — stored as feedback for future drafts."
                )
                st.rerun()

    decided = [a for a in queue.all() if a.status != ActionStatus.pending]
    if decided:
        with st.expander(f"History ({len(decided)})"):
            st.dataframe(
                [
                    {
                        "id": a.id[:8],
                        "type": a.type.value,
                        "thread": a.payload.thread_id,
                        "status": a.status.value,
                        "note": a.user_note,
                        "decided": f"{a.decided_at:%m-%d %H:%M}" if a.decided_at else "",
                    }
                    for a in decided
                ],
                width="stretch",
                hide_index=True,
            )

# ---- Memory ------------------------------------------------------------------

with tab_memory:
    st.subheader("Long-term memory")
    kind_filter = st.selectbox(
        "Kind", ["all"] + [k.value for k in MemoryKind], index=0
    )
    records = memory.recall(
        kind=None if kind_filter == "all" else MemoryKind(kind_filter)
    )
    st.dataframe(
        [
            {
                "kind": r.kind.value,
                "key": r.key,
                "value": r.value,
                "source": r.source,
                "updated": f"{r.updated_at:%Y-%m-%d %H:%M}",
            }
            for r in sorted(records, key=lambda r: r.updated_at, reverse=True)
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "`feedback` records come from your edit/reject decisions in Review — "
        "the DraftAgent injects the latest ones into its prompt, so behavior "
        "changes on the next run."
    )

# ---- Telemetry ----------------------------------------------------------------

with tab_telemetry:
    st.subheader("Run telemetry")
    if st.session_state.last_metrics:
        st.code(st.session_state.last_metrics, language="text")
        report = st.session_state.last_report
        if report and report.get("steps"):
            st.markdown("##### Agent steps (spans)")
            st.dataframe(
                [
                    {
                        "agent": s["agent"],
                        "tool": s["tool"],
                        "input": s["input_summary"],
                        "output": s["output_summary"],
                        "latency ms": s["latency_ms"],
                        "in tokens": s["input_tokens"],
                        "out tokens": s["output_tokens"],
                        "error": s["error"],
                    }
                    for s in report["steps"]
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        st.caption("No run in this session yet.")

    runs_dir = SETTINGS.runs_dir
    run_files = sorted(runs_dir.glob("*.json"), reverse=True) if runs_dir.exists() else []
    if run_files:
        st.markdown("##### Past run reports")
        chosen = st.selectbox("Report", [f.name for f in run_files])
        with st.expander("Report JSON"):
            st.json(json.loads((runs_dir / chosen).read_text(encoding="utf-8")))
