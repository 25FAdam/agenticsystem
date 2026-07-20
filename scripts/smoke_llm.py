"""Live smoke test of the LLM layer — makes ONE real Claude API call.

Requires ANTHROPIC_API_KEY (in .env or the environment). Runs a mini-triage
on the urgent fixture thread (t-001) and prints the validated result plus the
metrics report, so you can see structured outputs, token accounting and
latency tracking working end to end.

Usage:  python scripts/smoke_llm.py
"""

import sys

from email_assistant.config import get_settings
from email_assistant.llm import ClaudeClient
from email_assistant.observability import bind_run, get_logger, new_id, setup_logging
from email_assistant.schemas import TriageResult
from email_assistant.tools.mail import MockMailProvider


def main() -> int:
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set (put it in .env) — skipping live call.")
        return 1

    setup_logging(level="INFO")
    bind_run(new_id("run"))
    log = get_logger(component="smoke_llm")

    mail = MockMailProvider(settings.fixtures_path, user_email=settings.user_email)
    thread = mail.get_thread("t-001")
    message = thread.last_message

    client = ClaudeClient(settings=settings)
    log.info("smoke_started", model=settings.model, thread_id=thread.id)

    result = client.structured(
        output_type=TriageResult,
        system=(
            "You are the triage component of an email assistant for "
            f"{settings.user_name} <{settings.user_email}>. Classify the email "
            "into intent, priority, labels and a route. thread_id must echo the "
            "given thread id. Confidence is your certainty in the route, 0-1."
        ),
        user=(
            f"thread_id: {thread.id}\n"
            f"from: {message.sender}\n"
            f"subject: {message.subject}\n\n"
            f"{message.body}"
        ),
        name="triage",
    )

    print("\n--- TriageResult (schema-validated) ---")
    print(result.model_dump_json(indent=2))
    print("\n" + client.metrics.render_report())

    ok = result.thread_id == thread.id and result.priority.value in ("urgent", "high")
    print(f"\nSanity check (urgent boss email → urgent/high priority): {'PASS' if ok else 'CHECK MANUALLY'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
