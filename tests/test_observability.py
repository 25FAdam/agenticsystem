from email_assistant.observability import MetricsRegistry, get_logger, new_id, setup_logging


def test_new_id_prefix_and_uniqueness():
    a, b = new_id("run"), new_id("run")
    assert a.startswith("run_") and b.startswith("run_")
    assert a != b


def test_metrics_registry_aggregates():
    metrics = MetricsRegistry()
    metrics.incr("emails_processed", 3)
    metrics.observe("triage_latency", 120.0)
    metrics.observe("triage_latency", 80.0)
    metrics.add_tokens(input_tokens=1000, output_tokens=250)
    metrics.error("parse_failure")

    data = metrics.as_dict()
    assert data["counters"]["emails_processed"] == 3
    assert data["latencies"]["triage_latency"]["count"] == 2
    assert data["latencies"]["triage_latency"]["avg_ms"] == 100.0
    assert data["tokens"] == {"input_tokens": 1000, "output_tokens": 250}
    assert data["errors"]["parse_failure"] == 1


def test_render_report_contains_key_sections():
    metrics = MetricsRegistry()
    metrics.incr("emails_processed")
    report = metrics.render_report()
    assert "emails_processed: 1" in report
    assert "input_tokens: 0" in report
    assert "none" in report  # no errors


def test_logging_smoke():
    setup_logging(level="INFO")
    logger = get_logger(component="test")
    logger.info("hello", answer=42)  # must not raise
