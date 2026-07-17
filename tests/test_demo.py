"""Smoke test: the offline demo must run end to end without an API key."""

import os
import subprocess
import sys

from tests.conftest import ROOT


def test_demo_offline_runs_clean():
    result = subprocess.run(
        [sys.executable, "demo.py", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "Act 1" in output and "Act 5" in output
    assert "Rejected with note" in output
    assert "New draft for t-001 (after feedback)" in output
    assert "What this demo proved" in output
