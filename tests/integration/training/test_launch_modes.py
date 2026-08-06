from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_plan_mode_reports_ready_and_blocked_without_training() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_training_matrix.py", "plan"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "plan"
    assert payload["status"] == "PLAN_READY"
    assert payload["task_count"] == 12
    assert payload["blocked_count"] == 12
    assert payload["completed_count"] == 0
