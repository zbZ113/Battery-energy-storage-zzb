from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_advanced_runner_exposes_plan_for_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_advanced_model_suite.py",
            "matr-three-batch",
            "smoke",
            "--plan-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "PLAN_READY"
    assert payload["mode"] == "smoke"
    assert payload["task_count"] == 4


def test_advanced_shell_pins_gpu_and_seed_subprocesses() -> None:
    path = ROOT / "scripts" / "a100" / "train_advanced_models.sh"
    script = path.read_text(encoding="utf-8")
    for marker in (
        "CUDA_VISIBLE_DEVICES=1",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "unset DISPLAY",
        "SEEDS=(38)",
        "SEEDS=(38 39 40)",
        "SEEDS=(38 39 40 41 42)",
    ):
        assert marker in script


def test_verify_script_is_executable_on_posix() -> None:
    if os.name == "nt":
        return
    mode = (ROOT / "scripts" / "a100" / "verify_advanced_run.sh").stat().st_mode
    assert mode & stat.S_IXUSR
