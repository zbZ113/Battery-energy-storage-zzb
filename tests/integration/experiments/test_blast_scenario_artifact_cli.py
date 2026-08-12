from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_blast_scenario_artifact_cli_builds_requested_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "scenario-artifacts"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_blast_scenario_artifacts.py"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BUILT"
    assert payload["scenario_count"] == 10
    assert payload["figure_count"] == 10
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_validation_sha = (
        REPOSITORY_ROOT
        / "reports"
        / "experiments"
        / "blast_naumann_v1"
        / "validation-v2"
        / "COMMITTED"
    ).read_text(encoding="ascii").strip()
    assert manifest["validation_result_sha256"] == expected_validation_sha
