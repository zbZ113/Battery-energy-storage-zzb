from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_blast_validation_cli_builds_requested_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "validation-bundle"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "data" / "prepare_blast_validation.py"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BUILT"
    assert payload["calendar_observation_count"] == 595
    assert payload["cycle_observation_count"] == 665
    assert (output_dir / "COMMITTED").is_file()
