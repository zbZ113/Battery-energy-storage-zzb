from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_blast_validation_cli_runs_real_naumann_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "blast-naumann-result"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_blast_validation.py"),
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
    assert payload["prediction_count"] == 1260
    assert payload["metric_count"] > 30
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "COMMITTED").is_file()
