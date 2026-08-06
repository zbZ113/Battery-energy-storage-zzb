from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.unit.training.test_promotion_bundle import _write_source


def test_export_promoted_bundle_cli_reports_inactive_status(tmp_path: Path) -> None:
    source = tmp_path / "best"
    _write_source(source)
    destination = tmp_path / "bundle"
    script = Path(__file__).resolve().parents[3] / "scripts" / "export_promoted_model_bundle.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            str(source),
            "--output-root",
            str(destination),
            "--task",
            "RUL",
            "--family",
            "demo",
            "--version",
            "v1",
            "--source-commit",
            "a" * 40,
            "--data-version",
            "data-v1",
            "--split-version",
            "cell-split-v1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "VERIFIED_NOT_ACTIVATED"
    assert (destination / "artifact_manifest.json").is_file()
