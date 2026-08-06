from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _catalog(root: Path) -> None:
    raw = root / "data" / "raw" / "MATR" / "v1" / "source.mat"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"reviewed raw bytes")
    import hashlib

    payload = [
        {
            "dataset_id": "MATR",
            "version": "test-v1",
            "source_uri": "https://example.invalid/matr",
            "paper_uri": "https://example.invalid/paper",
            "license_status": "reviewed-for-test",
            "ingestion_mode": "hdf5",
            "expected_suffixes": [".mat"],
            "downloaded_at": "2026-08-04T00:00:00Z",
            "artifact_paths": ["data/raw/MATR/v1/source.mat"],
            "artifact_sha256": [hashlib.sha256(raw.read_bytes()).hexdigest()],
        }
    ]
    config = root / "configs" / "data_sources.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(payload), encoding="utf-8")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/data/prepare_all_datasets.py",
            *args,
            "--project-root",
            str(root),
        ],
        cwd=Path(__file__).parents[3],
        check=False,
        capture_output=True,
        text=True,
    )


def test_plan_is_read_only_and_build_is_idempotent(tmp_path: Path) -> None:
    _catalog(tmp_path)

    planned = _run(tmp_path, "plan")
    assert planned.returncode == 0
    assert '"status": "READY"' in planned.stdout
    assert not (tmp_path / "data" / "processed").exists()

    built = _run(tmp_path, "build", "--dataset", "MATR")
    assert built.returncode == 0
    repeated = _run(tmp_path, "build", "--dataset", "MATR")
    assert '"status": "SKIPPED_VALID"' in repeated.stdout
