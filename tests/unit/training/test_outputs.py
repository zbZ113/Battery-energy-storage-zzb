from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.training.outputs import (
    build_training_output_index,
    verify_training_output_index,
)


def _write_valid_output(root: Path) -> None:
    started = datetime(2026, 7, 16, 1, 0, tzinfo=UTC)
    finished = datetime(2026, 7, 16, 1, 5, tzinfo=UTC)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "a100-training-run-v1",
                "run_id": str(uuid4()),
                "training_bundle_sha256": "a" * 64,
                "preflight_sha256": "b" * 64,
                "git_commit": "c" * 40,
                "git_dirty": False,
                "data_version": "safe-hust-v1",
                "split_version": "cell-split-v1",
                "feature_version": "early-v1",
                "cutoffs": [20, 50, 100, 150],
                "models": ["dummy", "variance", "xgboost", "cpmlp", "hybrid"],
                "seeds": [20260712],
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (root / "config_resolved.json").write_text("{}", encoding="utf-8")
    (root / "metrics.json").write_text('{"status": "not-yet-evaluated"}', encoding="utf-8")
    (root / "metrics.csv").write_text("model,status\ndummy,pending\n", encoding="utf-8")
    (root / "training_log.jsonl").write_text('{"event":"finished"}\n', encoding="utf-8")
    (root / "model_card.md").write_text("# Model card\n", encoding="utf-8")
    (root / "environment.json").write_text('{"python":"3.11.13"}', encoding="utf-8")
    (root / "plots").mkdir()
    (root / "plots" / "placeholder.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8"
    )
    (root / "artifacts").mkdir()
    (root / "artifacts" / "variance.json").write_text('{"coef": 1.0}', encoding="utf-8")


def test_training_output_index_binds_complete_safe_run(tmp_path: Path) -> None:
    _write_valid_output(tmp_path)

    index = build_training_output_index(
        tmp_path,
        created_at=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
    )

    assert index.run_manifest.run_id
    assert len(index.files) == 9
    assert len(index.output_sha256) == 64
    verify_training_output_index(tmp_path, index)


def test_training_output_index_rejects_missing_required_evidence(tmp_path: Path) -> None:
    _write_valid_output(tmp_path)
    (tmp_path / "metrics.csv").unlink()

    with pytest.raises(ValueError, match="required output"):
        build_training_output_index(
            tmp_path,
            created_at=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
        )


def test_training_output_index_rejects_executable_model_format(tmp_path: Path) -> None:
    _write_valid_output(tmp_path)
    (tmp_path / "artifacts" / "model.pt").write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="forbidden"):
        build_training_output_index(
            tmp_path,
            created_at=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
        )


def test_training_output_verification_detects_tampering(tmp_path: Path) -> None:
    _write_valid_output(tmp_path)
    index = build_training_output_index(
        tmp_path,
        created_at=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
    )
    (tmp_path / "metrics.json").write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"size|SHA-256"):
        verify_training_output_index(tmp_path, index)
