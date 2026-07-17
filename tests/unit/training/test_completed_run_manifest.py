from __future__ import annotations

import json
from pathlib import Path

import pytest

from quanxin_life.core import PredictionTarget
from quanxin_life.training.checkpoint import CheckpointContext
from quanxin_life.training.orchestrator import (
    _load_completed_run,
    _write_completed_run_manifest,
)


def _context() -> CheckpointContext:
    return CheckpointContext(
        run_id="matr-xgboost-c20-s20260712",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="xgboost",
        cutoff_cycle=20,
        seed=20260712,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="matr-2018-04-12-v1",
        split_version="matr-cell-split-v1",
        feature_version="matr-early-cycle-v1",
        source_commit="c" * 40,
    )


def _completed_run(root: Path) -> None:
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "model.ubj").write_bytes(b"safe-ubj")
    (root / "config_resolved.json").write_text("{}\n", encoding="utf-8")
    (root / "metrics_test.json").write_text(
        '{"mae": 12.5, "target": "matr_official_cycle_life"}\n',
        encoding="utf-8",
    )


def test_completed_manifest_exposes_full_versioned_run_context(tmp_path: Path) -> None:
    _completed_run(tmp_path)
    context = _context()

    _write_completed_run_manifest(tmp_path, context)

    payload = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "completed-run-v2"
    assert payload["run_id"] == context.run_id
    assert payload["dataset_id"] == "MATR"
    assert payload["target"] == "matr_official_cycle_life"
    assert payload["model_name"] == "xgboost"
    assert payload["cutoff_cycle"] == 20
    assert payload["seed"] == 20260712
    assert payload["config_sha256"] == "a" * 64
    assert payload["input_bundle_sha256"] == "b" * 64
    assert payload["source_commit"] == "c" * 40
    assert payload["data_version"] == "matr-2018-04-12-v1"
    assert payload["split_version"] == "matr-cell-split-v1"
    assert payload["feature_version"] == "matr-early-cycle-v1"
    assert payload["completed_at"].endswith("Z")
    assert _load_completed_run(tmp_path, context)["mae"] == 12.5


def test_completed_manifest_rejects_explicit_context_tampering(tmp_path: Path) -> None:
    _completed_run(tmp_path)
    context = _context()
    _write_completed_run_manifest(tmp_path, context)
    manifest_path = tmp_path / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["target"] = "unified_eol80_cycle"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="context"):
        _load_completed_run(tmp_path, context)
