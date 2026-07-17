from __future__ import annotations

import csv
import json
from pathlib import Path

from quanxin_life.core import PredictionTarget
from quanxin_life.training.checkpoint import CheckpointContext
from quanxin_life.training.orchestrator import _write_static_baseline_logs


def _context() -> CheckpointContext:
    return CheckpointContext(
        run_id="matr-dummy-c20-s20260712",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="dummy",
        cutoff_cycle=20,
        seed=20260712,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="data-v1",
        split_version="split-v1",
        feature_version="feature-v1",
        source_commit="c" * 40,
    )


def test_static_baseline_writes_validation_and_fit_logs(tmp_path: Path) -> None:
    _write_static_baseline_logs(
        tmp_path,
        context=_context(),
        training_time_seconds=0.125,
        validation_metrics={"mae": 10.0, "rmse": 12.0, "mape": 2.5, "r2": 0.8},
    )

    event = json.loads(
        (tmp_path / "training_log.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert event["event"] == "fit_and_validation_complete"
    assert event["target"] == "matr_official_cycle_life"
    assert event["validation_mae"] == 10.0
    with (tmp_path / "metrics_epoch.csv").open(encoding="utf-8", newline="") as handle:
        epoch_rows = list(csv.DictReader(handle))
    with (tmp_path / "metrics_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        validation_rows = list(csv.DictReader(handle))
    assert epoch_rows[0]["epoch"] == "0"
    assert validation_rows[0]["mae"] == "10.0"
