from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from quanxin_life.training.advanced_outputs import (
    build_advanced_training_output_index,
    verify_advanced_training_output_index,
)


def _write_run(
    root: Path,
    name: str,
    *,
    with_test_metrics: bool,
    status: str = "COMPLETED",
) -> None:
    run = root / name
    checkpoint = run / "checkpoints" / "epoch-000001"
    checkpoint.mkdir(parents=True)
    (run / "run_status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (run / "training_log.jsonl").write_text('{"epoch":1}\n', encoding="utf-8")
    (run / "metrics_validation.csv").write_text("mae\n1.0\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"safe-model")
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    if with_test_metrics:
        (run / "metrics_test.json").write_text(
            json.dumps({"mae": 1.0}), encoding="utf-8"
        )


def test_advanced_final_output_index_requires_and_verifies_all_runs(tmp_path: Path) -> None:
    for index in range(80):
        _write_run(tmp_path, f"run-{index:03d}", with_test_metrics=True)
    (tmp_path / "aggregate_metrics.json").write_text(
        json.dumps({"run_count": 80}), encoding="utf-8"
    )

    result = build_advanced_training_output_index(
        tmp_path,
        mode="final",
        source_commit="a" * 40,
        config_sha256="b" * 64,
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
    )

    assert result.operation_count == 80
    verify_advanced_training_output_index(tmp_path, result)


def test_advanced_select_accepts_stage_one_paused_candidates(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "stage1/eliminated",
        with_test_metrics=False,
        status="PAUSED_STAGE",
    )
    trace = {
        "stage1_evidence": [{} for _ in range(13)],
        "stage2_evidence": [{} for _ in range(7)],
        "recheck_evidence": [{} for _ in range(84)],
    }
    (tmp_path / "selection_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (tmp_path / "selection_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final_config_resolved.json").write_text("{}", encoding="utf-8")

    result = build_advanced_training_output_index(
        tmp_path,
        mode="select",
        source_commit="a" * 40,
        config_sha256="b" * 64,
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
    )

    assert result.operation_count == 104
