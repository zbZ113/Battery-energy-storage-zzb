from __future__ import annotations

import csv
from pathlib import Path

from quanxin_life.training.reports import write_matr_experiment_reports


def test_report_writer_emits_machine_and_human_readable_summaries(tmp_path: Path) -> None:
    aggregate = {
        "schema_version": "matr-aggregate-metrics-v1",
        "mode": "final",
        "formal_performance_claim": True,
        "expected_seed_count": 2,
        "run_count": 2,
        "failed_metric_rows": [],
        "summaries": [
            {
                "model": "xgboost",
                "cutoff_cycle": 20,
                "target": "matr_official_cycle_life",
                "run_count": 2,
                "seeds": [20260712, 20260713],
                "complete_seed_matrix": True,
                "metrics": {
                    "mae": {"count": 2, "mean": 12.0, "std": 2.0},
                    "picp": {"count": 2, "mean": 0.9, "std": 0.1},
                    "mpiw_cycle": {"count": 2, "mean": 45.0, "std": 5.0},
                },
                "mae_mean": 12.0,
                "mae_std": 2.0,
            }
        ],
    }

    write_matr_experiment_reports(tmp_path, aggregate)

    with (tmp_path / "aggregate_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target"] == "matr_official_cycle_life"
    assert rows[0]["mae_mean"] == "12.0"
    assert rows[0]["picp_mean"] == "0.9"
    report = (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert "MATR 官方 cycle-life" in report
    assert "不得解释为统一 EOL80" in report
    assert "xgboost" in report
    assert "12.0000 ± 2.0000" in report
    assert "正式五种子矩阵完整" in report


def test_report_writer_marks_incomplete_or_smoke_runs_as_non_formal(tmp_path: Path) -> None:
    aggregate = {
        "schema_version": "matr-aggregate-metrics-v1",
        "mode": "smoke",
        "formal_performance_claim": False,
        "expected_seed_count": 1,
        "run_count": 1,
        "failed_metric_rows": [{"model": "cpmlp", "reason": "INTERRUPTED"}],
        "summaries": [],
    }

    write_matr_experiment_reports(tmp_path, aggregate)

    report = (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert "不得作为正式性能结论" in report
    assert "失败或不完整记录: 1" in report
