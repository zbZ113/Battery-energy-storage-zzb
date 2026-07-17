from quanxin_life.training.orchestrator import _aggregate


def test_aggregate_reports_each_comparable_metric_across_expected_seeds() -> None:
    rows = [
        {
            "model": "xgboost",
            "cutoff_cycle": 20,
            "target": "matr_official_cycle_life",
            "seed": 20260712,
            "mae": 10.0,
            "rmse": 12.0,
            "picp": 0.8,
            "mpiw_cycle": 40.0,
            "best_iteration": 100,
            "warnings": [],
        },
        {
            "model": "xgboost",
            "cutoff_cycle": 20,
            "target": "matr_official_cycle_life",
            "seed": 20260713,
            "mae": 14.0,
            "rmse": 16.0,
            "picp": 1.0,
            "mpiw_cycle": 50.0,
            "best_iteration": 120,
            "warnings": [],
        },
    ]

    aggregate = _aggregate(
        rows,
        mode="final",
        expected_seeds=(20260712, 20260713),
    )
    summary = aggregate["summaries"][0]

    assert summary["complete_seed_matrix"] is True
    assert summary["seeds"] == [20260712, 20260713]
    assert summary["metrics"]["mae"] == {"count": 2, "mean": 12.0, "std": 2**0.5 * 2}
    assert summary["metrics"]["picp"]["mean"] == 0.9
    assert summary["metrics"]["best_iteration"]["mean"] == 110.0
    assert "warnings" not in summary["metrics"]
