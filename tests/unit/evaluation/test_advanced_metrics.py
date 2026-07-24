from __future__ import annotations

import math

import pytest

from quanxin_life.evaluation.advanced_metrics import (
    RulPredictionPoint,
    SohPredictionPoint,
    compare_rul_models,
    compare_soh_models,
    summarize_rul_predictions,
    summarize_soh_predictions,
)


def _point(
    *,
    family: str,
    seed: int,
    cell_id: str,
    actual: float,
    predicted: float,
) -> RulPredictionPoint:
    return RulPredictionPoint(
        run_id=f"{family}-s{seed}",
        family=family,
        candidate_id=f"{family}-candidate",
        cutoff_cycle=100,
        seed=seed,
        cell_id=cell_id,
        batch_date="2017-05-12",
        true_cycle_life=actual,
        predicted_cycle_life=predicted,
    )


def test_summarize_rul_predictions_reports_regression_and_tolerance_metrics() -> None:
    points = (
        _point(family="direct", seed=38, cell_id="a", actual=100, predicted=110),
        _point(family="direct", seed=38, cell_id="b", actual=200, predicted=180),
        _point(family="direct", seed=39, cell_id="a", actual=100, predicted=110),
        _point(family="direct", seed=39, cell_id="b", actual=200, predicted=180),
    )

    summary = summarize_rul_predictions(
        points,
        bootstrap_resamples=200,
        bootstrap_seed=20260712,
    )

    assert summary.cell_count == 2
    assert summary.seed_count == 2
    assert summary.mae_cycle == pytest.approx(15.0)
    assert summary.rmse_cycle == pytest.approx(math.sqrt(250.0))
    assert summary.mape_percent == pytest.approx(10.0)
    assert summary.r2 == pytest.approx(0.9)
    assert summary.accuracy_at_tolerance_percent == {
        "5": pytest.approx(0.0),
        "10": pytest.approx(100.0),
        "15": pytest.approx(100.0),
        "20": pytest.approx(100.0),
    }
    assert summary.absolute_error_quantiles_cycle["50"] == pytest.approx(15.0)
    assert summary.bootstrap_intervals["mae_cycle"].estimate == pytest.approx(15.0)
    assert summary.bootstrap_intervals["mae_cycle"].lower <= 15.0
    assert summary.bootstrap_intervals["mae_cycle"].upper >= 15.0


def test_summarize_rul_predictions_requires_rectangular_cell_seed_cohort() -> None:
    points = (
        _point(family="direct", seed=38, cell_id="a", actual=100, predicted=110),
        _point(family="direct", seed=38, cell_id="b", actual=200, predicted=180),
        _point(family="direct", seed=39, cell_id="a", actual=100, predicted=109),
    )

    with pytest.raises(ValueError, match="same cell_id cohort"):
        summarize_rul_predictions(points, bootstrap_resamples=20)


def test_summarize_rul_predictions_rejects_inconsistent_observed_lifetime() -> None:
    points = (
        _point(family="direct", seed=38, cell_id="a", actual=100, predicted=110),
        _point(family="direct", seed=39, cell_id="a", actual=101, predicted=109),
    )

    with pytest.raises(ValueError, match="true cycle life"):
        summarize_rul_predictions(points, bootstrap_resamples=20)


def test_compare_rul_models_uses_paired_cells_and_is_deterministic() -> None:
    direct = (
        _point(family="direct", seed=38, cell_id="a", actual=100, predicted=105),
        _point(family="direct", seed=38, cell_id="b", actual=200, predicted=190),
        _point(family="direct", seed=39, cell_id="a", actual=100, predicted=104),
        _point(family="direct", seed=39, cell_id="b", actual=200, predicted=191),
    )
    batlinet = (
        _point(family="batlinet", seed=38, cell_id="a", actual=100, predicted=120),
        _point(family="batlinet", seed=38, cell_id="b", actual=200, predicted=180),
        _point(family="batlinet", seed=39, cell_id="a", actual=100, predicted=118),
        _point(family="batlinet", seed=39, cell_id="b", actual=200, predicted=182),
    )

    first = compare_rul_models(
        direct,
        batlinet,
        bootstrap_resamples=200,
        bootstrap_seed=20260712,
    )
    second = compare_rul_models(
        direct,
        batlinet,
        bootstrap_resamples=200,
        bootstrap_seed=20260712,
    )

    assert first == second
    assert first.left_family == "direct"
    assert first.right_family == "batlinet"
    assert first.mean_absolute_error_delta_cycle < 0
    assert first.left_cell_win_rate_percent == pytest.approx(100.0)
    assert first.bootstrap_interval.estimate == pytest.approx(
        first.mean_absolute_error_delta_cycle
    )


def test_compare_rul_models_rejects_different_cutoffs() -> None:
    left = _point(family="direct", seed=38, cell_id="a", actual=100, predicted=105)
    right = _point(family="batlinet", seed=38, cell_id="a", actual=100, predicted=110)
    right = right.model_copy(update={"cutoff_cycle": 150})

    with pytest.raises(ValueError, match="same cutoff_cycle"):
        compare_rul_models((left,), (right,), bootstrap_resamples=20)


def _soh_point(
    *,
    family: str,
    seed: int,
    cell_id: str,
    cycle: int,
    actual: float,
    predicted: float,
) -> SohPredictionPoint:
    return SohPredictionPoint(
        run_id=f"{family}-s{seed}",
        family=family,
        candidate_id=f"{family}-candidate",
        cutoff_cycle=20,
        seed=seed,
        cell_id=cell_id,
        batch_date="2017-05-12",
        cycle=cycle,
        true_soh=actual,
        predicted_soh=predicted,
    )


def test_summarize_soh_predictions_reports_tail_horizon_and_shape_metrics() -> None:
    points = tuple(
        _soh_point(
            family="patch",
            seed=seed,
            cell_id=cell_id,
            cycle=cycle,
            actual=actual,
            predicted=predicted,
        )
        for seed in (38, 39)
        for cell_id in ("a", "b")
        for cycle, actual, predicted in ((21, 1.0, 1.01), (22, 0.9, 0.88))
    )

    summary = summarize_soh_predictions(
        points,
        bootstrap_resamples=200,
        bootstrap_seed=20260712,
    )

    assert summary.cell_count == 2
    assert summary.seed_count == 2
    assert summary.forecast_point_count_per_seed == 4
    assert summary.forecast_horizon_cycle_min == 1
    assert summary.forecast_horizon_cycle_max == 2
    assert summary.mae_soh == pytest.approx(0.015)
    assert summary.rmse_soh == pytest.approx(math.sqrt(0.00025))
    assert summary.mean_signed_error_soh == pytest.approx(-0.005)
    assert summary.monotonic_violation_rate_percent == pytest.approx(0.0)
    assert summary.accuracy_at_soh_point_tolerance_percent["1"] == pytest.approx(50.0)
    assert summary.accuracy_at_soh_point_tolerance_percent["2"] == pytest.approx(100.0)
    assert summary.cell_mae_quantiles_soh["100"] == pytest.approx(0.015)
    assert summary.bootstrap_intervals["mae_soh"].estimate == pytest.approx(0.015)


def test_summarize_soh_predictions_rejects_incomplete_seed_trajectory() -> None:
    points = (
        _soh_point(
            family="patch", seed=38, cell_id="a", cycle=21, actual=1.0, predicted=0.99
        ),
        _soh_point(
            family="patch", seed=38, cell_id="a", cycle=22, actual=0.9, predicted=0.89
        ),
        _soh_point(
            family="patch", seed=39, cell_id="a", cycle=21, actual=1.0, predicted=0.99
        ),
    )

    with pytest.raises(ValueError, match="same cell/cycle trajectory"):
        summarize_soh_predictions(points, bootstrap_resamples=20)


def test_compare_soh_models_uses_paired_cell_trajectories() -> None:
    left = tuple(
        _soh_point(
            family="patch",
            seed=seed,
            cell_id=cell_id,
            cycle=cycle,
            actual=actual,
            predicted=actual - 0.01,
        )
        for seed in (38, 39)
        for cell_id in ("a", "b")
        for cycle, actual in ((21, 1.0), (22, 0.9))
    )
    right = tuple(
        _soh_point(
            family="current",
            seed=seed,
            cell_id=cell_id,
            cycle=cycle,
            actual=actual,
            predicted=actual - 0.02,
        )
        for seed in (38, 39)
        for cell_id in ("a", "b")
        for cycle, actual in ((21, 1.0), (22, 0.9))
    )

    comparison = compare_soh_models(
        left,
        right,
        bootstrap_resamples=200,
        bootstrap_seed=20260712,
    )

    assert comparison.left_family == "patch"
    assert comparison.mean_cell_mae_delta_soh == pytest.approx(-0.01)
    assert comparison.left_cell_win_rate_percent == pytest.approx(100.0)
    assert comparison.left_only_point_count == 0
    assert comparison.right_only_point_count == 0
    assert comparison.comparison_point_count == 8


def test_compare_soh_models_uses_common_cycles_and_reports_dropped_points() -> None:
    left = (
        _soh_point(
            family="patch", seed=38, cell_id="a", cycle=21, actual=1.0, predicted=0.99
        ),
        _soh_point(
            family="patch", seed=38, cell_id="a", cycle=22, actual=0.9, predicted=0.89
        ),
    )
    right = (
        _soh_point(
            family="current", seed=38, cell_id="a", cycle=21, actual=1.0, predicted=0.98
        ),
        _soh_point(
            family="current", seed=38, cell_id="a", cycle=22, actual=0.9, predicted=0.88
        ),
        _soh_point(
            family="current", seed=38, cell_id="a", cycle=23, actual=0.8, predicted=0.78
        ),
    )

    comparison = compare_soh_models(left, right, bootstrap_resamples=20)

    assert comparison.comparison_point_count == 2
    assert comparison.left_only_point_count == 0
    assert comparison.right_only_point_count == 1
    assert comparison.mean_cell_mae_delta_soh == pytest.approx(-0.01)
