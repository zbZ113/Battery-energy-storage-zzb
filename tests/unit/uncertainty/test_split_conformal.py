from __future__ import annotations

import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.uncertainty.split_conformal import (
    calibrate_split_conformal,
    evaluate_interval_coverage,
    make_prediction_interval,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("train-1",),
        validation=("validation-1",),
        calibration=("calibration-1", "calibration-2", "calibration-3"),
        test=("test-1",),
    )


def _prediction(
    cell_id: str,
    *,
    predicted: float,
    observed: int | None = None,
    right_censored: bool | None = None,
    dataset_id: str = "MATR",
    cutoff: int = 20,
    feature_version: str = "early-cycle-v1",
    split_version: str = "split-v1",
    model_version: str = "xgboost-v1",
    data_version: str = "data-v1",
) -> LifePrediction:
    is_censored = observed is None if right_censored is None else right_censored
    return LifePrediction(
        dataset_id=dataset_id,
        cell_id=cell_id,
        cutoff_cycle=cutoff,
        predicted_eol_cycle=predicted,
        observed_eol_cycle=observed,
        right_censored=is_censored,
        feature_version=feature_version,
        split_version=split_version,
        model_version=model_version,
        data_version=data_version,
    )


def _calibration_predictions() -> tuple[LifePrediction, ...]:
    return (
        _prediction("calibration-1", predicted=100.0, observed=110),
        _prediction("calibration-2", predicted=200.0, observed=180),
        _prediction("calibration-3", predicted=300.0, observed=330),
    )


def test_calibrates_explicit_cell_disjoint_eol80_predictions() -> None:
    calibration = calibrate_split_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )

    # n=3, ceil((n + 1) * (1 - alpha)) = 3: the third sorted residual is 30.
    assert calibration.residual_quantile_cycle == 30.0
    assert calibration.calibration_cell_count == 3
    assert calibration.alpha == 0.25
    assert calibration.feature_version == "early-cycle-v1"


@pytest.mark.parametrize(
    ("prediction", "expected_message"),
    [
        (_prediction("train-1", predicted=100.0, observed=110), "calibration split"),
        (
            _prediction("calibration-1", predicted=100.0, observed=None, right_censored=True),
            "right-censored",
        ),
        (_prediction("calibration-1", predicted=100.0, observed=110), "duplicate"),
    ],
)
def test_calibration_rejects_non_calibration_censored_or_duplicate_cells(
    prediction: LifePrediction, expected_message: str
) -> None:
    calibration_predictions = list(_calibration_predictions())
    if expected_message == "duplicate":
        calibration_predictions.append(prediction)
    else:
        calibration_predictions[0] = prediction

    with pytest.raises(ValueError, match=expected_message):
        calibrate_split_conformal(calibration_predictions, split_manifest=_manifest(), alpha=0.1)


def test_calibration_rejects_context_mixing() -> None:
    mixed = list(_calibration_predictions())
    mixed[-1] = _prediction(
        "calibration-3", predicted=300.0, observed=330, model_version="other-model"
    )

    with pytest.raises(ValueError, match="model_version"):
        calibrate_split_conformal(mixed, split_manifest=_manifest(), alpha=0.1)


def test_interval_matches_calibration_context_and_respects_cutoff() -> None:
    calibration = calibrate_split_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    prediction = _prediction("test-1", predicted=25.0)

    interval = make_prediction_interval(prediction, calibration=calibration)

    assert interval.lower_eol_cycle == 20.0
    assert interval.point_prediction_cycle == 25.0
    assert interval.upper_eol_cycle == 55.0
    assert interval.calibration == calibration


def test_interval_rejects_version_mismatch() -> None:
    calibration = calibrate_split_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    prediction = _prediction("test-1", predicted=120.0, data_version="other-data")

    with pytest.raises(ValueError, match="data_version"):
        make_prediction_interval(prediction, calibration=calibration)


def test_evaluates_homogeneous_observed_cell_unique_interval_cohort() -> None:
    calibration = calibrate_split_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    observed_predictions = (
        _prediction("test-1", predicted=100.0, observed=115),
        _prediction("test-2", predicted=200.0, observed=230),
    )
    intervals = tuple(
        make_prediction_interval(prediction, calibration=calibration)
        for prediction in observed_predictions
    )

    coverage = evaluate_interval_coverage(intervals, observed_predictions)

    assert coverage.evaluated_cell_count == 2
    assert coverage.picp == 1.0
    assert coverage.mpiw == 60.0
    assert coverage.calibration == calibration


def test_coverage_rejects_missing_observation_or_version_mixing() -> None:
    calibration = calibrate_split_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    interval = make_prediction_interval(_prediction("test-1", predicted=100.0), calibration)

    with pytest.raises(ValueError, match="right-censored"):
        evaluate_interval_coverage((interval,), (_prediction("test-1", predicted=100.0),))

    observed = _prediction("test-1", predicted=100.0, observed=110, model_version="other")
    with pytest.raises(ValueError, match="model_version"):
        evaluate_interval_coverage((interval,), (observed,))
