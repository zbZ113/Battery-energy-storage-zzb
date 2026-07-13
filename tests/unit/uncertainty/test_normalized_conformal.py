from __future__ import annotations

import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest


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
) -> LifePrediction:
    return LifePrediction(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        predicted_eol_cycle=predicted,
        observed_eol_cycle=observed,
        right_censored=observed is None if right_censored is None else right_censored,
        feature_version="early-cycle-v1",
        split_version="split-v1",
        model_version="xgboost-v1",
        data_version="data-v1",
    )


def _scaled_prediction(cell_id: str, *, predicted: float, observed: int, scale: float):
    from quanxin_life.uncertainty.normalized_conformal import ScaledLifePrediction

    return ScaledLifePrediction(
        prediction=_prediction(cell_id, predicted=predicted, observed=observed),
        difficulty_scale_cycle=scale,
        scale_version="residual-scale-v1",
    )


def _calibration_predictions():
    return (
        _scaled_prediction("calibration-1", predicted=100.0, observed=110, scale=5.0),
        _scaled_prediction("calibration-2", predicted=200.0, observed=180, scale=10.0),
        _scaled_prediction("calibration-3", predicted=300.0, observed=330, scale=15.0),
    )


def test_calibrates_normalized_residuals_from_calibration_cells_only() -> None:
    from quanxin_life.uncertainty.normalized_conformal import calibrate_normalized_conformal

    calibration = calibrate_normalized_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )

    # Normalized residuals are 2, 2 and 2; finite-sample rank selects 2.
    assert calibration.normalized_score_quantile == 2.0
    assert calibration.calibration_cell_count == 3
    assert calibration.scale_version == "residual-scale-v1"


def test_interval_width_scales_with_model_difficulty_without_changing_point_prediction() -> None:
    from quanxin_life.uncertainty.normalized_conformal import (
        calibrate_normalized_conformal,
        make_normalized_prediction_interval,
    )

    calibration = calibrate_normalized_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    narrow = make_normalized_prediction_interval(
        _scaled_prediction("test-1", predicted=100.0, observed=110, scale=5.0), calibration
    )
    wide = make_normalized_prediction_interval(
        _scaled_prediction("test-1", predicted=100.0, observed=110, scale=15.0), calibration
    )

    assert narrow.point_prediction_cycle == wide.point_prediction_cycle == 100.0
    assert wide.upper_eol_cycle - wide.lower_eol_cycle > narrow.upper_eol_cycle - narrow.lower_eol_cycle
    assert narrow.lower_eol_cycle >= 20.0


@pytest.mark.parametrize(
    "scaled_prediction",
    [
        lambda: _scaled_prediction("train-1", predicted=100.0, observed=110, scale=5.0),
        lambda: _scaled_prediction("calibration-1", predicted=100.0, observed=110, scale=0.0),
    ],
)
def test_rejects_non_calibration_cells_and_nonpositive_scales(scaled_prediction) -> None:
    from quanxin_life.uncertainty.normalized_conformal import calibrate_normalized_conformal

    with pytest.raises(ValueError):
        calibrate_normalized_conformal((scaled_prediction(),), split_manifest=_manifest())


def test_rejects_scale_version_mismatch_when_constructing_interval() -> None:
    from quanxin_life.uncertainty.normalized_conformal import (
        ScaledLifePrediction,
        calibrate_normalized_conformal,
        make_normalized_prediction_interval,
    )

    calibration = calibrate_normalized_conformal(
        _calibration_predictions(), split_manifest=_manifest(), alpha=0.25
    )
    incompatible = ScaledLifePrediction(
        prediction=_prediction("test-1", predicted=100.0),
        difficulty_scale_cycle=5.0,
        scale_version="different-scale-v1",
    )

    with pytest.raises(ValueError, match="scale_version"):
        make_normalized_prediction_interval(incompatible, calibration)
