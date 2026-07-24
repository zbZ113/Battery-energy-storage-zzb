from __future__ import annotations

import pytest

from quanxin_life.core import CycleLifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest


def _manifest() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("train-1",),
        validation=("validation-1",),
        calibration=("calibration-1", "calibration-2", "calibration-3"),
        test=("test-1", "test-2"),
    )


def _prediction(
    cell_id: str,
    *,
    predicted: float,
    observed: int,
    model_version: str = "cyclepatch-direct-ensemble-v1",
) -> CycleLifePrediction:
    return CycleLifePrediction(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        predicted_cycle=predicted,
        observed_cycle=observed,
        right_censored=False,
        feature_version="advanced-cyclepatch-v1",
        split_version="matr-three-batch-split-v1",
        model_version=model_version,
        data_version="matr-three-batch-v1",
    )


def _scaled(
    cell_id: str,
    *,
    predicted: float,
    observed: int,
    scale: float,
):
    from quanxin_life.uncertainty.normalized_cycle_life_conformal import (
        ScaledCycleLifePrediction,
    )

    return ScaledCycleLifePrediction(
        prediction=_prediction(cell_id, predicted=predicted, observed=observed),
        difficulty_scale_cycle=scale,
        scale_version="five-seed-sample-standard-deviation-v1",
    )


def _calibration_predictions():
    return (
        _scaled("calibration-1", predicted=100.0, observed=110, scale=5.0),
        _scaled("calibration-2", predicted=200.0, observed=180, scale=10.0),
        _scaled("calibration-3", predicted=300.0, observed=330, scale=15.0),
    )


def test_calibrates_normalized_explicit_cycle_life_predictions() -> None:
    from quanxin_life.uncertainty.normalized_cycle_life_conformal import (
        calibrate_normalized_cycle_life_conformal,
    )

    calibration = calibrate_normalized_cycle_life_conformal(
        _calibration_predictions(),
        split_manifest=_manifest(),
        alpha=0.25,
    )

    assert calibration.target is PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert calibration.normalized_score_quantile == 2.0
    assert calibration.calibration_cell_count == 3
    assert calibration.scale_version == "five-seed-sample-standard-deviation-v1"


def test_interval_width_uses_model_scale_and_respects_cutoff() -> None:
    from quanxin_life.uncertainty.normalized_cycle_life_conformal import (
        calibrate_normalized_cycle_life_conformal,
        make_normalized_cycle_life_interval,
    )

    calibration = calibrate_normalized_cycle_life_conformal(
        _calibration_predictions(),
        split_manifest=_manifest(),
        alpha=0.25,
    )
    narrow = make_normalized_cycle_life_interval(
        _scaled("test-1", predicted=25.0, observed=30, scale=5.0),
        calibration,
    )
    wide = make_normalized_cycle_life_interval(
        _scaled("test-2", predicted=100.0, observed=120, scale=15.0),
        calibration,
    )

    assert narrow.lower_cycle == 20.0
    assert narrow.upper_cycle == 35.0
    assert wide.upper_cycle - wide.lower_cycle > narrow.upper_cycle - narrow.lower_cycle


def test_calibration_rejects_wrong_partition_duplicate_and_context_mixing() -> None:
    from quanxin_life.uncertainty.normalized_cycle_life_conformal import (
        ScaledCycleLifePrediction,
        calibrate_normalized_cycle_life_conformal,
    )

    with pytest.raises(ValueError, match="calibration split"):
        calibrate_normalized_cycle_life_conformal(
            (_scaled("train-1", predicted=100.0, observed=110, scale=5.0),),
            split_manifest=_manifest(),
            alpha=0.1,
        )

    with pytest.raises(ValueError, match="duplicate"):
        calibrate_normalized_cycle_life_conformal(
            (*_calibration_predictions(), _calibration_predictions()[0]),
            split_manifest=_manifest(),
            alpha=0.1,
        )

    mixed = ScaledCycleLifePrediction(
        prediction=_prediction(
            "calibration-3",
            predicted=300.0,
            observed=330,
            model_version="other-model",
        ),
        difficulty_scale_cycle=15.0,
        scale_version="five-seed-sample-standard-deviation-v1",
    )
    with pytest.raises(ValueError, match="model_version"):
        calibrate_normalized_cycle_life_conformal(
            (*_calibration_predictions()[:2], mixed),
            split_manifest=_manifest(),
            alpha=0.1,
        )


def test_coverage_requires_test_cells_and_emits_small_cohort_warning() -> None:
    from quanxin_life.uncertainty.normalized_cycle_life_conformal import (
        calibrate_normalized_cycle_life_conformal,
        evaluate_normalized_cycle_life_interval_coverage,
        make_normalized_cycle_life_interval,
    )

    calibration = calibrate_normalized_cycle_life_conformal(
        _calibration_predictions(),
        split_manifest=_manifest(),
        alpha=0.25,
    )
    intervals = (
        make_normalized_cycle_life_interval(
            _scaled("test-1", predicted=100.0, observed=110, scale=5.0),
            calibration,
        ),
        make_normalized_cycle_life_interval(
            _scaled("test-2", predicted=200.0, observed=250, scale=10.0),
            calibration,
        ),
    )

    coverage = evaluate_normalized_cycle_life_interval_coverage(
        intervals,
        split_manifest=_manifest(),
    )

    assert coverage.evaluated_cell_count == 2
    assert coverage.picp == 0.5
    assert coverage.mpiw_cycle == 30.0
    assert coverage.warnings == ("SMALL_CALIBRATION_COHORT",)

    invalid = make_normalized_cycle_life_interval(
        _scaled("calibration-1", predicted=100.0, observed=110, scale=5.0),
        calibration,
    )
    with pytest.raises(ValueError, match="test split"):
        evaluate_normalized_cycle_life_interval_coverage(
            (invalid,),
            split_manifest=_manifest(),
        )
