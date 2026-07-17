from __future__ import annotations

import pytest

from quanxin_life.core import CycleLifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.uncertainty.cycle_life_conformal import (
    calibrate_cycle_life_conformal,
    evaluate_cycle_life_interval_coverage,
    make_cycle_life_interval,
)


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("train",),
        validation=("validation",),
        calibration=("cal-1", "cal-2", "cal-3", "cal-4"),
        test=("test-1", "test-2"),
    )


def _prediction(cell_id: str, predicted: float, observed: int) -> CycleLifePrediction:
    return CycleLifePrediction(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        predicted_cycle=predicted,
        observed_cycle=observed,
        right_censored=False,
        feature_version="curve-v1",
        split_version="split-v1",
        model_version="xgboost-v1",
        data_version="matr-v1",
    )


def test_official_cycle_life_conformal_reports_picp_mpiw_and_small_cohort_warning() -> None:
    calibration = calibrate_cycle_life_conformal(
        (
            _prediction("cal-1", 100.0, 110),
            _prediction("cal-2", 200.0, 180),
            _prediction("cal-3", 300.0, 330),
            _prediction("cal-4", 400.0, 360),
        ),
        split_manifest=_split(),
        alpha=0.1,
    )
    intervals = (
        make_cycle_life_interval(_prediction("test-1", 200.0, 220), calibration),
        make_cycle_life_interval(_prediction("test-2", 300.0, 360), calibration),
    )
    coverage = evaluate_cycle_life_interval_coverage(
        intervals,
        split_manifest=_split(),
    )

    assert calibration.target is PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert calibration.residual_quantile_cycle == 40.0
    assert coverage.picp == pytest.approx(0.5)
    assert coverage.mpiw_cycle == pytest.approx(80.0)
    assert "SMALL_CALIBRATION_COHORT" in coverage.warnings


def test_cycle_life_conformal_rejects_non_calibration_or_mixed_target_cells() -> None:
    valid = _prediction("cal-1", 100.0, 110)
    with pytest.raises(ValueError, match="calibration split"):
        calibrate_cycle_life_conformal(
            (_prediction("test-1", 100.0, 110),),
            split_manifest=_split(),
            alpha=0.1,
        )
    mixed = valid.model_copy(update={"target": PredictionTarget.UNIFIED_EOL80_CYCLE})
    with pytest.raises(ValueError, match="target"):
        calibrate_cycle_life_conformal(
            (valid, mixed.model_copy(update={"cell_id": "cal-2"})),
            split_manifest=_split(),
            alpha=0.1,
        )
