"""Target-aware split conformal intervals for explicit cycle-life semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import CycleLifePrediction, PredictionTarget
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest


class CycleLifeConformalCalibration(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    target: PredictionTarget
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    residual_quantile_cycle: float = Field(ge=0, allow_inf_nan=False)
    calibration_cell_count: int = Field(gt=0)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_is_explicit(self) -> CycleLifeConformalCalibration:
        if self.target not in {
            PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            PredictionTarget.UNIFIED_EOL80_CYCLE,
        }:
            raise ValueError("cycle-life calibration requires an explicit target")
        return self


class CycleLifePredictionInterval(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: CycleLifePrediction
    lower_cycle: float = Field(ge=0, allow_inf_nan=False)
    upper_cycle: float = Field(ge=0, allow_inf_nan=False)
    calibration: CycleLifeConformalCalibration

    @model_validator(mode="after")
    def interval_is_ordered_and_bound(self) -> CycleLifePredictionInterval:
        if not self.lower_cycle <= self.prediction.predicted_cycle <= self.upper_cycle:
            raise ValueError("cycle-life point prediction must lie inside its interval")
        if self.lower_cycle < self.prediction.cutoff_cycle:
            raise ValueError("cycle-life interval cannot precede the observation cutoff")
        _require_context(self.prediction, self.calibration)
        return self


class CycleLifeIntervalCoverage(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: PredictionTarget
    evaluated_cell_count: int = Field(gt=0)
    picp: float = Field(ge=0, le=1, allow_inf_nan=False)
    mpiw_cycle: float = Field(ge=0, allow_inf_nan=False)
    calibration_cell_count: int = Field(gt=0)
    warnings: tuple[str, ...] = ()


def calibrate_cycle_life_conformal(
    calibration_predictions: Sequence[CycleLifePrediction],
    *,
    split_manifest: SplitManifest,
    alpha: float,
) -> CycleLifeConformalCalibration:
    cohort = tuple(calibration_predictions)
    if not cohort:
        raise ValueError("at least one calibration-cell prediction is required")
    if not 0 < alpha < 1 or not math.isfinite(alpha):
        raise ValueError("conformal alpha must be finite and between zero and one")
    reference = cohort[0]
    seen: set[str] = set()
    residuals: list[float] = []
    for prediction in cohort:
        if prediction.cell_id in seen:
            raise ValueError("duplicate calibration cell_id")
        seen.add(prediction.cell_id)
        if prediction.dataset_id != split_manifest.dataset_id:
            raise ValueError("calibration dataset does not match the split manifest")
        if prediction.cell_id not in split_manifest.calibration:
            raise ValueError("prediction cell is outside the calibration split")
        if prediction.right_censored or prediction.observed_cycle is None:
            raise ValueError("calibration requires an observed non-censored cycle")
        _require_prediction_context(prediction, reference)
        residuals.append(abs(prediction.predicted_cycle - prediction.observed_cycle))
    residuals.sort()
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - alpha)))
    return CycleLifeConformalCalibration(
        dataset_id=reference.dataset_id,
        cutoff_cycle=reference.cutoff_cycle,
        target=reference.target,
        alpha=alpha,
        residual_quantile_cycle=residuals[rank - 1],
        calibration_cell_count=len(cohort),
        feature_version=reference.feature_version,
        split_version=reference.split_version,
        model_version=reference.model_version,
        data_version=reference.data_version,
    )


def make_cycle_life_interval(
    prediction: CycleLifePrediction,
    calibration: CycleLifeConformalCalibration,
) -> CycleLifePredictionInterval:
    _require_context(prediction, calibration)
    radius = calibration.residual_quantile_cycle
    return CycleLifePredictionInterval(
        prediction=prediction,
        lower_cycle=max(float(prediction.cutoff_cycle), prediction.predicted_cycle - radius),
        upper_cycle=prediction.predicted_cycle + radius,
        calibration=calibration,
    )


def evaluate_cycle_life_interval_coverage(
    intervals: Sequence[CycleLifePredictionInterval],
    *,
    split_manifest: SplitManifest,
) -> CycleLifeIntervalCoverage:
    cohort = tuple(intervals)
    if not cohort:
        raise ValueError("at least one test interval is required")
    reference = cohort[0]
    covered = 0
    widths: list[float] = []
    seen: set[str] = set()
    for interval in cohort:
        prediction = interval.prediction
        if prediction.cell_id in seen:
            raise ValueError("duplicate test interval cell_id")
        seen.add(prediction.cell_id)
        if prediction.dataset_id != split_manifest.dataset_id:
            raise ValueError("test interval dataset does not match the split manifest")
        if prediction.cell_id not in split_manifest.test:
            raise ValueError("interval cell is outside the test split")
        if prediction.right_censored or prediction.observed_cycle is None:
            raise ValueError("coverage requires an observed non-censored test cycle")
        if interval.calibration != reference.calibration:
            raise ValueError("test intervals must share one conformal calibration")
        observed = prediction.observed_cycle
        covered += int(interval.lower_cycle <= observed <= interval.upper_cycle)
        widths.append(interval.upper_cycle - interval.lower_cycle)
    warnings = (
        ("SMALL_CALIBRATION_COHORT",)
        if reference.calibration.calibration_cell_count < 20
        else ()
    )
    return CycleLifeIntervalCoverage(
        target=reference.prediction.target,
        evaluated_cell_count=len(cohort),
        picp=covered / len(cohort),
        mpiw_cycle=sum(widths) / len(widths),
        calibration_cell_count=reference.calibration.calibration_cell_count,
        warnings=warnings,
    )


def _require_prediction_context(
    prediction: CycleLifePrediction,
    reference: CycleLifePrediction,
) -> None:
    for field in (
        "dataset_id",
        "cutoff_cycle",
        "target",
        "feature_version",
        "split_version",
        "model_version",
        "data_version",
    ):
        if getattr(prediction, field) != getattr(reference, field):
            raise ValueError(f"calibration predictions must share {field}")


def _require_context(
    prediction: CycleLifePrediction,
    calibration: CycleLifeConformalCalibration,
) -> None:
    for field in (
        "dataset_id",
        "cutoff_cycle",
        "target",
        "feature_version",
        "split_version",
        "model_version",
        "data_version",
    ):
        if getattr(prediction, field) != getattr(calibration, field):
            raise ValueError(f"prediction {field} must match conformal calibration")
