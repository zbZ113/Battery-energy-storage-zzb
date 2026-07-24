"""Scale-aware conformal intervals for explicit cycle-life targets."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import CycleLifePrediction, PredictionTarget
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest

_CONTEXT_FIELDS = (
    "dataset_id",
    "cutoff_cycle",
    "target",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)


class ScaledCycleLifePrediction(ContractModel):
    """One explicit cycle-life prediction and a model-produced difficulty scale."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    prediction: CycleLifePrediction
    difficulty_scale_cycle: float = Field(gt=0, allow_inf_nan=False)
    scale_version: str = Field(min_length=1)


class NormalizedCycleLifeConformalCalibration(ContractModel):
    """Finite-sample normalized calibration for one homogeneous model cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dataset_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    target: PredictionTarget
    alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    normalized_score_quantile: float = Field(ge=0, allow_inf_nan=False)
    calibration_cell_count: int = Field(gt=0)
    scale_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_is_explicit(self) -> NormalizedCycleLifeConformalCalibration:
        if self.target not in {
            PredictionTarget.UNIFIED_EOL80_CYCLE,
            PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        }:
            raise ValueError("normalized cycle-life calibration requires an explicit target")
        return self


class NormalizedCycleLifePredictionInterval(ContractModel):
    """One scale-aware interval bound to its point prediction and calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    prediction: CycleLifePrediction
    lower_cycle: float = Field(ge=0, allow_inf_nan=False)
    upper_cycle: float = Field(ge=0, allow_inf_nan=False)
    difficulty_scale_cycle: float = Field(gt=0, allow_inf_nan=False)
    scale_version: str = Field(min_length=1)
    calibration: NormalizedCycleLifeConformalCalibration

    @model_validator(mode="after")
    def interval_is_ordered_and_bound(self) -> NormalizedCycleLifePredictionInterval:
        if not self.lower_cycle <= self.prediction.predicted_cycle <= self.upper_cycle:
            raise ValueError("cycle-life point prediction must lie inside its interval")
        if self.lower_cycle < self.prediction.cutoff_cycle:
            raise ValueError("cycle-life interval cannot precede the observation cutoff")
        if self.scale_version != self.calibration.scale_version:
            raise ValueError("interval scale_version must match conformal calibration")
        _require_context(self.prediction, self.calibration)
        return self


class NormalizedCycleLifeIntervalCoverage(ContractModel):
    """Cell-level PICP and MPIW for normalized explicit cycle-life intervals."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target: PredictionTarget
    evaluated_cell_count: int = Field(gt=0)
    picp: float = Field(ge=0, le=1, allow_inf_nan=False)
    mpiw_cycle: float = Field(ge=0, allow_inf_nan=False)
    calibration_cell_count: int = Field(gt=0)
    warnings: tuple[str, ...] = ()


def calibrate_normalized_cycle_life_conformal(
    calibration_predictions: Sequence[ScaledCycleLifePrediction],
    *,
    split_manifest: SplitManifest,
    alpha: float,
) -> NormalizedCycleLifeConformalCalibration:
    """Calibrate normalized residuals using calibration cells only."""

    cohort = tuple(calibration_predictions)
    if not cohort:
        raise ValueError("at least one calibration-cell prediction is required")
    if not 0 < alpha < 1 or not math.isfinite(alpha):
        raise ValueError("conformal alpha must be finite and between zero and one")
    reference = cohort[0]
    seen: set[str] = set()
    scores: list[float] = []
    for scaled_prediction in cohort:
        prediction = scaled_prediction.prediction
        if prediction.cell_id in seen:
            raise ValueError("duplicate calibration cell_id")
        seen.add(prediction.cell_id)
        if prediction.dataset_id != split_manifest.dataset_id:
            raise ValueError("calibration dataset does not match the split manifest")
        if prediction.cell_id not in split_manifest.calibration:
            raise ValueError("prediction cell is outside the calibration split")
        if prediction.right_censored or prediction.observed_cycle is None:
            raise ValueError("calibration requires an observed non-censored cycle")
        if scaled_prediction.scale_version != reference.scale_version:
            raise ValueError("calibration predictions must share scale_version")
        _require_prediction_context(prediction, reference.prediction)
        scores.append(
            abs(prediction.predicted_cycle - prediction.observed_cycle)
            / scaled_prediction.difficulty_scale_cycle
        )
    scores.sort()
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    prediction = reference.prediction
    return NormalizedCycleLifeConformalCalibration(
        dataset_id=prediction.dataset_id,
        cutoff_cycle=prediction.cutoff_cycle,
        target=prediction.target,
        alpha=alpha,
        normalized_score_quantile=scores[rank - 1],
        calibration_cell_count=len(cohort),
        scale_version=reference.scale_version,
        feature_version=prediction.feature_version,
        split_version=prediction.split_version,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
    )


def make_normalized_cycle_life_interval(
    scaled_prediction: ScaledCycleLifePrediction,
    calibration: NormalizedCycleLifeConformalCalibration,
) -> NormalizedCycleLifePredictionInterval:
    """Expand a point prediction by its versioned model difficulty scale."""

    prediction = scaled_prediction.prediction
    _require_context(prediction, calibration)
    if scaled_prediction.scale_version != calibration.scale_version:
        raise ValueError("scale_version must match normalized conformal calibration")
    radius = (
        calibration.normalized_score_quantile
        * scaled_prediction.difficulty_scale_cycle
    )
    return NormalizedCycleLifePredictionInterval(
        prediction=prediction,
        lower_cycle=max(float(prediction.cutoff_cycle), prediction.predicted_cycle - radius),
        upper_cycle=prediction.predicted_cycle + radius,
        difficulty_scale_cycle=scaled_prediction.difficulty_scale_cycle,
        scale_version=scaled_prediction.scale_version,
        calibration=calibration,
    )


def evaluate_normalized_cycle_life_interval_coverage(
    intervals: Sequence[NormalizedCycleLifePredictionInterval],
    *,
    split_manifest: SplitManifest,
) -> NormalizedCycleLifeIntervalCoverage:
    """Evaluate empirical coverage on unique held-out test cells."""

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
        if interval.scale_version != reference.scale_version:
            raise ValueError("test intervals must share one scale_version")
        covered += int(interval.lower_cycle <= prediction.observed_cycle <= interval.upper_cycle)
        widths.append(interval.upper_cycle - interval.lower_cycle)
    warnings = (
        ("SMALL_CALIBRATION_COHORT",)
        if reference.calibration.calibration_cell_count < 20
        else ()
    )
    return NormalizedCycleLifeIntervalCoverage(
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
    for field in _CONTEXT_FIELDS:
        if getattr(prediction, field) != getattr(reference, field):
            raise ValueError(f"calibration predictions must share {field}")


def _require_context(
    prediction: CycleLifePrediction,
    calibration: NormalizedCycleLifeConformalCalibration,
) -> None:
    for field in _CONTEXT_FIELDS:
        if getattr(prediction, field) != getattr(calibration, field):
            raise ValueError(f"prediction {field} must match conformal calibration")


__all__ = [
    "NormalizedCycleLifeConformalCalibration",
    "NormalizedCycleLifeIntervalCoverage",
    "NormalizedCycleLifePredictionInterval",
    "ScaledCycleLifePrediction",
    "calibrate_normalized_cycle_life_conformal",
    "evaluate_normalized_cycle_life_interval_coverage",
    "make_normalized_cycle_life_interval",
]
