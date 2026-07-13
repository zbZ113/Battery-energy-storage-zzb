"""Cell-disjoint normalized conformal intervals for EOL80 predictions.

Unlike the constant-width split conformal baseline, this module calibrates
absolute residuals after dividing by a precomputed, model-produced difficulty
scale.  The scale is an explicit input with a version, never a number supplied
by an LLM or inferred from a test label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean

from pydantic import ConfigDict, Field

from quanxin_life.core import (
    LifePrediction,
    NormalizedConformalCalibration,
    NormalizedPredictionInterval,
    PredictionTarget,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest

_PREDICTION_CONTEXT_FIELDS = (
    "dataset_id",
    "cutoff_cycle",
    "target",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)
_CALIBRATION_VERSION_FIELDS = (
    "target",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)


class ScaledLifePrediction(ContractModel):
    """One existing model prediction and its positive difficulty scale."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    prediction: LifePrediction
    difficulty_scale_cycle: float = Field(gt=0, allow_inf_nan=False)
    scale_version: str = Field(min_length=1)


class NormalizedIntervalCoverage(ContractModel):
    """Cell-level PICP/MPIW for one homogeneous normalized-interval cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target: PredictionTarget = PredictionTarget.EOL80_CYCLE
    evaluated_cell_count: int = Field(gt=0)
    picp: float = Field(ge=0, le=1)
    mpiw: float = Field(ge=0)
    calibration: NormalizedConformalCalibration


def calibrate_normalized_conformal(
    calibration_predictions: Sequence[ScaledLifePrediction],
    *,
    split_manifest: SplitManifest,
    alpha: float = 0.10,
) -> NormalizedConformalCalibration:
    """Calibrate normalized residual scores using only manifest calibration cells."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    cohort = tuple(calibration_predictions)
    if not cohort:
        raise ValueError("at least one calibration-cell prediction is required")

    reference = cohort[0]
    seen_cell_ids: set[str] = set()
    normalized_scores: list[float] = []
    for scaled_prediction in cohort:
        _validate_calibration_prediction(
            scaled_prediction,
            split_manifest=split_manifest,
            seen_cell_ids=seen_cell_ids,
            reference=reference,
        )
        prediction = scaled_prediction.prediction
        assert prediction.observed_eol_cycle is not None
        normalized_scores.append(
            abs(prediction.predicted_eol_cycle - prediction.observed_eol_cycle)
            / scaled_prediction.difficulty_scale_cycle
        )

    normalized_scores.sort()
    finite_sample_rank = math.ceil((len(normalized_scores) + 1) * (1.0 - alpha))
    selected_index = min(finite_sample_rank, len(normalized_scores)) - 1
    return NormalizedConformalCalibration(
        target=reference.prediction.target,
        alpha=alpha,
        normalized_score_quantile=normalized_scores[selected_index],
        calibration_cell_count=len(cohort),
        scale_version=reference.scale_version,
        feature_version=reference.prediction.feature_version,
        split_version=reference.prediction.split_version,
        model_version=reference.prediction.model_version,
        data_version=reference.prediction.data_version,
    )


def make_normalized_prediction_interval(
    scaled_prediction: ScaledLifePrediction,
    calibration: NormalizedConformalCalibration,
) -> NormalizedPredictionInterval:
    """Expand one existing point estimate with a scale-aware conformal radius."""

    prediction = scaled_prediction.prediction
    if prediction.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 point predictions support normalized conformal intervals")
    _require_calibration_versions(prediction, calibration)
    if scaled_prediction.scale_version != calibration.scale_version:
        raise ValueError("scale_version must match normalized conformal calibration")
    radius = calibration.normalized_score_quantile * scaled_prediction.difficulty_scale_cycle
    lower = max(float(prediction.cutoff_cycle), prediction.predicted_eol_cycle - radius)
    upper = prediction.predicted_eol_cycle + radius
    return NormalizedPredictionInterval(
        dataset_id=prediction.dataset_id,
        cell_id=prediction.cell_id,
        cutoff_cycle=prediction.cutoff_cycle,
        target=prediction.target,
        point_prediction_cycle=prediction.predicted_eol_cycle,
        lower_eol_cycle=lower,
        upper_eol_cycle=upper,
        difficulty_scale_cycle=scaled_prediction.difficulty_scale_cycle,
        calibration=calibration,
    )


def evaluate_normalized_interval_coverage(
    intervals: Sequence[NormalizedPredictionInterval],
    observed_predictions: Sequence[LifePrediction],
) -> NormalizedIntervalCoverage:
    """Compute PICP and MPIW without treating right-censored labels as observed."""

    interval_cohort = tuple(intervals)
    observation_cohort = tuple(observed_predictions)
    if not interval_cohort:
        raise ValueError("at least one normalized prediction interval is required for coverage")
    if len(interval_cohort) != len(observation_cohort):
        raise ValueError("intervals and observed predictions must have the same cell count")

    reference = interval_cohort[0]
    interval_by_cell: dict[str, NormalizedPredictionInterval] = {}
    for interval in interval_cohort:
        _validate_interval_context(interval, reference=reference, known_cells=interval_by_cell)
        interval_by_cell[interval.cell_id] = interval

    observed_cells: set[str] = set()
    coverage: list[float] = []
    widths: list[float] = []
    for observation in observation_cohort:
        if observation.cell_id in observed_cells:
            raise ValueError(f"duplicate cell_id in observed prediction cohort: {observation.cell_id}")
        observed_cells.add(observation.cell_id)
        interval = interval_by_cell.get(observation.cell_id)
        if interval is None:
            raise ValueError("interval and observed prediction cell_ids must exactly match")
        _validate_observation_context(observation, interval=interval)
        assert observation.observed_eol_cycle is not None
        coverage.append(
            float(interval.lower_eol_cycle <= observation.observed_eol_cycle <= interval.upper_eol_cycle)
        )
        widths.append(interval.upper_eol_cycle - interval.lower_eol_cycle)

    if observed_cells != set(interval_by_cell):
        raise ValueError("interval and observed prediction cell_ids must exactly match")
    return NormalizedIntervalCoverage(
        target=reference.target,
        evaluated_cell_count=len(interval_cohort),
        picp=fmean(coverage),
        mpiw=fmean(widths),
        calibration=reference.calibration,
    )


def _validate_calibration_prediction(
    scaled_prediction: ScaledLifePrediction,
    *,
    split_manifest: SplitManifest,
    seen_cell_ids: set[str],
    reference: ScaledLifePrediction,
) -> None:
    prediction = scaled_prediction.prediction
    if prediction.cell_id in seen_cell_ids:
        raise ValueError(f"duplicate calibration cell_id: {prediction.cell_id}")
    seen_cell_ids.add(prediction.cell_id)
    if prediction.dataset_id != split_manifest.dataset_id:
        raise ValueError("calibration prediction dataset_id must match the split manifest")
    if prediction.cell_id not in split_manifest.calibration:
        raise ValueError("calibration prediction cell_id is outside the calibration split")
    if prediction.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 predictions can calibrate normalized conformal intervals")
    if prediction.right_censored or prediction.observed_eol_cycle is None:
        raise ValueError("calibration predictions require an observed non-censored EOL80 cycle")
    if scaled_prediction.scale_version != reference.scale_version:
        raise ValueError("calibration predictions must share scale_version")
    for field in _PREDICTION_CONTEXT_FIELDS:
        if getattr(prediction, field) != getattr(reference.prediction, field):
            raise ValueError(f"calibration predictions must share {field}")


def _require_calibration_versions(
    prediction: LifePrediction, calibration: NormalizedConformalCalibration
) -> None:
    for field in _CALIBRATION_VERSION_FIELDS:
        if getattr(prediction, field) != getattr(calibration, field):
            raise ValueError(f"prediction {field} must match the normalized conformal calibration")


def _validate_interval_context(
    interval: NormalizedPredictionInterval,
    *,
    reference: NormalizedPredictionInterval,
    known_cells: dict[str, NormalizedPredictionInterval],
) -> None:
    if interval.cell_id in known_cells:
        raise ValueError(f"duplicate cell_id in normalized interval cohort: {interval.cell_id}")
    for field in ("dataset_id", "cutoff_cycle", "target", "calibration"):
        if getattr(interval, field) != getattr(reference, field):
            raise ValueError(f"normalized interval cohort must share {field}")


def _validate_observation_context(
    observation: LifePrediction, *, interval: NormalizedPredictionInterval
) -> None:
    if observation.right_censored or observation.observed_eol_cycle is None:
        raise ValueError("observed EOL80 is required for normalized interval coverage")
    for field in ("dataset_id", "cell_id", "cutoff_cycle", "target"):
        if getattr(observation, field) != getattr(interval, field):
            raise ValueError(f"observed prediction {field} must match its interval")
    _require_calibration_versions(observation, interval.calibration)
