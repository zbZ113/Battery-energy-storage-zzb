"""Pure route-bound Split Conformal services for Advanced predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask
from quanxin_life.core.schemas import ContractModel, Sha256

FiniteValue = Annotated[float, Field(allow_inf_nan=False)]
SOHValue = Annotated[float, Field(ge=0.0, le=1.5, allow_inf_nan=False)]
ArtifactKind = Literal[
    "cyclepatch_direct",
    "cyclepatch_batlinet",
    "hybridpatch_v2",
    "current_hybrid",
]
OutputTarget = Literal["matr_official_cycle_life", "soh_trajectory"]
SOHCoverageScope = Literal["simultaneous_finite_trajectory"]
_SOH_ROLES = {
    AdvancedModelRouteRole.MEAN_ACCURACY,
    AdvancedModelRouteRole.TAIL_EFFICIENCY,
}


class AdvancedConformalRuntimeIdentity(ContractModel):
    """Exact active-route identity shared by calibration and target prediction."""

    task: AdvancedModelTask
    route_role: AdvancedModelRouteRole
    output_target: OutputTarget
    artifact_kind: ArtifactKind
    artifact_id: str
    artifact_manifest_sha256: Sha256
    model_version: str = Field(min_length=1)
    dataset_id: Literal["MATR"]
    cutoff_cycle: int = Field(ge=0)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    normalization_statistics_sha256: Sha256
    decision_event_id: str
    ledger_sequence_number: int = Field(ge=1)
    ledger_head_sha256: Sha256

    @field_validator("artifact_id", "decision_event_id")
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("runtime identifiers must be UUID strings") from exc

    @model_validator(mode="after")
    def require_task_specific_identity(self) -> AdvancedConformalRuntimeIdentity:
        if self.task is AdvancedModelTask.RUL:
            expected_role = (
                AdvancedModelRouteRole.DEFAULT
                if self.cutoff_cycle == 20
                else AdvancedModelRouteRole.COVERAGE
            )
            if self.route_role is not expected_role:
                raise ValueError(
                    "Advanced RUL Split Conformal requires DEFAULT at cutoff 20 "
                    "or COVERAGE otherwise"
                )
            if self.output_target != "matr_official_cycle_life":
                raise ValueError("Advanced RUL output target must be MATR official cycle life")
            if self.artifact_kind not in {
                "cyclepatch_direct",
                "cyclepatch_batlinet",
            }:
                raise ValueError("Advanced RUL artifact kind is inconsistent")
        else:
            if self.route_role not in _SOH_ROLES:
                raise ValueError("Advanced SOH Split Conformal requires an approved SOH role")
            if self.output_target != "soh_trajectory":
                raise ValueError("Advanced SOH output target must be a finite SOH trajectory")
            if self.artifact_kind not in {"hybridpatch_v2", "current_hybrid"}:
                raise ValueError("Advanced SOH artifact kind is inconsistent")
        return self


class AdvancedRULCalibrationSample(ContractModel):
    cell_id: str = Field(min_length=1)
    point_prediction_cycle: FiniteValue = Field(ge=0)
    observed_cycle: int = Field(ge=0)
    runtime: AdvancedConformalRuntimeIdentity

    @model_validator(mode="after")
    def require_rul_sample(self) -> AdvancedRULCalibrationSample:
        if self.runtime.task is not AdvancedModelTask.RUL:
            raise ValueError("RUL calibration sample requires a RUL runtime")
        if (
            self.point_prediction_cycle < self.runtime.cutoff_cycle
            or self.observed_cycle < self.runtime.cutoff_cycle
        ):
            raise ValueError("MATR official cycle life cannot precede cutoff_cycle")
        return self


class AdvancedRULPointPrediction(ContractModel):
    cell_id: str = Field(min_length=1)
    point_prediction_cycle: FiniteValue = Field(ge=0)
    runtime: AdvancedConformalRuntimeIdentity

    @model_validator(mode="after")
    def require_rul_point(self) -> AdvancedRULPointPrediction:
        if self.runtime.task is not AdvancedModelTask.RUL:
            raise ValueError("RUL point prediction requires a RUL runtime")
        if self.point_prediction_cycle < self.runtime.cutoff_cycle:
            raise ValueError("MATR official cycle life cannot precede cutoff_cycle")
        return self


class AdvancedRULSplitCalibration(ContractModel):
    runtime: AdvancedConformalRuntimeIdentity
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    coverage_target: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    residual_quantile_cycle: FiniteValue = Field(ge=0)
    calibration_cell_ids: tuple[str, ...] = Field(min_length=1)
    calibration_method: Literal["split_conformal_absolute_residual_v1"] = (
        "split_conformal_absolute_residual_v1"
    )


class AdvancedRULInterval(ContractModel):
    cell_id: str = Field(min_length=1)
    point_prediction_cycle: FiniteValue = Field(ge=0)
    lower_cycle: FiniteValue = Field(ge=0)
    upper_cycle: FiniteValue = Field(ge=0)
    derived_rul_cycle: FiniteValue = Field(ge=0)
    lower_rul_cycle: FiniteValue = Field(ge=0)
    upper_rul_cycle: FiniteValue = Field(ge=0)
    calibration: AdvancedRULSplitCalibration

    @model_validator(mode="after")
    def require_ordered_interval(self) -> AdvancedRULInterval:
        if not (
            self.lower_cycle
            <= self.point_prediction_cycle
            <= self.upper_cycle
        ):
            raise ValueError("RUL point prediction must be contained in its interval")
        if not (
            self.lower_rul_cycle
            <= self.derived_rul_cycle
            <= self.upper_rul_cycle
        ):
            raise ValueError("derived RUL must be contained in its interval")
        return self


class AdvancedSOHCalibrationSample(ContractModel):
    cell_id: str = Field(min_length=1)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    predicted_soh: tuple[SOHValue, ...] = Field(min_length=1)
    observed_soh: tuple[SOHValue, ...] = Field(min_length=1)
    runtime: AdvancedConformalRuntimeIdentity

    @model_validator(mode="after")
    def require_finite_sample(self) -> AdvancedSOHCalibrationSample:
        _validate_soh_axes(
            self.prediction_cycles,
            self.predicted_soh,
            self.runtime,
        )
        if len(self.observed_soh) != len(self.prediction_cycles):
            raise ValueError("observed SOH must align with prediction cycles")
        return self


class AdvancedSOHPointPrediction(ContractModel):
    cell_id: str = Field(min_length=1)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    predicted_soh: tuple[SOHValue, ...] = Field(min_length=1)
    runtime: AdvancedConformalRuntimeIdentity

    @model_validator(mode="after")
    def require_finite_prediction(self) -> AdvancedSOHPointPrediction:
        _validate_soh_axes(
            self.prediction_cycles,
            self.predicted_soh,
            self.runtime,
        )
        return self


class AdvancedSOHSplitCalibration(ContractModel):
    runtime: AdvancedConformalRuntimeIdentity
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    coverage_target: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    residual_quantile_soh: FiniteValue = Field(ge=0)
    calibration_cell_ids: tuple[str, ...] = Field(min_length=1)
    finite_horizon_only: Literal[True] = True
    coverage_scope: SOHCoverageScope = "simultaneous_finite_trajectory"
    calibration_method: Literal["split_conformal_cell_max_residual_v1"] = (
        "split_conformal_cell_max_residual_v1"
    )


class AdvancedSOHBand(ContractModel):
    cell_id: str = Field(min_length=1)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    predicted_soh: tuple[SOHValue, ...] = Field(min_length=1)
    lower_soh: tuple[SOHValue, ...] = Field(min_length=1)
    upper_soh: tuple[SOHValue, ...] = Field(min_length=1)
    finite_horizon_only: Literal[True] = True
    coverage_scope: SOHCoverageScope = "simultaneous_finite_trajectory"
    calibration: AdvancedSOHSplitCalibration

    @model_validator(mode="after")
    def require_ordered_band(self) -> AdvancedSOHBand:
        lengths = {
            len(self.prediction_cycles),
            len(self.predicted_soh),
            len(self.lower_soh),
            len(self.upper_soh),
        }
        if len(lengths) != 1:
            raise ValueError("SOH band arrays must align")
        if any(
            not lower <= point <= upper
            for lower, point, upper in zip(
                self.lower_soh,
                self.predicted_soh,
                self.upper_soh,
                strict=True,
            )
        ):
            raise ValueError("SOH point trajectory must be contained in its band")
        return self


def calibrate_advanced_rul_split_conformal(
    samples: Sequence[AdvancedRULCalibrationSample],
    *,
    alpha: float,
) -> AdvancedRULSplitCalibration:
    cohort = tuple(
        AdvancedRULCalibrationSample.model_validate(
            sample.model_dump(mode="json")
        )
        for sample in samples
    )
    if not cohort:
        raise ValueError("at least one RUL calibration cell is required")
    runtime = cohort[0].runtime
    cell_ids = _unique_cells(cohort)
    residuals: list[float] = []
    for sample in cohort:
        _require_same_runtime(sample.runtime, runtime)
        residuals.append(
            abs(sample.point_prediction_cycle - float(sample.observed_cycle))
        )
    return AdvancedRULSplitCalibration(
        runtime=runtime,
        alpha=alpha,
        coverage_target=1.0 - alpha,
        residual_quantile_cycle=_finite_sample_quantile(residuals, alpha),
        calibration_cell_ids=cell_ids,
    )


def issue_advanced_rul_interval(
    prediction: AdvancedRULPointPrediction,
    calibration: AdvancedRULSplitCalibration,
) -> AdvancedRULInterval:
    point = AdvancedRULPointPrediction.model_validate(
        prediction.model_dump(mode="json")
    )
    calibrated = AdvancedRULSplitCalibration.model_validate(
        calibration.model_dump(mode="json")
    )
    _require_same_runtime(point.runtime, calibrated.runtime)
    _require_target_not_calibration(point.cell_id, calibrated.calibration_cell_ids)
    cutoff = float(point.runtime.cutoff_cycle)
    lower = max(
        cutoff,
        point.point_prediction_cycle - calibrated.residual_quantile_cycle,
    )
    upper = point.point_prediction_cycle + calibrated.residual_quantile_cycle
    return AdvancedRULInterval(
        cell_id=point.cell_id,
        point_prediction_cycle=point.point_prediction_cycle,
        lower_cycle=lower,
        upper_cycle=upper,
        derived_rul_cycle=point.point_prediction_cycle - cutoff,
        lower_rul_cycle=lower - cutoff,
        upper_rul_cycle=upper - cutoff,
        calibration=calibrated,
    )


def calibrate_advanced_soh_split_conformal(
    samples: Sequence[AdvancedSOHCalibrationSample],
    *,
    alpha: float,
) -> AdvancedSOHSplitCalibration:
    cohort = tuple(
        AdvancedSOHCalibrationSample.model_validate(
            sample.model_dump(mode="json")
        )
        for sample in samples
    )
    if not cohort:
        raise ValueError("at least one SOH calibration cell is required")
    reference = cohort[0]
    cell_ids = _unique_cells(cohort)
    scores: list[float] = []
    for sample in cohort:
        _require_same_runtime(sample.runtime, reference.runtime)
        if sample.prediction_cycles != reference.prediction_cycles:
            raise ValueError("SOH calibration prediction cycles must match")
        scores.append(
            max(
                abs(predicted - observed)
                for predicted, observed in zip(
                    sample.predicted_soh,
                    sample.observed_soh,
                    strict=True,
                )
            )
        )
    return AdvancedSOHSplitCalibration(
        runtime=reference.runtime,
        alpha=alpha,
        coverage_target=1.0 - alpha,
        prediction_cycles=reference.prediction_cycles,
        residual_quantile_soh=_finite_sample_quantile(scores, alpha),
        calibration_cell_ids=cell_ids,
    )


def issue_advanced_soh_band(
    prediction: AdvancedSOHPointPrediction,
    calibration: AdvancedSOHSplitCalibration,
) -> AdvancedSOHBand:
    point = AdvancedSOHPointPrediction.model_validate(
        prediction.model_dump(mode="json")
    )
    calibrated = AdvancedSOHSplitCalibration.model_validate(
        calibration.model_dump(mode="json")
    )
    _require_same_runtime(point.runtime, calibrated.runtime)
    _require_target_not_calibration(point.cell_id, calibrated.calibration_cell_ids)
    if point.prediction_cycles != calibrated.prediction_cycles:
        raise ValueError("SOH target prediction cycles must match calibration")
    radius = calibrated.residual_quantile_soh
    return AdvancedSOHBand(
        cell_id=point.cell_id,
        prediction_cycles=point.prediction_cycles,
        predicted_soh=point.predicted_soh,
        lower_soh=tuple(max(0.0, value - radius) for value in point.predicted_soh),
        upper_soh=tuple(min(1.5, value + radius) for value in point.predicted_soh),
        calibration=calibrated,
    )


def _finite_sample_quantile(values: Sequence[float], alpha: float) -> float:
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("conformal residuals must be finite and non-negative")
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    if rank > len(ordered):
        raise ValueError(
            "not enough calibration cells for requested coverage target"
        )
    return ordered[rank - 1]


def _unique_cells(
    cohort: Sequence[AdvancedRULCalibrationSample | AdvancedSOHCalibrationSample],
) -> tuple[str, ...]:
    cell_ids = tuple(sample.cell_id for sample in cohort)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("calibration cell_ids must be unique")
    return cell_ids


def _require_same_runtime(
    observed: AdvancedConformalRuntimeIdentity,
    expected: AdvancedConformalRuntimeIdentity,
) -> None:
    if observed != expected:
        raise ValueError("calibration and prediction must share the exact runtime identity")


def _require_target_not_calibration(
    cell_id: str,
    calibration_cell_ids: Sequence[str],
) -> None:
    if cell_id in calibration_cell_ids:
        raise ValueError("target prediction cell_id must not be a calibration cell")


def _validate_soh_axes(
    prediction_cycles: tuple[int, ...],
    predicted_soh: tuple[float, ...],
    runtime: AdvancedConformalRuntimeIdentity,
) -> None:
    if runtime.task is not AdvancedModelTask.SOH:
        raise ValueError("SOH prediction requires a SOH runtime")
    if len(prediction_cycles) != len(predicted_soh):
        raise ValueError("predicted SOH must align with prediction cycles")
    if any(
        current >= following
        for current, following in pairwise(prediction_cycles)
    ):
        raise ValueError("prediction cycles must be strictly increasing")
    if (
        prediction_cycles[0] <= runtime.cutoff_cycle
        or prediction_cycles[-1] > 500
    ):
        raise ValueError("SOH prediction cycles must be after cutoff and end by cycle 500")


__all__ = [
    "AdvancedConformalRuntimeIdentity",
    "AdvancedRULCalibrationSample",
    "AdvancedRULInterval",
    "AdvancedRULPointPrediction",
    "AdvancedRULSplitCalibration",
    "AdvancedSOHBand",
    "AdvancedSOHCalibrationSample",
    "AdvancedSOHPointPrediction",
    "AdvancedSOHSplitCalibration",
    "calibrate_advanced_rul_split_conformal",
    "calibrate_advanced_soh_split_conformal",
    "issue_advanced_rul_interval",
    "issue_advanced_soh_band",
]
