"""PROJECT-only route-specific Split Conformal ToolResult orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.application.advanced_split_conformal import (
    AdvancedConformalRuntimeIdentity,
    AdvancedRULCalibrationSample,
    AdvancedRULPointPrediction,
    AdvancedRULSplitCalibration,
    AdvancedSOHCalibrationSample,
    AdvancedSOHPointPrediction,
    AdvancedSOHSplitCalibration,
    calibrate_advanced_rul_split_conformal,
    calibrate_advanced_soh_split_conformal,
    issue_advanced_rul_interval,
    issue_advanced_soh_band,
)
from quanxin_life.audit.project_ledger import (
    BoundProjectResultResolver,
    ProjectResultLedger,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    CycleLifePrediction,
    PredictionTarget,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )

ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION = "advanced-split-conformal-tool-v1"
ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE = (
    "quanxin_life.advanced_rul_calibration_sample.v1"
)
ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE = (
    "quanxin_life.advanced_soh_calibration_sample.v1"
)
ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE = (
    "quanxin_life.advanced_rul_split_calibration.v1"
)
ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE = (
    "quanxin_life.advanced_rul_split_interval.v1"
)
ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE = (
    "quanxin_life.advanced_soh_split_calibration.v1"
)
ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE = (
    "quanxin_life.advanced_soh_split_band.v1"
)
Operation = Literal["calibrate", "issue"]
Clock = Callable[[], datetime]
_SOH_ROLES = {
    AdvancedModelRouteRole.MEAN_ACCURACY,
    AdvancedModelRouteRole.TAIL_EFFICIENCY,
}


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class AdvancedSplitConformalToolInput(ContractModel):
    operation: Operation
    task: AdvancedModelTask
    route_role: AdvancedModelRouteRole
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    calibration_sample_result_ids: tuple[str, ...] = ()
    prediction_result_id: str | None = None
    calibration_result_id: str | None = None

    @field_validator("prediction_result_id", "calibration_result_id")
    @classmethod
    def require_optional_uuid(cls, value: str | None) -> str | None:
        return None if value is None else _uuid(value, "result IDs")

    @field_validator("calibration_sample_result_ids")
    @classmethod
    def require_sample_uuids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_uuid(value, "calibration sample result IDs") for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("calibration sample result IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def require_operation_contract(self) -> AdvancedSplitConformalToolInput:
        if self.task is AdvancedModelTask.RUL:
            if self.route_role not in {
                AdvancedModelRouteRole.DEFAULT,
                AdvancedModelRouteRole.COVERAGE,
            }:
                raise ValueError(
                    "Advanced RUL Split Conformal requires DEFAULT at cutoff 20 "
                    "or COVERAGE otherwise"
                )
        elif self.route_role not in _SOH_ROLES:
            raise ValueError("Advanced SOH Split Conformal requires an approved SOH role")
        if self.operation == "calibrate":
            if (
                self.alpha is None
                or not self.calibration_sample_result_ids
                or self.prediction_result_id is not None
                or self.calibration_result_id is not None
            ):
                raise ValueError(
                    "calibrate requires alpha and calibration sample result IDs only"
                )
        elif (
            self.alpha is not None
            or self.calibration_sample_result_ids
            or self.prediction_result_id is None
            or self.calibration_result_id is None
        ):
            raise ValueError(
                "issue requires prediction and calibration result IDs only"
            )
        return self


def execute_advanced_split_conformal_tool(
    input_value: AdvancedSplitConformalToolInput,
    *,
    context: VerifiedProjectInvocationContext,
    result_resolver: RegisteredResultResolver,
    clock: Clock = lambda: datetime.now(UTC),
) -> ToolResult:
    """Calibrate or issue one route-bound Advanced Split Conformal result."""

    del context
    validated = AdvancedSplitConformalToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    if validated.operation == "calibrate":
        return _calibrate(validated, result_resolver=result_resolver, clock=clock)
    return _issue(validated, result_resolver=result_resolver, clock=clock)


def register_project_advanced_split_conformal_tool(
    registry: ToolRegistry,
    *,
    project_audit_ledger: ProjectResultLedger,
    clock: Clock = lambda: datetime.now(UTC),
) -> RegisteredTool[AdvancedSplitConformalToolInput]:
    """Register Advanced Split Conformal behind the PROJECT result ledger."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            tool_version=ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
            input_model=AdvancedSplitConformalToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_advanced_split_conformal_tool(
                    input_value,
                    context=context,
                    result_resolver=BoundProjectResultResolver(
                        project_audit_ledger,
                        context,
                    ),
                    clock=clock,
                )
            ),
        )
    )


def _calibrate(
    input_value: AdvancedSplitConformalToolInput,
    *,
    result_resolver: RegisteredResultResolver,
    clock: Clock,
) -> ToolResult:
    assert input_value.alpha is not None
    sample_results = tuple(
        _resolve(result_resolver, result_id)
        for result_id in input_value.calibration_sample_result_ids
    )
    if input_value.task is AdvancedModelTask.RUL:
        rul_calibration = calibrate_advanced_rul_split_conformal(
            tuple(_decode_rul_sample(result) for result in sample_results),
            alpha=input_value.alpha,
        )
        _require_requested_identity(input_value, rul_calibration.runtime)
        artifact_type = ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE
        artifact = {
            **_runtime_payload(rul_calibration.runtime),
            "alpha": rul_calibration.alpha,
            "coverage_target": rul_calibration.coverage_target,
            "residual_quantile_cycle": rul_calibration.residual_quantile_cycle,
            "calibration_cell_count": len(rul_calibration.calibration_cell_ids),
            "calibration_cell_ids": list(rul_calibration.calibration_cell_ids),
            "calibration_sample_result_ids": list(
                input_value.calibration_sample_result_ids
            ),
            "calibration_method": rul_calibration.calibration_method,
        }
        runtime = rul_calibration.runtime
    else:
        soh_calibration = calibrate_advanced_soh_split_conformal(
            tuple(_decode_soh_sample(result) for result in sample_results),
            alpha=input_value.alpha,
        )
        _require_requested_identity(input_value, soh_calibration.runtime)
        artifact_type = ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE
        artifact = {
            **_runtime_payload(soh_calibration.runtime),
            "alpha": soh_calibration.alpha,
            "coverage_target": soh_calibration.coverage_target,
            "prediction_cycles": list(soh_calibration.prediction_cycles),
            "residual_quantile_soh": soh_calibration.residual_quantile_soh,
            "calibration_cell_count": len(soh_calibration.calibration_cell_ids),
            "calibration_cell_ids": list(soh_calibration.calibration_cell_ids),
            "calibration_sample_result_ids": list(
                input_value.calibration_sample_result_ids
            ),
            "finite_horizon_only": soh_calibration.finite_horizon_only,
            "coverage_scope": soh_calibration.coverage_scope,
            "calibration_method": soh_calibration.calibration_method,
        }
        runtime = soh_calibration.runtime
    return _tool_result(
        input_value=input_value,
        artifact_type=artifact_type,
        artifact=artifact,
        runtime=runtime,
        uncertainty=None,
        provenance=_merge_provenance(*(result.provenance for result in sample_results)),
        clock=clock,
    )


def _issue(
    input_value: AdvancedSplitConformalToolInput,
    *,
    result_resolver: RegisteredResultResolver,
    clock: Clock,
) -> ToolResult:
    assert input_value.prediction_result_id is not None
    assert input_value.calibration_result_id is not None
    prediction_result = _resolve(
        result_resolver,
        input_value.prediction_result_id,
    )
    if input_value.task is AdvancedModelTask.RUL:
        rul_point = _decode_rul_prediction(prediction_result)
        _require_requested_identity(input_value, rul_point.runtime)
        calibration_result = _resolve(
            result_resolver,
            input_value.calibration_result_id,
        )
        rul_calibration = _decode_rul_calibration(calibration_result)
        interval = issue_advanced_rul_interval(rul_point, rul_calibration)
        artifact_type = ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE
        artifact = {
            **_runtime_payload(rul_point.runtime),
            "cell_id": rul_point.cell_id,
            "point_prediction_cycle": interval.point_prediction_cycle,
            "lower_cycle": interval.lower_cycle,
            "upper_cycle": interval.upper_cycle,
            "derived_rul_cycle": interval.derived_rul_cycle,
            "lower_rul_cycle": interval.lower_rul_cycle,
            "upper_rul_cycle": interval.upper_rul_cycle,
            "alpha": rul_calibration.alpha,
            "coverage_target": rul_calibration.coverage_target,
            "calibration_method": rul_calibration.calibration_method,
            "prediction_result_id": input_value.prediction_result_id,
            "calibration_result_id": input_value.calibration_result_id,
        }
        uncertainty: dict[str, object] = {
            "coverage_target": rul_calibration.coverage_target,
            "point_prediction_cycle": interval.point_prediction_cycle,
            "lower_cycle": interval.lower_cycle,
            "upper_cycle": interval.upper_cycle,
            "derived_rul_cycle": interval.derived_rul_cycle,
            "lower_rul_cycle": interval.lower_rul_cycle,
            "upper_rul_cycle": interval.upper_rul_cycle,
        }
        return _tool_result(
            input_value=input_value,
            artifact_type=artifact_type,
            artifact=artifact,
            runtime=rul_point.runtime,
            uncertainty=uncertainty,
            provenance=_merge_provenance(
                prediction_result.provenance,
                calibration_result.provenance,
            ),
            clock=clock,
        )

    soh_point = _decode_soh_prediction(prediction_result)
    _require_requested_identity(input_value, soh_point.runtime)
    calibration_result = _resolve(
        result_resolver,
        input_value.calibration_result_id,
    )
    soh_calibration = _decode_soh_calibration(calibration_result)
    band = issue_advanced_soh_band(soh_point, soh_calibration)
    artifact = {
        **_runtime_payload(soh_point.runtime),
        "cell_id": soh_point.cell_id,
        "prediction_cycles": list(band.prediction_cycles),
        "predicted_soh": list(band.predicted_soh),
        "lower_soh": list(band.lower_soh),
        "upper_soh": list(band.upper_soh),
        "alpha": soh_calibration.alpha,
        "coverage_target": soh_calibration.coverage_target,
        "finite_horizon_only": band.finite_horizon_only,
        "coverage_scope": band.coverage_scope,
        "calibration_method": soh_calibration.calibration_method,
        "prediction_result_id": input_value.prediction_result_id,
        "calibration_result_id": input_value.calibration_result_id,
    }
    soh_uncertainty: dict[str, object] = {
        "coverage_target": soh_calibration.coverage_target,
        "finite_horizon_only": True,
        "coverage_scope": "simultaneous_finite_trajectory",
    }
    return _tool_result(
        input_value=input_value,
        artifact_type=ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
        artifact=artifact,
        runtime=soh_point.runtime,
        uncertainty=soh_uncertainty,
        provenance=_merge_provenance(
            prediction_result.provenance,
            calibration_result.provenance,
        ),
        clock=clock,
    )


def _decode_rul_sample(result: ToolResult) -> AdvancedRULCalibrationSample:
    artifact = _artifact(result, ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.PREDICT_CYCLE_LIFE)
    _require_calibration_provenance(result)
    if artifact.get("split_partition") != "calibration":
        raise ValueError("RUL calibration sample must belong to the calibration split")
    runtime = _runtime_from_result(result, artifact)
    return AdvancedRULCalibrationSample.model_validate(
        {
            "cell_id": artifact["cell_id"],
            "point_prediction_cycle": artifact["point_prediction_cycle"],
            "observed_cycle": artifact["observed_cycle"],
            "runtime": runtime.model_dump(mode="json"),
        }
    )


def _decode_soh_sample(result: ToolResult) -> AdvancedSOHCalibrationSample:
    artifact = _artifact(result, ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.PREDICT_SOH_TRAJECTORY)
    _require_calibration_provenance(result)
    if artifact.get("split_partition") != "calibration":
        raise ValueError("SOH calibration sample must belong to the calibration split")
    runtime = _runtime_from_result(result, artifact)
    return AdvancedSOHCalibrationSample.model_validate(
        {
            "cell_id": artifact["cell_id"],
            "prediction_cycles": artifact["prediction_cycles"],
            "predicted_soh": artifact["predicted_soh"],
            "observed_soh": artifact["observed_soh"],
            "runtime": runtime.model_dump(mode="json"),
        }
    )


def _decode_rul_prediction(result: ToolResult) -> AdvancedRULPointPrediction:
    artifact = _artifact(result, ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.PREDICT_CYCLE_LIFE)
    prediction = CycleLifePrediction.model_validate(
        artifact["cycle_life_prediction"]
    )
    if (
        prediction.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        or not prediction.right_censored
        or prediction.observed_cycle is not None
    ):
        raise ValueError("Advanced RUL point prediction is not label-free MATR cycle life")
    _require_result_versions(
        result,
        data_version=prediction.data_version,
        feature_version=prediction.feature_version,
        model_version=prediction.model_version,
    )
    derived_remaining_cycles = artifact.get("derived_remaining_cycles")
    if not isinstance(derived_remaining_cycles, int | float) or isinstance(
        derived_remaining_cycles,
        bool,
    ):
        raise ValueError("Advanced RUL derived remaining cycles are invalid")
    if (
        artifact.get("dataset_id") != prediction.dataset_id
        or artifact.get("cell_id") != prediction.cell_id
        or artifact.get("cutoff_cycle") != prediction.cutoff_cycle
        or artifact.get("split_version") != prediction.split_version
        or not math.isclose(
            float(derived_remaining_cycles),
            prediction.derived_remaining_cycles,
        )
    ):
        raise ValueError("Advanced RUL ToolResult payload is inconsistent")
    return AdvancedRULPointPrediction(
        cell_id=prediction.cell_id,
        point_prediction_cycle=prediction.predicted_cycle,
        runtime=_runtime_from_result(result, artifact),
    )


def _decode_soh_prediction(result: ToolResult) -> AdvancedSOHPointPrediction:
    artifact = _artifact(result, ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.PREDICT_SOH_TRAJECTORY)
    runtime = _runtime_from_result(result, artifact)
    _require_result_versions(
        result,
        data_version=runtime.data_version,
        feature_version=runtime.feature_version,
        model_version=runtime.model_version,
    )
    return AdvancedSOHPointPrediction.model_validate(
        {
            "cell_id": artifact["cell_id"],
            "prediction_cycles": artifact["prediction_cycles"],
            "predicted_soh": artifact["predicted_soh"],
            "runtime": runtime.model_dump(mode="json"),
        }
    )


def _decode_rul_calibration(result: ToolResult) -> AdvancedRULSplitCalibration:
    artifact = _artifact(result, ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.CALIBRATE_PREDICTION_INTERVAL)
    runtime = _runtime_from_result(result, artifact)
    return AdvancedRULSplitCalibration.model_validate(
        {
            "runtime": runtime.model_dump(mode="json"),
            "alpha": artifact["alpha"],
            "coverage_target": artifact["coverage_target"],
            "residual_quantile_cycle": artifact["residual_quantile_cycle"],
            "calibration_cell_ids": artifact["calibration_cell_ids"],
            "calibration_method": artifact["calibration_method"],
        }
    )


def _decode_soh_calibration(result: ToolResult) -> AdvancedSOHSplitCalibration:
    artifact = _artifact(result, ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE)
    _require_tool_name(result, StandardToolName.CALIBRATE_PREDICTION_INTERVAL)
    runtime = _runtime_from_result(result, artifact)
    return AdvancedSOHSplitCalibration.model_validate(
        {
            "runtime": runtime.model_dump(mode="json"),
            "alpha": artifact["alpha"],
            "coverage_target": artifact["coverage_target"],
            "prediction_cycles": artifact["prediction_cycles"],
            "residual_quantile_soh": artifact["residual_quantile_soh"],
            "calibration_cell_ids": artifact["calibration_cell_ids"],
            "finite_horizon_only": artifact["finite_horizon_only"],
            "coverage_scope": artifact["coverage_scope"],
            "calibration_method": artifact["calibration_method"],
        }
    )


def _runtime_from_result(
    result: ToolResult,
    artifact: dict[str, object],
) -> AdvancedConformalRuntimeIdentity:
    runtime = AdvancedConformalRuntimeIdentity.model_validate(
        {
            "task": artifact["task"],
            "route_role": artifact["route_role"],
            "output_target": artifact["output_target"],
            "artifact_kind": artifact["artifact_kind"],
            "artifact_id": artifact["artifact_id"],
            "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
            "model_version": result.model_version,
            "dataset_id": artifact["dataset_id"],
            "cutoff_cycle": artifact["cutoff_cycle"],
            "data_version": result.data_version,
            "feature_version": result.feature_version,
            "split_version": artifact["split_version"],
            "normalization_statistics_sha256": artifact[
                "normalization_statistics_sha256"
            ],
            "decision_event_id": artifact["decision_event_id"],
            "ledger_sequence_number": artifact["ledger_sequence_number"],
            "ledger_head_sha256": artifact["ledger_head_sha256"],
        }
    )
    _require_result_versions(
        result,
        data_version=runtime.data_version,
        feature_version=runtime.feature_version,
        model_version=runtime.model_version,
    )
    return runtime


def _runtime_payload(
    runtime: AdvancedConformalRuntimeIdentity,
) -> dict[str, object]:
    return runtime.model_dump(mode="json")


def _tool_result(
    *,
    input_value: AdvancedSplitConformalToolInput,
    artifact_type: str,
    artifact: dict[str, object],
    runtime: AdvancedConformalRuntimeIdentity,
    uncertainty: dict[str, object] | None,
    provenance: Sequence[ProvenanceRecord],
    clock: Clock,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        tool_version=ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
        model_version=runtime.model_version,
        data_version=runtime.data_version,
        feature_version=runtime.feature_version,
        input_hash=sha256_canonical(input_value.model_dump(mode="json")),
        values={"artifact_type": artifact_type, "artifact": artifact},
        uncertainty=uncertainty,
        warnings=[],
        provenance=list(provenance),
        created_at=_timestamp(clock),
    )


def _artifact(result: ToolResult, expected_type: str) -> dict[str, object]:
    values = result.values
    if set(values) != {"artifact_type", "artifact"}:
        raise ValueError("Advanced Conformal dependency has an invalid envelope")
    if values["artifact_type"] != expected_type:
        raise ValueError("Advanced Conformal dependency artifact type is invalid")
    artifact = values["artifact"]
    if not isinstance(artifact, dict):
        raise ValueError("Advanced Conformal dependency artifact is invalid")
    return artifact


def _resolve(
    resolver: RegisteredResultResolver,
    result_id: str,
) -> ToolResult:
    resolved = resolver.resolve_registered_result(result_id)
    result = ToolResult.model_validate(resolved.model_dump(mode="json"))
    if result.result_id != result_id:
        raise ValueError("project result resolver returned a mismatched result")
    return result


def _require_requested_identity(
    request: AdvancedSplitConformalToolInput,
    runtime: AdvancedConformalRuntimeIdentity,
) -> None:
    if runtime.task is not request.task or runtime.route_role is not request.route_role:
        raise ValueError("requested Conformal route does not match prediction runtime")


def _require_result_versions(
    result: ToolResult,
    *,
    data_version: str,
    feature_version: str,
    model_version: str,
) -> None:
    if (
        result.data_version != data_version
        or result.feature_version != feature_version
        or result.model_version != model_version
    ):
        raise ValueError("ToolResult versions do not match its Advanced artifact")


def _require_tool_name(
    result: ToolResult,
    expected: StandardToolName,
) -> None:
    if result.tool_name != expected.value:
        raise ValueError("Advanced Conformal dependency tool name is invalid")


def _require_calibration_provenance(result: ToolResult) -> None:
    kinds = {record.source_kind for record in result.provenance}
    if SourceKind.OBSERVED not in kinds or SourceKind.PREDICTED not in kinds:
        raise ValueError(
            "calibration samples require observed and predicted provenance"
        )


def _merge_provenance(
    *chains: Sequence[ProvenanceRecord],
) -> tuple[ProvenanceRecord, ...]:
    merged: list[ProvenanceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in chains:
        for record in chain:
            key = (record.source_id, record.uri, record.sha256)
            if key not in seen:
                seen.add(key)
                merged.append(record)
    return tuple(merged)


def _timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be UUID strings") from exc


__all__ = [
    "ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE",
    "ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE",
    "ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE",
    "ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE",
    "ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE",
    "ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE",
    "ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION",
    "AdvancedSplitConformalToolInput",
    "execute_advanced_split_conformal_tool",
    "register_project_advanced_split_conformal_tool",
]
