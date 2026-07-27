"""Project-bound target-aware Advanced cycle-life prediction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

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
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.advanced_input import (
    AdvancedInputEvidence,
    decode_advanced_input_result,
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

ADVANCED_RUL_PREDICTION_TOOL_VERSION = "advanced-rul-prediction-tool-v1"
ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE = (
    "quanxin_life.advanced_rul_prediction.v1"
)
Clock = Callable[[], datetime]


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class PredictAdvancedRULToolInput(ContractModel):
    upstream_result_id: str
    route_role: AdvancedModelRouteRole

    @field_validator("upstream_result_id")
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_id must be a UUID string") from exc

    @field_validator("route_role")
    @classmethod
    def require_rul_role(
        cls,
        value: AdvancedModelRouteRole,
    ) -> AdvancedModelRouteRole:
        if value not in {
            AdvancedModelRouteRole.DEFAULT,
            AdvancedModelRouteRole.POINT_ACCURACY,
            AdvancedModelRouteRole.COVERAGE,
        }:
            raise ValueError("route_role must be an approved RUL route")
        return value


class AdvancedRULInference(ContractModel):
    """Pure-data result returned by the verified runtime inference boundary."""

    prediction: CycleLifePrediction
    task: Literal[AdvancedModelTask.RUL]
    route_role: AdvancedModelRouteRole
    output_target: Literal["matr_official_cycle_life"]
    artifact_kind: Literal["cyclepatch_direct", "cyclepatch_batlinet"]
    artifact_id: str
    artifact_manifest_sha256: Sha256
    raw_sequence_input_sha256: Sha256
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
    def require_formal_prediction(self) -> AdvancedRULInference:
        if self.prediction.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
            raise ValueError("Advanced RUL prediction target must be MATR official cycle life")
        if not self.prediction.right_censored or self.prediction.observed_cycle is not None:
            raise ValueError("Advanced RUL inference must not expose an observed label")
        return self


class AdvancedRULInferenceService(Protocol):
    def predict(
        self,
        context: VerifiedProjectInvocationContext,
        evidence: AdvancedInputEvidence,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedRULInference: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _validate_requested_role(
    cutoff_cycle: int,
    route_role: AdvancedModelRouteRole,
) -> None:
    if cutoff_cycle == 20:
        if route_role is not AdvancedModelRouteRole.DEFAULT:
            raise ValueError("cutoff 20 requires the frozen DEFAULT RUL route")
        return
    if route_role not in {
        AdvancedModelRouteRole.POINT_ACCURACY,
        AdvancedModelRouteRole.COVERAGE,
    }:
        raise ValueError("cutoff 50/100/150 requires a point or coverage RUL route")


def _validate_inference(
    inference: AdvancedRULInference,
    evidence: AdvancedInputEvidence,
    route_role: AdvancedModelRouteRole,
) -> AdvancedRULInference:
    validated = AdvancedRULInference.model_validate(
        inference.model_dump(mode="json")
    )
    prediction = validated.prediction
    expected = {
        "dataset_id": evidence.dataset_id,
        "cell_id": evidence.cell_id,
        "cutoff_cycle": evidence.cutoff_cycle,
        "data_version": evidence.data_version,
        "feature_version": evidence.feature_version,
        "split_version": evidence.split_version,
    }
    for name, value in expected.items():
        if getattr(prediction, name) != value:
            raise ValueError(f"Advanced RUL prediction {name} is inconsistent")
    if validated.raw_sequence_input_sha256 != evidence.raw_sequence_input_sha256:
        raise ValueError("Advanced RUL raw sequence SHA-256 is inconsistent")
    if validated.route_role is not route_role:
        raise ValueError("Advanced RUL runtime role is inconsistent")
    return validated


def execute_predict_advanced_rul_tool(
    input_value: PredictAdvancedRULToolInput,
    *,
    context: VerifiedProjectInvocationContext,
    result_resolver: RegisteredResultResolver,
    inference_service: AdvancedRULInferenceService,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Issue one formal MATR cycle-life ToolResult from verified model inference."""

    validated_input = PredictAdvancedRULToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    upstream = ToolResult.model_validate(
        result_resolver.resolve_registered_result(
            validated_input.upstream_result_id
        ).model_dump(mode="json")
    )
    evidence = decode_advanced_input_result(upstream)
    _validate_requested_role(evidence.cutoff_cycle, validated_input.route_role)
    inference = _validate_inference(
        inference_service.predict(
            context,
            evidence,
            route_role=validated_input.route_role,
        ),
        evidence,
        validated_input.route_role,
    )
    created_at = _timestamp(clock)
    prediction = inference.prediction
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version=ADVANCED_RUL_PREDICTION_TOOL_VERSION,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
        feature_version=prediction.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": evidence.record_batch_id,
                "dataset_id": evidence.dataset_id,
                "cell_id": evidence.cell_id,
                "cutoff_cycle": evidence.cutoff_cycle,
                "cycle_life_prediction": prediction.model_dump(mode="json"),
                "derived_remaining_cycles": prediction.derived_remaining_cycles,
                "upstream_result_id": validated_input.upstream_result_id,
                "raw_sequence_input_sha256": inference.raw_sequence_input_sha256,
                "transform_config_sha256": evidence.transform_config_sha256,
                "source_manifest_hash": evidence.source_manifest_hash,
                "split_version": evidence.split_version,
                "normalization_statistics_sha256": (
                    inference.normalization_statistics_sha256
                ),
                "task": inference.task.value,
                "route_role": inference.route_role.value,
                "output_target": inference.output_target,
                "artifact_kind": inference.artifact_kind,
                "artifact_id": inference.artifact_id,
                "artifact_manifest_sha256": (
                    inference.artifact_manifest_sha256
                ),
                "decision_event_id": inference.decision_event_id,
                "ledger_sequence_number": inference.ledger_sequence_number,
                "ledger_head_sha256": inference.ledger_head_sha256,
            },
        },
        uncertainty=None,
        warnings=[],
        provenance=[
            *upstream.provenance,
            ProvenanceRecord(
                source_id=f"advanced-model-{inference.artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{inference.artifact_id}",
                sha256=inference.artifact_manifest_sha256,
                description="Verified active Advanced RUL runtime used for inference",
                created_at=created_at,
            ),
        ],
        created_at=created_at,
    )


def register_project_predict_advanced_rul_tool(
    registry: ToolRegistry,
    *,
    inference_service: AdvancedRULInferenceService,
    project_audit_ledger: ProjectResultLedger,
    clock: Clock = _utc_now,
) -> RegisteredTool[PredictAdvancedRULToolInput]:
    """Register formal Advanced RUL inference behind the PROJECT boundary."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version=ADVANCED_RUL_PREDICTION_TOOL_VERSION,
            input_model=PredictAdvancedRULToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_predict_advanced_rul_tool(
                    input_value,
                    context=context,
                    result_resolver=BoundProjectResultResolver(
                        project_audit_ledger,
                        context,
                    ),
                    inference_service=inference_service,
                    clock=clock,
                )
            ),
        )
    )


__all__ = [
    "ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE",
    "ADVANCED_RUL_PREDICTION_TOOL_VERSION",
    "AdvancedRULInference",
    "AdvancedRULInferenceService",
    "PredictAdvancedRULToolInput",
    "execute_predict_advanced_rul_tool",
    "register_project_predict_advanced_rul_tool",
]
