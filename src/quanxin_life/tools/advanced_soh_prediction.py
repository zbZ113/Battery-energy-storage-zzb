"""Project-bound finite-horizon Advanced SOH trajectory prediction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.audit.project_ledger import (
    BoundProjectResultResolver,
    ProjectResultLedger,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.advanced_input import (
    AdvancedInputEvidence,
    cell_metadata_evidence_payload,
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

ADVANCED_SOH_PREDICTION_TOOL_VERSION = "advanced-soh-prediction-tool-v1"
ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1 = (
    "quanxin_life.advanced_soh_trajectory.v1"
)
ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE = (
    "quanxin_life.advanced_soh_trajectory.v2"
)
ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES = frozenset(
    {
        ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
        ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    }
)
SOHValue = Annotated[float, Field(ge=0.0, le=1.5, allow_inf_nan=False)]
Clock = Callable[[], datetime]
_SOH_ROLES = {
    AdvancedModelRouteRole.MEAN_ACCURACY,
    AdvancedModelRouteRole.TAIL_EFFICIENCY,
}


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class PredictAdvancedSOHToolInput(ContractModel):
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
    def require_soh_role(
        cls, value: AdvancedModelRouteRole
    ) -> AdvancedModelRouteRole:
        if value not in _SOH_ROLES:
            raise ValueError("route_role must be an approved SOH role")
        return value


class AdvancedSOHInference(ContractModel):
    """Pure-data finite trajectory returned by a verified SOH runtime."""

    dataset_id: Literal["MATR"]
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    task: Literal[AdvancedModelTask.SOH]
    route_role: AdvancedModelRouteRole
    output_target: Literal["soh_trajectory"]
    artifact_kind: Literal["hybridpatch_v2", "current_hybrid"]
    artifact_id: str
    artifact_manifest_sha256: Sha256
    raw_sequence_input_sha256: Sha256
    normalization_statistics_sha256: Sha256
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    predicted_soh: tuple[SOHValue, ...] = Field(min_length=1)
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

    @field_validator("route_role")
    @classmethod
    def require_soh_role(
        cls, value: AdvancedModelRouteRole
    ) -> AdvancedModelRouteRole:
        if value not in _SOH_ROLES:
            raise ValueError("runtime role must be an approved SOH role")
        return value

    @model_validator(mode="after")
    def validate_finite_trajectory(self) -> AdvancedSOHInference:
        if len(self.prediction_cycles) != len(self.predicted_soh):
            raise ValueError("prediction cycles and SOH values must align")
        if any(
            current >= following
            for current, following in zip(
                self.prediction_cycles,
                self.prediction_cycles[1:],
                strict=False,
            )
        ):
            raise ValueError("prediction cycles must be strictly increasing")
        if (
            self.prediction_cycles[0] <= self.cutoff_cycle
            or self.prediction_cycles[-1] > 500
        ):
            raise ValueError("SOH trajectory must be after cutoff and end by cycle 500")
        if any(
            following > current
            for current, following in zip(
                self.predicted_soh,
                self.predicted_soh[1:],
                strict=False,
            )
        ):
            raise ValueError("predicted SOH must be non-increasing")
        return self


class AdvancedSOHInferenceService(Protocol):
    def predict(
        self,
        context: VerifiedProjectInvocationContext,
        evidence: AdvancedInputEvidence,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedSOHInference: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _validate_inference(
    inference: AdvancedSOHInference,
    evidence: AdvancedInputEvidence,
    route_role: AdvancedModelRouteRole,
) -> AdvancedSOHInference:
    validated = AdvancedSOHInference.model_validate(
        inference.model_dump(mode="json")
    )
    expected = {
        "dataset_id": evidence.dataset_id,
        "cell_id": evidence.cell_id,
        "cutoff_cycle": evidence.cutoff_cycle,
        "data_version": evidence.data_version,
        "feature_version": evidence.feature_version,
        "split_version": evidence.split_version,
    }
    for name, value in expected.items():
        if getattr(validated, name) != value:
            raise ValueError(f"Advanced SOH prediction {name} is inconsistent")
    if validated.raw_sequence_input_sha256 != evidence.raw_sequence_input_sha256:
        raise ValueError("Advanced SOH raw sequence SHA-256 is inconsistent")
    if validated.route_role is not route_role:
        raise ValueError("Advanced SOH runtime role is inconsistent")
    return validated


def execute_predict_advanced_soh_tool(
    input_value: PredictAdvancedSOHToolInput,
    *,
    context: VerifiedProjectInvocationContext,
    result_resolver: RegisteredResultResolver,
    inference_service: AdvancedSOHInferenceService,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Issue one finite SOH trajectory ToolResult from verified inference."""

    validated_input = PredictAdvancedSOHToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    upstream = ToolResult.model_validate(
        result_resolver.resolve_registered_result(
            validated_input.upstream_result_id
        ).model_dump(mode="json")
    )
    evidence = decode_advanced_input_result(upstream)
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
    cell_metadata = cell_metadata_evidence_payload(evidence.cell_metadata)
    evidence_type = (
        ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE
        if cell_metadata is not None
        else ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
        tool_version=ADVANCED_SOH_PREDICTION_TOOL_VERSION,
        model_version=inference.model_version,
        data_version=inference.data_version,
        feature_version=inference.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": evidence_type,
            "artifact": {
                "record_batch_id": evidence.record_batch_id,
                "dataset_id": inference.dataset_id,
                "cell_id": inference.cell_id,
                **(
                    {"cell_metadata": cell_metadata}
                    if cell_metadata is not None
                    else {}
                ),
                "cutoff_cycle": inference.cutoff_cycle,
                "prediction_cycles": list(inference.prediction_cycles),
                "predicted_soh": list(inference.predicted_soh),
                "horizon_end_cycle": inference.prediction_cycles[-1],
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
        uncertainty={
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        },
        warnings=[],
        provenance=[
            *upstream.provenance,
            ProvenanceRecord(
                source_id=f"advanced-model-{inference.artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{inference.artifact_id}",
                sha256=inference.artifact_manifest_sha256,
                description="Verified active Advanced SOH runtime used for inference",
                created_at=created_at,
            ),
        ],
        created_at=created_at,
    )


def register_project_predict_advanced_soh_tool(
    registry: ToolRegistry,
    *,
    inference_service: AdvancedSOHInferenceService,
    project_audit_ledger: ProjectResultLedger,
    clock: Clock = _utc_now,
) -> RegisteredTool[PredictAdvancedSOHToolInput]:
    """Register finite Advanced SOH inference behind the PROJECT boundary."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
            tool_version=ADVANCED_SOH_PREDICTION_TOOL_VERSION,
            input_model=PredictAdvancedSOHToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_predict_advanced_soh_tool(
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
    "ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE",
    "ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES",
    "ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1",
    "ADVANCED_SOH_PREDICTION_TOOL_VERSION",
    "AdvancedSOHInference",
    "AdvancedSOHInferenceService",
    "PredictAdvancedSOHToolInput",
    "execute_predict_advanced_soh_tool",
    "register_project_predict_advanced_soh_tool",
]
