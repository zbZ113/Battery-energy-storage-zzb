"""Project-bound preparation of verified Advanced model inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import Field, ValidationError, field_validator

from quanxin_life.core import (
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.features.early_cycle_sequence import PHASE_NAMES, VARIABLE_NAMES
from quanxin_life.features.multichannel_cycle import (
    CONDITION_NAMES,
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch
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

PREPARE_ADVANCED_INPUT_TOOL_VERSION = "advanced-input-tool-v1"
ADVANCED_INPUT_TRANSFORM_VERSION = "advanced-multichannel-transform-v1"
ADVANCED_INPUT_EVIDENCE_TYPE = "quanxin_life.advanced_input_evidence.v1"
Clock = Callable[[], datetime]


class ProjectEarlyCycleBatchResolver(Protocol):
    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch: ...


class PrepareAdvancedInputToolInput(ContractModel):
    """Select one project-owned canonical record batch."""

    record_batch_id: str

    @field_validator("record_batch_id")
    @classmethod
    def require_server_binding_identifier(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("record_batch_id must be a server-issued UUID") from exc


class AdvancedInputEvidence(ContractModel):
    """Strict reusable view of one route-independent Advanced input result."""

    record_batch_id: str
    dataset_id: Literal["MATR"]
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    source_manifest_hash: Sha256
    raw_sequence_input_sha256: Sha256
    transform_config_sha256: Sha256
    phase_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    condition_names: tuple[str, ...]
    normalization_version: Literal["none"]

    @field_validator("record_batch_id")
    @classmethod
    def require_server_binding_identifier(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("record_batch_id must be a server-issued UUID") from exc

    @field_validator("cutoff_cycle")
    @classmethod
    def require_supported_cutoff(cls, value: int) -> int:
        if value not in {20, 50, 100, 150}:
            raise ValueError("cutoff_cycle must be an approved Advanced cutoff")
        return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_batch(
    resolver: ProjectEarlyCycleBatchResolver,
    context: VerifiedProjectInvocationContext,
    record_batch_id: str,
) -> VerifiedEarlyCycleBatch:
    resolved = resolver.resolve_verified_early_cycle_batch(
        context,
        record_batch_id,
    )
    batch = VerifiedEarlyCycleBatch.model_validate(resolved.model_dump(mode="json"))
    if batch.record_batch_id != record_batch_id:
        raise ValueError("project batch resolver returned a mismatched binding")
    return batch


def decode_advanced_input_result(result: ToolResult) -> AdvancedInputEvidence:
    """Decode one persisted Advanced input result without trusting caller fields."""

    validated = ToolResult.model_validate(result.model_dump(mode="json"))
    if validated.tool_name != StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value:
        raise ValueError("Advanced input result has an unexpected tool_name")
    if validated.tool_version != PREPARE_ADVANCED_INPUT_TOOL_VERSION:
        raise ValueError("Advanced input result has an unsupported tool_version")
    if validated.model_version != ADVANCED_INPUT_TRANSFORM_VERSION:
        raise ValueError("Advanced input result has an unsupported model_version")
    if validated.data_version is None or validated.feature_version is None:
        raise ValueError("Advanced input result versions are incomplete")
    if set(validated.values) != {"artifact_type", "artifact"}:
        raise ValueError("Advanced input result values do not match the approved contract")
    if validated.values["artifact_type"] != ADVANCED_INPUT_EVIDENCE_TYPE:
        raise ValueError("Advanced input result has an unexpected artifact_type")
    artifact = validated.values["artifact"]
    if not isinstance(artifact, dict):
        raise ValueError("Advanced input result artifact must be an object")
    try:
        evidence = AdvancedInputEvidence.model_validate(
            {
                **artifact,
                "data_version": validated.data_version,
                "feature_version": validated.feature_version,
            }
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("Advanced input artifact does not satisfy its contract") from exc
    expected_input_hash = sha256_canonical(
        {"record_batch_id": evidence.record_batch_id}
    )
    if validated.input_hash != expected_input_hash:
        raise ValueError("Advanced input result input_hash is inconsistent")
    if evidence.phase_names != PHASE_NAMES:
        raise ValueError("Advanced input phase axis is inconsistent")
    if evidence.variable_names != VARIABLE_NAMES:
        raise ValueError("Advanced input variable axis is inconsistent")
    if evidence.condition_names != CONDITION_NAMES:
        raise ValueError("Advanced input condition axis is inconsistent")
    if not any(
        record.source_kind is SourceKind.OBSERVED
        for record in validated.provenance
    ):
        raise ValueError("Advanced input result lacks OBSERVED provenance")
    return evidence


def execute_prepare_advanced_input_tool(
    input_value: PrepareAdvancedInputToolInput,
    *,
    context: VerifiedProjectInvocationContext,
    batch_resolver: ProjectEarlyCycleBatchResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Build and hash one raw Advanced input without predicting values."""

    validated_input = PrepareAdvancedInputToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    batch = _resolve_batch(
        batch_resolver,
        context,
        validated_input.record_batch_id,
    )
    if batch.metadata.dataset_id != "MATR":
        raise ValueError("Advanced deployment input currently requires MATR data")
    cutoff_cycle = batch.feature_config.cutoff_cycle
    transform_config = MultichannelCycleConfig(
        cutoff_cycle=cutoff_cycle,
        feature_version=batch.feature_config.feature_version,
    )
    raw_sequence = build_early_cycle_sequence(
        batch.records,
        config=transform_config,
        data_version=batch.data_version,
    )
    created_at = _execution_timestamp(clock)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version=ADVANCED_INPUT_TRANSFORM_VERSION,
        data_version=batch.data_version,
        feature_version=transform_config.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": ADVANCED_INPUT_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": batch.record_batch_id,
                "dataset_id": batch.metadata.dataset_id,
                "cell_id": batch.metadata.cell_id,
                "cutoff_cycle": cutoff_cycle,
                "raw_sequence_input_sha256": raw_sequence.input_hash,
                "transform_config_sha256": sha256_canonical(
                    asdict(transform_config)
                ),
                "source_manifest_hash": batch.source_manifest_hash,
                "split_version": batch.split_version,
                "phase_names": list(PHASE_NAMES),
                "variable_names": list(VARIABLE_NAMES),
                "condition_names": list(CONDITION_NAMES),
                "normalization_version": raw_sequence.normalization_version,
            },
        },
        uncertainty=None,
        warnings=[],
        provenance=list(batch.provenance),
        created_at=created_at,
    )


def register_project_prepare_advanced_input_tool(
    registry: ToolRegistry,
    *,
    batch_resolver: ProjectEarlyCycleBatchResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[PrepareAdvancedInputToolInput]:
    """Register Advanced input preparation behind the PROJECT boundary."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
            input_model=PrepareAdvancedInputToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_prepare_advanced_input_tool(
                    input_value,
                    context=context,
                    batch_resolver=batch_resolver,
                    clock=clock,
                )
            ),
        )
    )


__all__ = [
    "ADVANCED_INPUT_EVIDENCE_TYPE",
    "ADVANCED_INPUT_TRANSFORM_VERSION",
    "PREPARE_ADVANCED_INPUT_TOOL_VERSION",
    "AdvancedInputEvidence",
    "PrepareAdvancedInputToolInput",
    "ProjectEarlyCycleBatchResolver",
    "decode_advanced_input_result",
    "execute_prepare_advanced_input_tool",
    "register_project_prepare_advanced_input_tool",
]
