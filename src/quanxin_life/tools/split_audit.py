"""Auditable, cell-level split manifest validation for shared tool clients."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest, SplitName
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

SPLIT_AUDIT_TOOL_VERSION = "split-audit-tool-v1"
SPLIT_AUDIT_MODEL_VERSION = "split-manifest-auditor-v1"
DEFAULT_REQUIRED_SPLITS = (
    SplitName.TRAIN,
    SplitName.VALIDATION,
    SplitName.CALIBRATION,
    SplitName.TEST,
)


class AuditDatasetSplitToolInput(ContractModel):
    """A versioned split manifest and its independently verified source record."""

    manifest: SplitManifest
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    required_nonempty_splits: tuple[SplitName, ...] = DEFAULT_REQUIRED_SPLITS
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)
    audited_at: datetime

    @field_validator("required_nonempty_splits")
    @classmethod
    def require_unique_required_splits(cls, value: tuple[SplitName, ...]) -> tuple[SplitName, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required_nonempty_splits must not contain duplicates")
        return value

    @field_validator("audited_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audited_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_manifest_context(self) -> AuditDatasetSplitToolInput:
        if not self.manifest.dataset_id.strip():
            raise ValueError("manifest dataset_id must not be blank")
        return self


def execute_audit_dataset_split_tool(input_value: AuditDatasetSplitToolInput) -> ToolResult:
    """Audit only cell partition membership; no data rows or labels are accessed."""
    validated_input = AuditDatasetSplitToolInput.model_validate(input_value.model_dump(mode="json"))
    manifest = validated_input.manifest
    partition_cells = {
        SplitName.TRAIN: manifest.train,
        SplitName.VALIDATION: manifest.validation,
        SplitName.CALIBRATION: manifest.calibration,
        SplitName.TEST: manifest.test,
    }
    empty_required = [
        partition.value
        for partition in validated_input.required_nonempty_splits
        if not partition_cells[partition]
    ]
    reason_codes = [f"EMPTY_REQUIRED_SPLIT:{partition}" for partition in empty_required]

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.AUDIT_DATASET_SPLIT.value,
        tool_version=SPLIT_AUDIT_TOOL_VERSION,
        model_version=SPLIT_AUDIT_MODEL_VERSION,
        data_version=validated_input.data_version,
        feature_version=validated_input.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "dataset_id": manifest.dataset_id,
            "split_version": validated_input.split_version,
            "seed": manifest.seed,
            "partition_counts": {
                partition.value: len(cells) for partition, cells in partition_cells.items()
            },
            "total_cell_count": len(manifest.all_cells),
            "cell_overlap_detected": False,
            "blocked": bool(reason_codes),
            "reason_codes": reason_codes,
        },
        uncertainty=None,
        warnings=reason_codes,
        provenance=list(validated_input.provenance),
        created_at=validated_input.audited_at,
    )


def register_audit_dataset_split_tool(
    registry: ToolRegistry,
) -> RegisteredTool[AuditDatasetSplitToolInput]:
    """Register the sole approved implementation of ``audit_dataset_split``."""
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.AUDIT_DATASET_SPLIT,
            tool_version=SPLIT_AUDIT_TOOL_VERSION,
            input_model=AuditDatasetSplitToolInput,
            executor=execute_audit_dataset_split_tool,
        )
    )
