"""Auditable binding for deterministic battery data-quality validation.

The tool delegates every engineering judgment to
``data.validation.validate_cycle_records``.  It neither repairs records nor
creates battery-health estimates; the returned score and issue list are the
direct output of the deterministic validation rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import Field, field_validator

from quanxin_life.core import ProvenanceRecord, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.validation import validate_cycle_records
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

DATA_QUALITY_TOOL_VERSION = "data-quality-tool-v1"
DATA_QUALITY_MODEL_VERSION = "data-quality-rule-engine-v1"


class ValidateBatteryDataToolInput(ContractModel):
    """Evidence and version declarations needed for deterministic validation."""

    records: tuple[CycleRecord, ...]
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)
    validated_at: datetime

    @field_validator("validated_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("validated_at must include a timezone")
        return value.astimezone(UTC)


def execute_validate_battery_data_tool(
    input_value: ValidateBatteryDataToolInput,
) -> ToolResult:
    """Validate raw records and return the deterministic result with provenance."""
    validated_input = ValidateBatteryDataToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    report = validate_cycle_records(validated_input.records)
    issues = [issue.model_dump(mode="json") for issue in report.issues]
    warnings = list(dict.fromkeys(issue["code"] for issue in issues))

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
        tool_version=DATA_QUALITY_TOOL_VERSION,
        model_version=DATA_QUALITY_MODEL_VERSION,
        data_version=validated_input.data_version,
        feature_version=validated_input.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "dataset_id": report.dataset_id,
            "blocked": report.blocked,
            "quality_score": report.quality_score,
            "issue_count": len(issues),
            "issues": issues,
        },
        uncertainty=None,
        warnings=warnings,
        provenance=list(validated_input.provenance),
        created_at=validated_input.validated_at,
    )


def register_validate_battery_data_tool(
    registry: ToolRegistry,
) -> RegisteredTool[ValidateBatteryDataToolInput]:
    """Register the sole approved implementation of ``validate_battery_data``."""
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version=DATA_QUALITY_TOOL_VERSION,
            input_model=ValidateBatteryDataToolInput,
            executor=execute_validate_battery_data_tool,
        )
    )
