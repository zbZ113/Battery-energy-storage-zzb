"""Auditable binding for short-horizon PyBaMM operating-condition checks.

This tool only transports results produced by the physics validator.  It has
no lifetime, EOL or RUL fields and preserves unavailable or failed solver
states instead of substituting simulated values.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import Field, field_validator

from quanxin_life.core import ProvenanceRecord, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.physics import (
    PhysicsValidationRequest,
    PhysicsValidationResult,
    validate_short_horizon,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

PHYSICS_CHECK_TOOL_VERSION = "physics-check-tool-v1"
PhysicsValidator = Callable[[PhysicsValidationRequest], PhysicsValidationResult]


class CheckOperatingConditionToolInput(ContractModel):
    """Versioned operating request and its source evidence for a short check."""

    request: PhysicsValidationRequest
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)
    checked_at: datetime

    @field_validator("checked_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must include a timezone")
        return value.astimezone(UTC)


def execute_check_operating_condition_tool(
    input_value: CheckOperatingConditionToolInput,
    *,
    validator: PhysicsValidator = validate_short_horizon,
) -> ToolResult:
    """Run one bounded physical check and expose its direct solver evidence."""
    validated_input = CheckOperatingConditionToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    physics_result = PhysicsValidationResult.model_validate(
        validator(validated_input.request).model_dump(mode="json")
    )
    physics_payload = physics_result.model_dump(mode="json")
    warnings = list(dict.fromkeys((*physics_result.warnings, *physics_result.boundary_risks)))

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CHECK_OPERATING_CONDITION.value,
        tool_version=PHYSICS_CHECK_TOOL_VERSION,
        model_version=physics_result.model_version,
        data_version=validated_input.data_version,
        feature_version=validated_input.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "status": physics_result.status.value,
            "request": physics_payload["request"],
            "samples": physics_payload["samples"],
            "termination_reason": physics_result.termination_reason,
            "boundary_risks": list(physics_result.boundary_risks),
            "model_name": physics_result.model_name,
            "physics_model_version": physics_result.model_version,
            "parameter_set": physics_result.parameter_set,
            "parameter_set_version": physics_result.parameter_set_version,
            "pybamm_version": physics_result.pybamm_version,
            "validator_version": physics_result.validator_version,
            "warnings": list(physics_result.warnings),
            "configuration": physics_payload["configuration"],
        },
        uncertainty={"short_horizon_only": True},
        warnings=warnings,
        provenance=list(validated_input.provenance),
        created_at=validated_input.checked_at,
    )


def register_check_operating_condition_tool(
    registry: ToolRegistry,
    *,
    validator: PhysicsValidator = validate_short_horizon,
) -> RegisteredTool[CheckOperatingConditionToolInput]:
    """Register the sole approved short-horizon physics validation tool."""
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.CHECK_OPERATING_CONDITION,
            tool_version=PHYSICS_CHECK_TOOL_VERSION,
            input_model=CheckOperatingConditionToolInput,
            executor=lambda input_value: execute_check_operating_condition_tool(
                input_value,
                validator=validator,
            ),
        )
    )
