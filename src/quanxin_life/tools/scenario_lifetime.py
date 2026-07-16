"""Audited conversion from predicted cycle life to scenario years.

This tool does not predict calendar life.  It only applies an explicit,
versioned operating-policy assumption to a verified EOL80 cycle prediction.
The resulting years value is therefore model inference and must never be
presented as a 15--25 year observation or warranty conclusion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    LifePrediction,
    ScenarioLifetimeRequest,
    ScenarioLifetimeResult,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.tools.cycle_life_prediction import (
    PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

SCENARIO_LIFETIME_TOOL_VERSION = "scenario-lifetime-tool-v1"
SCENARIO_LIFETIME_ARTIFACT_TYPE = "quanxin_life.scenario_lifetime.v1"
SCENARIO_CONVERSION_WARNING = "SCENARIO_CONVERSION_NOT_OBSERVED_YEARS"
_LIMITATIONS = (
    "This is a policy-based conversion of a model-predicted EOL80 cycle count, "
    "not observed calendar life.",
    "The result changes when equivalent daily cycles or the operating policy changes.",
    "Calendar aging, downtime, dispatch variation and site conditions are not "
    "introduced by this conversion.",
)
Clock = Callable[[], datetime]

# Reuse the public core contract instead of defining a semantically duplicate DTO.
ScenarioLifetimeToolInput = ScenarioLifetimeRequest


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scenario lifetime execution timestamp must include a timezone")
    return value.astimezone(UTC)


def _decode_life_prediction(result: ToolResult) -> LifePrediction:
    if result.tool_name != StandardToolName.PREDICT_CYCLE_LIFE.value:
        raise ValueError("scenario lifetime requires a predict_cycle_life ToolResult")
    artifact_type = result.values.get("artifact_type")
    artifact = result.values.get("artifact")
    if artifact_type != PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE or not isinstance(
        artifact, Mapping
    ):
        raise ValueError("cycle-life ToolResult has an unsupported evidence artifact")
    prediction = artifact.get("life_prediction")
    if not isinstance(prediction, Mapping):
        raise ValueError("cycle-life ToolResult is missing life_prediction evidence")
    try:
        validated = LifePrediction.model_validate(dict(prediction))
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("cycle-life ToolResult contains invalid prediction evidence") from exc
    if validated.model_version != result.model_version:
        raise ValueError("cycle-life model version does not match ToolResult metadata")
    if validated.data_version != result.data_version:
        raise ValueError("cycle-life data version does not match ToolResult metadata")
    if validated.feature_version != result.feature_version:
        raise ValueError("cycle-life feature version does not match ToolResult metadata")
    return validated


def execute_scenario_lifetime_tool(
    input_value: ScenarioLifetimeToolInput,
    *,
    audit_ledger: AuditLedger,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Convert a ledger-bound EOL80 cycle prediction using explicit assumptions."""

    validated_input = ScenarioLifetimeToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    upstream = audit_ledger.resolve_registered_result(
        validated_input.lifetime_result_id
    )
    prediction = _decode_life_prediction(upstream)
    scenario_years = prediction.predicted_eol_cycle / (
        validated_input.equivalent_cycles_per_day * validated_input.days_per_year
    )
    scenario = ScenarioLifetimeResult(
        source_lifetime_result_id=upstream.result_id,
        operation_policy_version=validated_input.operation_policy_version,
        equivalent_cycles_per_day=validated_input.equivalent_cycles_per_day,
        days_per_year=validated_input.days_per_year,
        scenario_years=scenario_years,
        limitations=_LIMITATIONS,
    )
    artifact: dict[str, Any] = scenario.model_dump(mode="json")
    artifact.update(
        {
            "dataset_id": prediction.dataset_id,
            "cell_id": prediction.cell_id,
            "predicted_eol_cycle": prediction.predicted_eol_cycle,
            "cutoff_cycle": prediction.cutoff_cycle,
        }
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CONVERT_SCENARIO_LIFETIME.value,
        tool_version=SCENARIO_LIFETIME_TOOL_VERSION,
        model_version=upstream.model_version,
        data_version=upstream.data_version,
        feature_version=upstream.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": SCENARIO_LIFETIME_ARTIFACT_TYPE,
            "artifact": artifact,
        },
        uncertainty=None,
        warnings=[SCENARIO_CONVERSION_WARNING],
        provenance=upstream.provenance,
        created_at=_execution_timestamp(clock),
    )


def register_scenario_lifetime_tool(
    registry: ToolRegistry,
    *,
    audit_ledger: AuditLedger,
    clock: Clock = _utc_now,
) -> RegisteredTool[ScenarioLifetimeToolInput]:
    """Register the sole audited scenario conversion implementation."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.CONVERT_SCENARIO_LIFETIME,
            tool_version=SCENARIO_LIFETIME_TOOL_VERSION,
            input_model=ScenarioLifetimeToolInput,
            executor=lambda input_value: execute_scenario_lifetime_tool(
                input_value,
                audit_ledger=audit_ledger,
                clock=clock,
            ),
        )
    )


__all__ = [
    "SCENARIO_CONVERSION_WARNING",
    "SCENARIO_LIFETIME_ARTIFACT_TYPE",
    "SCENARIO_LIFETIME_TOOL_VERSION",
    "ScenarioLifetimeToolInput",
    "execute_scenario_lifetime_tool",
    "register_scenario_lifetime_tool",
]
