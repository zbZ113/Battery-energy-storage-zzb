"""Auditable ``make_batch_decision`` tool binding for the shared registry.

The binding only wraps an already-calibrated interval and quality report.  It
does not estimate life, produce an interval or invent an industrial threshold.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import (
    NormalizedPredictionInterval,
    PredictionInterval,
    ProvenanceRecord,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import DataQualityReport
from quanxin_life.decision import BatchDecisionPolicy, make_batch_decision
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

BATCH_DECISION_TOOL_VERSION = "batch-decision-tool-v1"
LifetimeInterval = PredictionInterval | NormalizedPredictionInterval


class BatchDecisionToolInput(ContractModel):
    """All upstream evidence required before deterministic batch triage."""

    prediction_interval: LifetimeInterval
    quality_report: DataQualityReport
    target_domain_calibrated: bool
    policy: BatchDecisionPolicy
    upstream_result_ids: tuple[str, ...] = Field(min_length=2)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)
    decided_at: datetime

    @field_validator("upstream_result_ids")
    @classmethod
    def require_unique_uuid_result_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("upstream_result_ids must be unique")
        try:
            for result_id in value:
                UUID(result_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_ids must contain UUID strings") from exc
        return value

    @model_validator(mode="after")
    def require_same_prediction_and_quality_context(self) -> BatchDecisionToolInput:
        if self.prediction_interval.dataset_id != self.quality_report.dataset_id:
            raise ValueError("quality_report dataset_id must match prediction interval dataset_id")
        return self


def execute_batch_decision_tool(input_value: BatchDecisionToolInput) -> ToolResult:
    """Create a registry-verifiable result from existing interval evidence only."""
    validated_input = BatchDecisionToolInput.model_validate(input_value.model_dump(mode="json"))
    outcome = make_batch_decision(
        prediction_interval=validated_input.prediction_interval,
        quality_report=validated_input.quality_report,
        target_domain_calibrated=validated_input.target_domain_calibrated,
        policy=validated_input.policy,
        decided_at=validated_input.decided_at,
    )
    calibration = validated_input.prediction_interval.calibration
    warnings = list(outcome.reason_codes)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.MAKE_BATCH_DECISION.value,
        tool_version=BATCH_DECISION_TOOL_VERSION,
        model_version=calibration.model_version,
        data_version=calibration.data_version,
        feature_version=calibration.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "cell_id": outcome.cell_id,
            "dataset_id": outcome.dataset_id,
            "decision": outcome.decision.value,
            "interval_lower_eol_cycle": outcome.interval_lower_eol_cycle,
            "interval_upper_eol_cycle": outcome.interval_upper_eol_cycle,
            "policy_version": outcome.policy_version,
            "quality_report_blocked": outcome.quality_report_blocked,
            "reason_codes": list(outcome.reason_codes),
            "required_eol_cycle": outcome.required_eol_cycle,
            "target_domain_calibrated": outcome.target_domain_calibrated,
            "upstream_result_ids": list(validated_input.upstream_result_ids),
        },
        uncertainty={
            "coverage_target": validated_input.prediction_interval.coverage_target,
            "lower_eol_cycle": outcome.interval_lower_eol_cycle,
            "upper_eol_cycle": outcome.interval_upper_eol_cycle,
        },
        warnings=warnings,
        provenance=list(validated_input.provenance),
        created_at=validated_input.decided_at,
    )


def register_batch_decision_tool(
    registry: ToolRegistry,
) -> RegisteredTool[BatchDecisionToolInput]:
    """Register the only approved implementation of ``make_batch_decision``."""
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.MAKE_BATCH_DECISION,
            tool_version=BATCH_DECISION_TOOL_VERSION,
            input_model=BatchDecisionToolInput,
            executor=execute_batch_decision_tool,
        )
    )
