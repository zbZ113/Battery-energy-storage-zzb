"""Deterministic, constrained multi-agent orchestration over shared tools.

The workflow is intentionally independent of an LLM or LangGraph runtime.  A
later LangGraph adapter may construct ``AgentStep`` objects, but it cannot
bypass role allowlists, Tool Registry validation, approval gates, or audit
checks in this module.  Agents only select and invoke registered tools; they
never calculate or rewrite engineering values.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.audit import AuditLedgerError
from quanxin_life.core import AgentRole, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName, ToolRegistryError

ORCHESTRATOR_VERSION = "constrained-agent-workflow-v1"


class WorkflowStatus(StrEnum):
    COMPLETED = "COMPLETED"
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"
    FAILED = "FAILED"


ROLE_TOOL_ALLOWLIST: dict[AgentRole, frozenset[StandardToolName]] = {
    AgentRole.DATA_QUALITY: frozenset(
        {
            StandardToolName.VALIDATE_BATTERY_DATA,
            StandardToolName.AUDIT_DATASET_SPLIT,
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        }
    ),
    AgentRole.LIFETIME: frozenset(
        {
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            StandardToolName.ADAPT_TO_TARGET_DOMAIN,
            StandardToolName.INGEST_NEWLY_OBSERVED_SOH,
            StandardToolName.UPDATE_CELL_PARAMETERS,
        }
    ),
    AgentRole.PHYSICS: frozenset({StandardToolName.CHECK_OPERATING_CONDITION}),
    AgentRole.EXPERIMENT: frozenset({StandardToolName.RECOMMEND_NEXT_EXPERIMENT}),
    AgentRole.SUPERVISOR: frozenset(
        {
            StandardToolName.MAKE_BATCH_DECISION,
            StandardToolName.RETRIEVE_BATTERY_EVIDENCE,
            StandardToolName.GENERATE_AUDITED_REPORT,
        }
    ),
}


class AgentStep(ContractModel):
    """One predeclared tool invocation; natural language is not executable here."""

    step_id: str = Field(min_length=1)
    role: AgentRole
    tool_name: StandardToolName
    input_value: dict[str, Any]
    requires_human_approval: bool = False

    @field_validator("input_value")
    @classmethod
    def require_json_input(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            sha256_canonical(value)
        except TypeError as exc:
            raise ValueError("agent tool input must be JSON-compatible") from exc
        return value


class AgentWorkflowResult(ContractModel):
    """Persistable workflow state for deterministic resume and later checkpoints."""

    request_id: str
    status: WorkflowStatus
    orchestrator_version: str = ORCHESTRATOR_VERSION
    execution_plan_hash: str
    completed_step_ids: tuple[str, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    pending_step_id: str | None = None
    failure_code: str | None = None
    warnings: tuple[str, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("request_id")
    @classmethod
    def require_uuid_request_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("request_id must be a UUID string") from exc
        return value

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        return value.astimezone(UTC)


def run_constrained_workflow(
    *,
    service: ToolInvocationService,
    request_id: str,
    steps: tuple[AgentStep, ...],
    prior_result: AgentWorkflowResult | None = None,
    approved_step_ids: tuple[str, ...] = (),
) -> AgentWorkflowResult:
    """Execute an approved, auditable role-tool plan or resume it after approval."""

    _validate_request_id(request_id)
    plan_hash = _validate_plan(steps)
    completed_steps, prior_results = _validate_prior_result(
        request_id=request_id,
        steps=steps,
        plan_hash=plan_hash,
        prior_result=prior_result,
    )
    if prior_results:
        if service.audit_ledger is None:
            raise ValueError("resuming an agent workflow requires a shared audit ledger")
        for prior_tool_result in prior_results:
            service.audit_ledger.ensure_result(prior_tool_result)
    approved = set(approved_step_ids)
    results = list(prior_results)

    for step in steps[len(completed_steps) :]:
        if step.requires_human_approval and step.step_id not in approved:
            return _result(
                request_id=request_id,
                status=WorkflowStatus.AWAITING_HUMAN_APPROVAL,
                plan_hash=plan_hash,
                completed_step_ids=completed_steps,
                tool_results=tuple(results),
                pending_step_id=step.step_id,
            )
        try:
            result = service.invoke_for_agent(
                ToolInvocation(tool_name=step.tool_name, input_value=step.input_value),
                allowed_tool_names=ROLE_TOOL_ALLOWLIST[step.role],
            )
        except (AuditLedgerError, ToolRegistryError) as exc:
            return _result(
                request_id=request_id,
                status=WorkflowStatus.FAILED,
                plan_hash=plan_hash,
                completed_step_ids=completed_steps,
                tool_results=tuple(results),
                failure_code=f"TOOL_EXECUTION_BLOCKED:{type(exc).__name__}",
                warnings=(str(exc),),
            )
        results.append(result)
        completed_steps = (*completed_steps, step.step_id)

    return _result(
        request_id=request_id,
        status=WorkflowStatus.COMPLETED,
        plan_hash=plan_hash,
        completed_step_ids=completed_steps,
        tool_results=tuple(results),
    )


def _validate_plan(steps: tuple[AgentStep, ...]) -> str:
    if not steps:
        raise ValueError("agent workflow requires at least one step")
    step_ids = tuple(step.step_id for step in steps)
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("agent workflow step_ids must be unique")
    for step in steps:
        allowed_tools = ROLE_TOOL_ALLOWLIST[step.role]
        if step.tool_name not in allowed_tools:
            raise ValueError(
                f"tool '{step.tool_name.value}' is not allowed for role '{step.role.value}'"
            )
    return sha256_canonical([step.model_dump(mode="json") for step in steps])


def _validate_prior_result(
    *,
    request_id: str,
    steps: tuple[AgentStep, ...],
    plan_hash: str,
    prior_result: AgentWorkflowResult | None,
) -> tuple[tuple[str, ...], tuple[ToolResult, ...]]:
    if prior_result is None:
        return (), ()
    if prior_result.request_id != request_id:
        raise ValueError("prior workflow request_id must match the current request")
    if prior_result.execution_plan_hash != plan_hash:
        raise ValueError("prior workflow execution plan does not match the current plan")
    if len(prior_result.completed_step_ids) != len(prior_result.tool_results):
        raise ValueError("prior workflow completed steps and ToolResults must have equal length")
    completed_count = len(prior_result.completed_step_ids)
    completed_plan_steps = steps[:completed_count]
    expected_step_ids = tuple(step.step_id for step in completed_plan_steps)
    if prior_result.completed_step_ids != expected_step_ids:
        raise ValueError("prior workflow steps must be a sequential prefix of the current plan")
    validated_results: list[ToolResult] = []
    for step, tool_result in zip(completed_plan_steps, prior_result.tool_results, strict=True):
        _validate_prior_tool_result(step, tool_result)
        validated_results.append(tool_result)
    return prior_result.completed_step_ids, tuple(validated_results)


def _validate_prior_tool_result(step: AgentStep, tool_result: ToolResult) -> None:
    try:
        validated = ToolResult.model_validate(tool_result.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise ValueError("prior ToolResult does not satisfy the public audit contract") from exc
    if validated.tool_name != step.tool_name.value:
        raise ValueError("prior ToolResult tool_name does not match its completed step")
    if validated.input_hash != sha256_canonical(step.input_value):
        raise ValueError("prior ToolResult input_hash does not match its completed step")


def _validate_request_id(request_id: str) -> None:
    try:
        UUID(request_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("request_id must be a UUID string") from exc


def _result(
    *,
    request_id: str,
    status: WorkflowStatus,
    plan_hash: str,
    completed_step_ids: tuple[str, ...],
    tool_results: tuple[ToolResult, ...],
    pending_step_id: str | None = None,
    failure_code: str | None = None,
    warnings: tuple[str, ...] = (),
) -> AgentWorkflowResult:
    return AgentWorkflowResult(
        request_id=request_id,
        status=status,
        execution_plan_hash=plan_hash,
        completed_step_ids=completed_step_ids,
        tool_results=tool_results,
        pending_step_id=pending_step_id,
        failure_code=failure_code,
        warnings=warnings,
    )
