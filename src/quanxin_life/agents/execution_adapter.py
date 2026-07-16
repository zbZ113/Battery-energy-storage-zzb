"""Compile persisted Agent plans into trusted deterministic tool invocations.

The planner stores references, never executable values.  This adapter resolves
those references exclusively from the authorized intent, server-owned runtime
context, or previously persisted ``ToolResult`` identities before handing one
step to the existing constrained orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from quanxin_life.agents.orchestrator import AgentStep
from quanxin_life.agents.supervisor import (
    DATASET_REFERENCE,
    STEP_RESULT_REFERENCE,
    SupervisorPlanner,
)
from quanxin_life.core import AgentIntent, AgentPlan, ToolResult
from quanxin_life.tools import StandardToolName


class AgentExecutionReferenceError(ValueError):
    """Raised when a persisted plan cannot be compiled from trusted references."""


class AgentExecutionContextResolver(Protocol):
    """Resolve an allowlisted server-owned context reference for one run."""

    def resolve(self, *, run_id: str, project_id: str, reference: str) -> object: ...


def compile_agent_step(
    *,
    run_id: str,
    intent: AgentIntent,
    plan: AgentPlan,
    step_id: str,
    context_resolver: AgentExecutionContextResolver,
    completed_results: Mapping[str, ToolResult],
) -> AgentStep:
    """Resolve one persisted plan step without accepting planner-provided values."""

    validated_intent, validated_plan = _validate_persisted_plan(intent, plan)
    step = next((item for item in validated_plan.steps if item.step_id == step_id), None)
    if step is None:
        raise AgentExecutionReferenceError("Agent step is not present in the persisted plan")

    resolved: dict[str, object] = {}
    for input_name, reference in step.input_references.items():
        dataset_match = DATASET_REFERENCE.fullmatch(reference)
        if dataset_match is not None:
            index = int(dataset_match.group("index"))
            try:
                validated_intent.dataset_ids[index]
            except IndexError as exc:  # pragma: no cover - planner policy is revalidated above
                raise AgentExecutionReferenceError(
                    "Agent step references a missing authorized dataset"
                ) from exc
            try:
                resolved[input_name] = context_resolver.resolve(
                    run_id=run_id,
                    project_id=validated_intent.project_id,
                    reference=reference,
                )
            except (LookupError, ValueError) as exc:
                raise AgentExecutionReferenceError(
                    "authorized dataset has no verified execution artifact"
                ) from exc
            continue

        result_match = STEP_RESULT_REFERENCE.fullmatch(reference)
        if result_match is not None:
            source_step_id = result_match.group("step_id")
            source_result = completed_results.get(source_step_id)
            if source_result is None:
                raise AgentExecutionReferenceError(
                    "Agent step dependency has no completed ToolResult"
                )
            source_step = next(
                item for item in validated_plan.steps if item.step_id == source_step_id
            )
            if source_result.tool_name != source_step.tool_name:
                raise AgentExecutionReferenceError(
                    "completed ToolResult does not match its persisted Agent step"
                )
            resolved[input_name] = source_result.result_id
            continue

        try:
            resolved[input_name] = context_resolver.resolve(
                run_id=run_id,
                project_id=validated_intent.project_id,
                reference=reference,
            )
        except (KeyError, LookupError, ValueError) as exc:
            raise AgentExecutionReferenceError(
                "server-owned Agent execution context could not resolve a reference"
            ) from exc

    try:
        return AgentStep(
            step_id=step.step_id,
            role=step.role,
            tool_name=StandardToolName(step.tool_name),
            input_value=resolved,
            requires_human_approval=step.requires_approval,
        )
    except (TypeError, ValueError) as exc:
        raise AgentExecutionReferenceError(
            "resolved Agent tool input does not satisfy the execution contract"
        ) from exc


def _validate_persisted_plan(
    intent: AgentIntent,
    plan: AgentPlan,
) -> tuple[AgentIntent, AgentPlan]:
    try:
        validated_intent = AgentIntent.model_validate(intent.model_dump(mode="json"))
        validated_plan = AgentPlan.model_validate(plan.model_dump(mode="json"))
        if validated_plan.intent_id != validated_intent.intent_id:
            raise ValueError("Agent intent and plan identities differ")
        available_tools = frozenset(
            StandardToolName(step.tool_name) for step in validated_plan.steps
        )
        secured = SupervisorPlanner._enforce_plan_policy(
            validated_plan.steps,
            intent=validated_intent,
            available_tools=available_tools,
        )
        if secured != validated_plan.steps:
            raise ValueError("persisted plan differs from the locally secured plan")
    except (TypeError, ValueError) as exc:
        raise AgentExecutionReferenceError("persisted Agent plan is not executable") from exc
    return validated_intent, validated_plan


__all__ = [
    "AgentExecutionContextResolver",
    "AgentExecutionReferenceError",
    "compile_agent_step",
]
