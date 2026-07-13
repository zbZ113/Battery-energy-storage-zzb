from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry


class _Input(ContractModel):
    request_label: str = Field(min_length=1)


def _tool_result(input_value: _Input, *, tool_name: StandardToolName) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version="agent-test-tool-v1",
        model_version="agent-test-model-v1",
        data_version="agent-test-data-v1",
        feature_version="agent-test-feature-v1",
        input_hash=sha256_canonical(input_value.model_dump(mode="json")),
        values={"execution_status": "computed"},
        provenance=[
            ProvenanceRecord(
                source_id="agent-test-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://agent/source",
                sha256=sha256_canonical({"source": "agent-test"}),
                description="Agent orchestration test provenance",
                created_at=datetime(2026, 7, 13, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_name in (
        StandardToolName.VALIDATE_BATTERY_DATA,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.CHECK_OPERATING_CONDITION,
        StandardToolName.RECOMMEND_NEXT_EXPERIMENT,
        StandardToolName.GENERATE_AUDITED_REPORT,
    ):
        registry.register(
            ToolDefinition(
                tool_name=tool_name,
                tool_version="agent-test-tool-v1",
                input_model=_Input,
                executor=lambda input_value, name=tool_name: _tool_result(
                    input_value, tool_name=name
                ),
            )
        )
    return registry


def _step(role: object, tool_name: StandardToolName, *, approval: bool = False):
    from quanxin_life.agents.orchestrator import AgentStep

    return AgentStep(
        step_id=f"{role.value}-{tool_name.value}",
        role=role,
        tool_name=tool_name,
        input_value={"request_label": "batch-A"},
        requires_human_approval=approval,
    )


def test_constrained_agents_execute_only_whitelisted_tools_and_preserve_results() -> None:
    from quanxin_life.agents.orchestrator import AgentRole, run_constrained_workflow

    outcome = run_constrained_workflow(
        registry=_registry(),
        request_id=str(uuid4()),
        steps=(
            _step(AgentRole.DATA_QUALITY, StandardToolName.VALIDATE_BATTERY_DATA),
            _step(AgentRole.LIFETIME, StandardToolName.PREDICT_CYCLE_LIFE),
            _step(AgentRole.PHYSICS, StandardToolName.CHECK_OPERATING_CONDITION),
            _step(AgentRole.EXPERIMENT, StandardToolName.RECOMMEND_NEXT_EXPERIMENT),
            _step(AgentRole.SUPERVISOR, StandardToolName.GENERATE_AUDITED_REPORT),
        ),
    )

    assert outcome.status.value == "COMPLETED"
    assert len(outcome.tool_results) == 5
    assert all(result.values == {"execution_status": "computed"} for result in outcome.tool_results)
    assert outcome.pending_step_id is None


def test_agent_rejects_role_tool_boundary_before_execution() -> None:
    from quanxin_life.agents.orchestrator import AgentRole, run_constrained_workflow

    with pytest.raises(ValueError, match="not allowed"):
        run_constrained_workflow(
            registry=_registry(),
            request_id=str(uuid4()),
            steps=(
                _step(AgentRole.PHYSICS, StandardToolName.PREDICT_CYCLE_LIFE),
            ),
        )


def test_workflow_stops_for_human_approval_and_resumes_without_replanning() -> None:
    from quanxin_life.agents.orchestrator import AgentRole, WorkflowStatus, run_constrained_workflow

    steps = (
        _step(AgentRole.DATA_QUALITY, StandardToolName.VALIDATE_BATTERY_DATA),
        _step(
            AgentRole.SUPERVISOR,
            StandardToolName.GENERATE_AUDITED_REPORT,
            approval=True,
        ),
    )
    paused = run_constrained_workflow(
        registry=_registry(), request_id=str(uuid4()), steps=steps
    )

    assert paused.status is WorkflowStatus.AWAITING_HUMAN_APPROVAL
    assert len(paused.tool_results) == 1
    assert paused.pending_step_id == steps[1].step_id

    resumed = run_constrained_workflow(
        registry=_registry(),
        request_id=paused.request_id,
        steps=steps,
        prior_result=paused,
        approved_step_ids=(steps[1].step_id,),
    )
    assert resumed.status is WorkflowStatus.COMPLETED
    assert len(resumed.tool_results) == 2


def test_workflow_rejects_a_tampered_prior_result() -> None:
    from quanxin_life.agents.orchestrator import AgentRole, run_constrained_workflow

    steps = (_step(AgentRole.DATA_QUALITY, StandardToolName.VALIDATE_BATTERY_DATA),)
    completed = run_constrained_workflow(
        registry=_registry(), request_id=str(uuid4()), steps=steps
    )
    tampered = completed.model_copy(
        update={
            "tool_results": (
                completed.tool_results[0].model_copy(update={"input_hash": "0" * 64}),
            )
        }
    )

    with pytest.raises(ValueError, match="prior ToolResult"):
        run_constrained_workflow(
            registry=_registry(),
            request_id=completed.request_id,
            steps=steps,
            prior_result=tampered,
        )
