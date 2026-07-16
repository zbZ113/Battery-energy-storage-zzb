from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import (
    AgentFailurePolicy,
    AgentIntent,
    AgentPlan,
    AgentPlanStep,
    AgentRole,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.tools import StandardToolName

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


class _ContextResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def resolve(self, *, run_id: str, project_id: str, reference: str) -> object:
        self.calls.append((run_id, project_id, reference))
        if reference.startswith("intent.dataset_ids["):
            return "verified-record-batch-v1"
        if reference == "context.calibration_cohort_id":
            return "cohort-reviewed-v1"
        raise KeyError(reference)


def _intent() -> AgentIntent:
    return AgentIntent(
        intent_id=str(uuid4()),
        project_id=str(uuid4()),
        goal="analyze the frozen cell dataset",
        dataset_ids=(str(uuid4()),),
        requested_outputs=("cycle_life",),
        created_at=NOW,
    )


def _result(tool_name: StandardToolName) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version="test-tool-v1",
        model_version=None,
        data_version="test-data-v1",
        feature_version="test-feature-v1",
        input_hash=sha256_canonical({"trusted": "input"}),
        values={"status": "computed"},
        provenance=[
            ProvenanceRecord(
                source_id="test-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://source",
                sha256=sha256_canonical({"source": "test"}),
                description="Test-only source",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_compile_step_resolves_only_intent_context_and_prior_result_references() -> None:
    from quanxin_life.agents.execution_adapter import compile_agent_step

    intent = _intent()
    steps = (
        AgentPlanStep(
            step_id="features",
            role=AgentRole.DATA_QUALITY,
            tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
            input_references={"record_batch_id": "intent.dataset_ids[0]"},
            failure_policy=AgentFailurePolicy.STOP,
        ),
        AgentPlanStep(
            step_id="predict",
            role=AgentRole.LIFETIME,
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            depends_on=("features",),
            input_references={"upstream_result_id": "step.features.result_id"},
            failure_policy=AgentFailurePolicy.STOP,
        ),
        AgentPlanStep(
            step_id="calibration",
            role=AgentRole.LIFETIME,
            tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
            input_references={"calibration_cohort_id": "context.calibration_cohort_id"},
            failure_policy=AgentFailurePolicy.STOP,
        ),
    )
    plan = AgentPlan.build(
        plan_version="test-plan-v1",
        intent_id=intent.intent_id,
        steps=steps,
        created_at=NOW,
    )
    feature_result = _result(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES)
    resolver = _ContextResolver()
    run_id = str(uuid4())

    dataset_step = compile_agent_step(
        run_id=run_id,
        intent=intent,
        plan=plan,
        step_id="features",
        context_resolver=resolver,
        completed_results={},
    )
    result_step = compile_agent_step(
        run_id=run_id,
        intent=intent,
        plan=plan,
        step_id="predict",
        context_resolver=resolver,
        completed_results={"features": feature_result},
    )
    context_step = compile_agent_step(
        run_id=run_id,
        intent=intent,
        plan=plan,
        step_id="calibration",
        context_resolver=resolver,
        completed_results={},
    )

    assert dataset_step.input_value == {"record_batch_id": "verified-record-batch-v1"}
    assert result_step.input_value == {"upstream_result_id": feature_result.result_id}
    assert context_step.input_value == {"calibration_cohort_id": "cohort-reviewed-v1"}
    assert resolver.calls == [
        (run_id, intent.project_id, "intent.dataset_ids[0]"),
        (run_id, intent.project_id, "context.calibration_cohort_id"),
    ]


def test_compile_step_fails_closed_when_dependency_result_is_missing() -> None:
    from quanxin_life.agents.execution_adapter import (
        AgentExecutionReferenceError,
        compile_agent_step,
    )

    intent = _intent()
    step = AgentPlanStep(
        step_id="predict",
        role=AgentRole.LIFETIME,
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        depends_on=("features",),
        input_references={"upstream_result_id": "step.features.result_id"},
        failure_policy=AgentFailurePolicy.STOP,
    )
    prerequisite = AgentPlanStep(
        step_id="features",
        role=AgentRole.DATA_QUALITY,
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        input_references={"record_batch_id": "intent.dataset_ids[0]"},
        failure_policy=AgentFailurePolicy.STOP,
    )
    plan = AgentPlan.build(
        plan_version="test-plan-v1",
        intent_id=intent.intent_id,
        steps=(prerequisite, step),
        created_at=NOW,
    )

    with pytest.raises(AgentExecutionReferenceError, match="completed ToolResult"):
        compile_agent_step(
            run_id=str(uuid4()),
            intent=intent,
            plan=plan,
            step_id="predict",
            context_resolver=_ContextResolver(),
            completed_results={},
        )


def test_compile_step_revalidates_persisted_plan_hash() -> None:
    from quanxin_life.agents.execution_adapter import (
        AgentExecutionReferenceError,
        compile_agent_step,
    )

    intent = _intent()
    step = AgentPlanStep(
        step_id="features",
        role=AgentRole.DATA_QUALITY,
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        input_references={"record_batch_id": "intent.dataset_ids[0]"},
    )
    plan = AgentPlan.build(
        plan_version="test-plan-v1",
        intent_id=intent.intent_id,
        steps=(step,),
        created_at=NOW,
    ).model_copy(update={"plan_hash": "0" * 64})

    with pytest.raises(AgentExecutionReferenceError, match="persisted Agent plan"):
        compile_agent_step(
            run_id=str(uuid4()),
            intent=intent,
            plan=plan,
            step_id="features",
            context_resolver=_ContextResolver(),
            completed_results={},
        )
