from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.agents.orchestrator import AgentStep
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
        self.dataset_calls: list[tuple[str, str, str]] = []
        self.context_calls: list[tuple[str, str, str]] = []

    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        self.dataset_calls.append((run_id, project_id, dataset_id))
        return "verified-record-batch-v1"

    def resolve_context(self, *, run_id: str, project_id: str, reference: str) -> object:
        self.context_calls.append((run_id, project_id, reference))
        advanced_values: dict[str, object] = {
            "context.rul_point_route_role": "POINT_ACCURACY",
            "context.rul_coverage_route_role": "COVERAGE",
            "context.soh_route_role": "MEAN_ACCURACY",
            "context.conformal_calibrate_operation": "calibrate",
            "context.conformal_issue_operation": "issue",
            "context.rul_task": "RUL",
            "context.soh_task": "SOH",
            "context.conformal_alpha": 0.1,
            "context.rul_calibration_sample_result_ids": (
                "00000000-0000-4000-8000-000000000101",
            ),
            "context.soh_calibration_sample_result_ids": (
                "00000000-0000-4000-8000-000000000102",
            ),
        }
        if reference in advanced_values:
            return advanced_values[reference]
        if reference == "context.calibration_cohort_id":
            return "cohort-reviewed-v1"
        if reference == "context.operation_policy_version":
            return "one-efc-daily-v1"
        if reference == "context.equivalent_cycles_per_day":
            return 1.0
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
        AgentPlanStep(
            step_id="scenario-years",
            role=AgentRole.LIFETIME,
            tool_name=StandardToolName.CONVERT_SCENARIO_LIFETIME.value,
            depends_on=("predict",),
            input_references={
                "lifetime_result_id": "step.predict.result_id",
                "operation_policy_version": "context.operation_policy_version",
                "equivalent_cycles_per_day": "context.equivalent_cycles_per_day",
            },
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
    prediction_result = _result(StandardToolName.PREDICT_CYCLE_LIFE)
    scenario_step = compile_agent_step(
        run_id=run_id,
        intent=intent,
        plan=plan,
        step_id="scenario-years",
        context_resolver=resolver,
        completed_results={"predict": prediction_result},
    )

    assert dataset_step.input_value == {"record_batch_id": "verified-record-batch-v1"}
    assert result_step.input_value == {"upstream_result_id": feature_result.result_id}
    assert context_step.input_value == {"calibration_cohort_id": "cohort-reviewed-v1"}
    assert scenario_step.input_value == {
        "lifetime_result_id": prediction_result.result_id,
        "operation_policy_version": "one-efc-daily-v1",
        "equivalent_cycles_per_day": 1.0,
    }
    assert resolver.dataset_calls == [
        (run_id, intent.project_id, intent.dataset_ids[0]),
    ]
    assert resolver.context_calls == [
        (run_id, intent.project_id, "context.calibration_cohort_id"),
        (run_id, intent.project_id, "context.operation_policy_version"),
        (run_id, intent.project_id, "context.equivalent_cycles_per_day"),
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


def test_project_advanced_fixed_plan_compiles_only_server_owned_values() -> None:
    from quanxin_life.agents.execution_adapter import compile_agent_step
    from quanxin_life.agents.supervisor import (
        SupervisorPlanner,
        SupervisorPlanningRequest,
    )

    intent = _intent()
    planning = SupervisorPlanner(gateway=None, clock=lambda: NOW).plan(
        SupervisorPlanningRequest(
            project_id=intent.project_id,
            user_goal=intent.goal,
            dataset_ids=intent.dataset_ids,
            requested_outputs=("cycle_life", "soh", "audited_report"),
        ),
        available_tools={
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            StandardToolName.GENERATE_AUDITED_REPORT,
        },
    )
    plan = planning.plan.model_copy(update={"intent_id": intent.intent_id})
    plan = AgentPlan.build(
        plan_version=plan.plan_version,
        intent_id=intent.intent_id,
        steps=plan.steps,
        planning_mode=plan.planning_mode,
        created_at=plan.created_at,
    )
    resolver = _ContextResolver()
    run_id = str(uuid4())
    completed: dict[str, ToolResult] = {}
    compiled: dict[str, AgentStep] = {}

    for step in plan.steps:
        compiled[step.step_id] = compile_agent_step(
            run_id=run_id,
            intent=intent,
            plan=plan,
            step_id=step.step_id,
            context_resolver=resolver,
            completed_results=completed,
        )
        completed[step.step_id] = _result(StandardToolName(step.tool_name))

    assert compiled["rul-point"].input_value["route_role"] == "POINT_ACCURACY"
    assert compiled["rul-calibration"].input_value == {
        "operation": "calibrate",
        "task": "RUL",
        "route_role": "COVERAGE",
        "alpha": 0.1,
        "calibration_sample_result_ids": (
            "00000000-0000-4000-8000-000000000101",
        ),
    }
    assert compiled["report"].input_value == {
        "rul_result_id": completed["rul-point"].result_id,
        "soh_result_id": completed["soh"].result_id,
        "rul_conformal_result_id": completed["rul-interval"].result_id,
        "soh_conformal_result_id": completed["soh-band"].result_id,
    }
