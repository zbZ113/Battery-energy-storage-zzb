from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from quanxin_life.agents import AgentRole
from quanxin_life.agents.supervisor import (
    AgentPlanPolicyError,
    SupervisorPlanner,
    SupervisorPlanningRequest,
)
from quanxin_life.core import AgentPlanningMode
from quanxin_life.llm import (
    LlmProviderUnavailableError,
    LlmStructuredRequest,
    LlmStructuredResponse,
    LlmTaskPurpose,
)
from quanxin_life.tools import StandardToolName

NOW = datetime(2026, 7, 15, 18, tzinfo=UTC)


def _response(
    purpose: LlmTaskPurpose,
    payload: dict[str, Any],
) -> LlmStructuredResponse:
    return LlmStructuredResponse(
        purpose=purpose,
        prompt_version=f"{purpose.value.lower()}-prompt-v1",
        model="planner-model",
        payload=payload,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        request_hash="a" * 64,
    )


class _FakeGateway:
    def __init__(self, responses: list[LlmStructuredResponse | Exception]) -> None:
        self.responses = responses
        self.requests: list[LlmStructuredRequest] = []

    def complete_json(self, request: LlmStructuredRequest) -> LlmStructuredResponse:
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _planning_request() -> SupervisorPlanningRequest:
    return SupervisorPlanningRequest(
        project_id="quanxin-demo",
        user_goal="Assess this cell and produce an audited report.",
        dataset_ids=("hust-safe-v1",),
        requested_outputs=("cycle_life", "audited_report"),
    )


def _available_tools() -> frozenset[StandardToolName]:
    return frozenset(StandardToolName)


def _valid_decision_step() -> dict[str, Any]:
    return {
        "step_id": "decision",
        "role": "supervisor",
        "tool_name": "make_batch_decision",
        "depends_on": [],
        "input_references": {
            "prediction_interval_result_id": "context.prediction_interval_result_id",
            "calibration_result_id": "context.calibration_result_id",
            "quality_result_id": "context.quality_result_id",
            "policy_id": "context.decision_policy_id",
        },
        "requires_approval": False,
        "failure_policy": "REPLAN",
    }


def test_supervisor_builds_local_hash_and_forces_formal_decision_approval() -> None:
    gateway = _FakeGateway(
        [
            _response(
                LlmTaskPurpose.INTENT,
                {"requested_outputs": ["cycle_life", "audited_report"]},
            ),
            _response(LlmTaskPurpose.PLAN, {"steps": [_valid_decision_step()]}),
        ]
    )
    planner = SupervisorPlanner(gateway=gateway, clock=lambda: NOW)

    result = planner.plan(_planning_request(), available_tools=_available_tools())

    UUID(result.intent.intent_id)
    assert result.intent.project_id == "quanxin-demo"
    assert result.intent.dataset_ids == ("hust-safe-v1",)
    assert result.intent.goal == _planning_request().user_goal
    assert result.plan.planning_mode is AgentPlanningMode.LLM
    assert result.plan.steps[0].requires_approval is True
    assert result.plan.steps[0].role is AgentRole.SUPERVISOR
    assert len(result.plan.plan_hash) == 64
    assert result.warnings == ()
    assert [item.purpose for item in gateway.requests] == [
        LlmTaskPurpose.INTENT,
        LlmTaskPurpose.PLAN,
    ]


def test_supervisor_falls_back_when_llm_is_unavailable() -> None:
    gateway = _FakeGateway([LlmProviderUnavailableError("provider unavailable")])
    planner = SupervisorPlanner(gateway=gateway, clock=lambda: NOW)

    result = planner.plan(_planning_request(), available_tools=_available_tools())

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:LlmProviderUnavailableError",)
    assert result.plan.steps[0].tool_name == "validate_battery_data"
    assert all(
        step.tool_name in {item.value for item in _available_tools()}
        for step in result.plan.steps
    )


def test_supervisor_rejects_llm_role_tool_escalation_and_uses_fallback() -> None:
    step = {
        "step_id": "escalate",
        "role": "physics",
        "tool_name": "predict_cycle_life",
        "input_references": {"upstream_result_id": "context.early_feature_result_id"},
        "failure_policy": "STOP",
    }
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["cycle_life"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [step]}),
        ]
    )
    planner = SupervisorPlanner(gateway=gateway, clock=lambda: NOW)

    result = planner.plan(_planning_request(), available_tools=_available_tools())

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)
    assert all(step.role is not AgentRole.PHYSICS for step in result.plan.steps)


def test_supervisor_rejects_literal_numeric_tool_input_and_uses_fallback() -> None:
    step = {
        "step_id": "predict",
        "role": "lifetime",
        "tool_name": "predict_cycle_life",
        "input_references": {"upstream_result_id": "2850"},
        "failure_policy": "STOP",
    }
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["cycle_life"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [step]}),
        ]
    )

    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        _planning_request(), available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)


def test_supervisor_accepts_scenario_conversion_only_from_trusted_context() -> None:
    step = {
        "step_id": "scenario-years",
        "role": "lifetime",
        "tool_name": "convert_scenario_lifetime",
        "input_references": {
            "lifetime_result_id": "context.prediction_result_id",
            "operation_policy_version": "context.operation_policy_version",
            "equivalent_cycles_per_day": "context.equivalent_cycles_per_day",
        },
        "failure_policy": "STOP",
    }
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["scenario_years"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [step]}),
        ]
    )

    request = _planning_request().model_copy(
        update={"requested_outputs": ("scenario_years",)}
    )
    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        request, available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.LLM
    assert result.plan.steps[0].tool_name == "convert_scenario_lifetime"
    assert result.plan.steps[0].role is AgentRole.LIFETIME


def test_supervisor_rejects_literal_scenario_policy_number() -> None:
    step = {
        "step_id": "scenario-years",
        "role": "lifetime",
        "tool_name": "convert_scenario_lifetime",
        "input_references": {
            "lifetime_result_id": "context.prediction_result_id",
            "operation_policy_version": "context.operation_policy_version",
            "equivalent_cycles_per_day": "1.0",
        },
        "failure_policy": "STOP",
    }
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["scenario_years"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [step]}),
        ]
    )
    request = _planning_request().model_copy(
        update={"requested_outputs": ("scenario_years",)}
    )

    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        request, available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)


def test_supervisor_rejects_step_reference_not_declared_as_dependency() -> None:
    steps = [
        {
            "step_id": "features",
            "role": "data_quality",
            "tool_name": "extract_early_cycle_features",
            "input_references": {"record_batch_id": "intent.dataset_ids[0]"},
            "failure_policy": "STOP",
        },
        {
            "step_id": "predict",
            "role": "lifetime",
            "tool_name": "predict_cycle_life",
            "depends_on": [],
            "input_references": {"upstream_result_id": "step.features.result_id"},
            "failure_policy": "STOP",
        },
    ]
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["cycle_life"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": steps}),
        ]
    )

    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        _planning_request(), available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)


def test_supervisor_rejects_wrong_tool_input_field_names() -> None:
    step = {
        "step_id": "predict",
        "role": "lifetime",
        "tool_name": "predict_cycle_life",
        "input_references": {
            "early_feature_result_id": "context.early_feature_result_id"
        },
        "failure_policy": "STOP",
    }
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["cycle_life"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [step]}),
        ]
    )

    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        _planning_request(), available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
    assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)


def test_fixed_fallback_uses_real_tool_fields_and_stops_at_decision() -> None:
    gateway = _FakeGateway([LlmProviderUnavailableError("provider unavailable")])
    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        _planning_request(), available_tools=_available_tools()
    )

    assert tuple(step.step_id for step in result.plan.steps) == (
        "validate",
        "features",
        "predict",
        "calibration",
        "interval",
        "decision",
    )
    assert result.plan.steps[3].input_references == {
        "calibration_cohort_id": "context.calibration_cohort_id"
    }
    assert result.plan.steps[4].input_references == {
        "prediction_result_id": "step.predict.result_id",
        "calibration_result_id": "step.calibration.result_id",
    }
    assert result.plan.steps[5].input_references == {
        "prediction_interval_result_id": "step.interval.result_id",
        "calibration_result_id": "step.calibration.result_id",
        "quality_result_id": "step.validate.result_id",
        "policy_id": "context.decision_policy_id",
    }


def test_fixed_fallback_omits_unavailable_optional_tools() -> None:
    gateway = _FakeGateway([LlmProviderUnavailableError("provider unavailable")])
    planner = SupervisorPlanner(gateway=gateway, clock=lambda: NOW)
    available = frozenset(
        {
            StandardToolName.VALIDATE_BATTERY_DATA,
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            StandardToolName.PREDICT_CYCLE_LIFE,
        }
    )

    result = planner.plan(_planning_request(), available_tools=available)

    assert tuple(step.tool_name for step in result.plan.steps) == (
        "validate_battery_data",
        "extract_early_cycle_features",
        "predict_cycle_life",
    )


def test_fixed_fallback_stops_before_a_missing_prerequisite() -> None:
    gateway = _FakeGateway([LlmProviderUnavailableError("provider unavailable")])
    planner = SupervisorPlanner(gateway=gateway, clock=lambda: NOW)
    available = frozenset(
        {
            StandardToolName.VALIDATE_BATTERY_DATA,
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            StandardToolName.MAKE_BATCH_DECISION,
            StandardToolName.GENERATE_AUDITED_REPORT,
        }
    )

    result = planner.plan(_planning_request(), available_tools=available)

    assert tuple(step.tool_name for step in result.plan.steps) == ("validate_battery_data",)


def test_fixed_fallback_rejects_a_missing_dataset_instead_of_dangling() -> None:
    request = SupervisorPlanningRequest(
        project_id="quanxin-demo",
        user_goal="Assess the available battery evidence.",
        requested_outputs=("cycle_life",),
    )

    with pytest.raises(AgentPlanPolicyError, match="missing intent dataset"):
        SupervisorPlanner(gateway=None, clock=lambda: NOW).plan(
            request, available_tools=_available_tools()
        )


def test_sensitive_goal_is_never_sent_to_the_llm() -> None:
    sensitive_goals = (
        "Assess this cell with api_key=sk-secret-value.",
        "Assess raw voltage array [3.1, 3.2, 3.3, 3.4].",
        "Read -----BEGIN PRIVATE KEY----- before analysis.",
    )
    for goal in sensitive_goals:
        gateway = _FakeGateway([])
        request = _planning_request().model_copy(update={"user_goal": goal})

        result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
            request, available_tools=_available_tools()
        )

        assert gateway.requests == []
        assert result.plan.planning_mode is AgentPlanningMode.FIXED_FALLBACK
        assert result.warnings == ("LLM_PLANNING_FALLBACK:AgentPlanPolicyError",)


def test_real_dataset_identifiers_are_replaced_with_local_aliases_for_llm() -> None:
    gateway = _FakeGateway(
        [
            _response(LlmTaskPurpose.INTENT, {"requested_outputs": ["cycle_life"]}),
            _response(LlmTaskPurpose.PLAN, {"steps": [_valid_decision_step()]}),
        ]
    )

    result = SupervisorPlanner(gateway=gateway, clock=lambda: NOW).plan(
        _planning_request(), available_tools=_available_tools()
    )

    assert result.plan.planning_mode is AgentPlanningMode.LLM
    outbound_text = "\n".join(request.user_instruction for request in gateway.requests)
    assert "hust-safe-v1" not in outbound_text
    assert "dataset_0" in outbound_text
