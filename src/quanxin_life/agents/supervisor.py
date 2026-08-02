"""Supervisor planning with strict local validation and deterministic fallback."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field, ValidationError, field_validator

from quanxin_life.agents.orchestrator import ROLE_TOOL_ALLOWLIST
from quanxin_life.core import (
    AgentFailurePolicy,
    AgentIntent,
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.llm import (
    LlmGatewayError,
    LlmStructuredRequest,
    LlmStructuredResponse,
    LlmTaskPurpose,
)
from quanxin_life.tools import StandardToolName

SUPERVISOR_PLANNER_VERSION = "supervisor-planner-v1"
INTENT_PROMPT_VERSION = "intent-prompt-v1"
PLAN_PROMPT_VERSION = "plan-prompt-v1"
ADVANCED_SINGLE_CELL_OUTPUT = "advanced_single_cell_analysis"
FIXED_ADVANCED_STEP_IDS = (
    "advanced-input",
    "rul-point",
    "rul-coverage",
    "soh",
    "rul-calibration",
    "rul-interval",
    "soh-calibration",
    "soh-band",
    "report",
)
FIXED_ADVANCED_TOOLS = frozenset(
    {
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        StandardToolName.GENERATE_AUDITED_REPORT,
    }
)
FORMAL_APPROVAL_TOOLS = frozenset({StandardToolName.MAKE_BATCH_DECISION})
PLANNABLE_INPUT_MODES: dict[StandardToolName, tuple[frozenset[str], ...]] = {
    StandardToolName.VALIDATE_BATTERY_DATA: (
        frozenset(
            {"records", "data_version", "feature_version", "provenance", "validated_at"}
        ),
    ),
    StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES: (frozenset({"record_batch_id"}),),
    StandardToolName.PREDICT_CYCLE_LIFE: (
        frozenset({"upstream_result_id"}),
        frozenset({"upstream_result_id", "route_role"}),
    ),
    StandardToolName.PREDICT_SOH_TRAJECTORY: (
        frozenset({"upstream_result_id", "route_role"}),
    ),
    StandardToolName.CONVERT_SCENARIO_LIFETIME: (
        frozenset(
            {
                "lifetime_result_id",
                "operation_policy_version",
                "equivalent_cycles_per_day",
            }
        ),
        frozenset(
            {
                "lifetime_result_id",
                "operation_policy_version",
                "equivalent_cycles_per_day",
                "days_per_year",
            }
        ),
    ),
    StandardToolName.CALIBRATE_PREDICTION_INTERVAL: (
        frozenset({"calibration_cohort_id"}),
        frozenset({"prediction_result_id", "calibration_result_id"}),
        frozenset(
            {
                "operation",
                "task",
                "route_role",
                "alpha",
                "calibration_sample_result_ids",
            }
        ),
        frozenset(
            {
                "operation",
                "task",
                "route_role",
                "prediction_result_id",
                "calibration_result_id",
            }
        ),
    ),
    StandardToolName.MAKE_BATCH_DECISION: (
        frozenset(
            {
                "prediction_interval_result_id",
                "calibration_result_id",
                "quality_result_id",
                "policy_id",
            }
        ),
    ),
    StandardToolName.GENERATE_AUDITED_REPORT: (
        frozenset(
            {
                "rul_result_id",
                "soh_result_id",
                "rul_conformal_result_id",
                "soh_conformal_result_id",
            }
        ),
    ),
}
ALLOWED_CONTEXT_REFERENCES: dict[str, frozenset[str]] = {
    "records": frozenset({"context.validation_records"}),
    "data_version": frozenset({"context.data_version"}),
    "feature_version": frozenset({"context.feature_version"}),
    "provenance": frozenset({"context.provenance"}),
    "validated_at": frozenset({"context.validated_at"}),
    "upstream_result_id": frozenset({"context.early_feature_result_id"}),
    "route_role": frozenset(
        {
            "context.rul_point_route_role",
            "context.rul_coverage_route_role",
            "context.soh_route_role",
        }
    ),
    "operation": frozenset(
        {
            "context.conformal_calibrate_operation",
            "context.conformal_issue_operation",
        }
    ),
    "task": frozenset({"context.rul_task", "context.soh_task"}),
    "alpha": frozenset({"context.conformal_alpha"}),
    "calibration_sample_result_ids": frozenset(
        {
            "context.rul_calibration_sample_result_ids",
            "context.soh_calibration_sample_result_ids",
        }
    ),
    "lifetime_result_id": frozenset({"context.prediction_result_id"}),
    "operation_policy_version": frozenset({"context.operation_policy_version"}),
    "equivalent_cycles_per_day": frozenset({"context.equivalent_cycles_per_day"}),
    "days_per_year": frozenset({"context.days_per_year"}),
    "calibration_cohort_id": frozenset({"context.calibration_cohort_id"}),
    "prediction_result_id": frozenset({"context.prediction_result_id"}),
    "calibration_result_id": frozenset({"context.calibration_result_id"}),
    "prediction_interval_result_id": frozenset(
        {"context.prediction_interval_result_id"}
    ),
    "quality_result_id": frozenset({"context.quality_result_id"}),
    "policy_id": frozenset({"context.decision_policy_id"}),
}
EXPECTED_RESULT_SOURCE_TOOLS: dict[str, frozenset[StandardToolName]] = {
    "upstream_result_id": frozenset({StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES}),
    "prediction_result_id": frozenset(
        {
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
        }
    ),
    "lifetime_result_id": frozenset({StandardToolName.PREDICT_CYCLE_LIFE}),
    "calibration_result_id": frozenset(
        {StandardToolName.CALIBRATE_PREDICTION_INTERVAL}
    ),
    "prediction_interval_result_id": frozenset(
        {StandardToolName.CALIBRATE_PREDICTION_INTERVAL}
    ),
    "quality_result_id": frozenset({StandardToolName.VALIDATE_BATTERY_DATA}),
    "rul_result_id": frozenset({StandardToolName.PREDICT_CYCLE_LIFE}),
    "soh_result_id": frozenset({StandardToolName.PREDICT_SOH_TRAJECTORY}),
    "rul_conformal_result_id": frozenset(
        {StandardToolName.CALIBRATE_PREDICTION_INTERVAL}
    ),
    "soh_conformal_result_id": frozenset(
        {StandardToolName.CALIBRATE_PREDICTION_INTERVAL}
    ),
}
STEP_RESULT_REFERENCE = re.compile(r"^step\.(?P<step_id>[A-Za-z0-9_-]+)\.result_id$")
DATASET_REFERENCE = re.compile(r"^intent\.dataset_ids\[(?P<index>[0-9]+)\]$")
SENSITIVE_OUTBOUND_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"\[(?:\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*,){3,}"
        r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\]"
    ),
)
Clock = Callable[[], datetime]
UuidFactory = Callable[[], UUID]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AgentPlanPolicyError(RuntimeError):
    """Raised when an LLM draft violates a local role or tool boundary."""


class StructuredPlanningGateway(Protocol):
    def complete_json(self, request: LlmStructuredRequest) -> LlmStructuredResponse: ...


class SupervisorPlanningRequest(ContractModel):
    """Server-owned context plus one bounded natural-language goal."""

    project_id: str = Field(min_length=1)
    user_goal: str = Field(min_length=2, max_length=4_000)
    dataset_ids: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = Field(min_length=1)

    @field_validator("project_id", "user_goal")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Supervisor planning text must not be blank")
        return normalized

    @field_validator("dataset_ids", "requested_outputs")
    @classmethod
    def sequences_are_unique(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("Supervisor planning identifiers must not be blank")
        if len(normalized) != len(set(normalized)):
            field_name = getattr(info, "field_name", "values")
            raise ValueError(f"{field_name} must be unique")
        return normalized


class _IntentDraft(ContractModel):
    requested_outputs: tuple[str, ...] = Field(min_length=1)


class _PlanDraft(ContractModel):
    steps: tuple[AgentPlanStep, ...] = Field(min_length=1, max_length=12)


class SupervisorPlanningResult(ContractModel):
    """One validated intent and plan, with explicit fallback warnings."""

    intent: AgentIntent
    plan: AgentPlan
    planner_version: str = SUPERVISOR_PLANNER_VERSION
    warnings: tuple[str, ...] = ()


class SupervisorPlanner:
    """Use an LLM for drafts while retaining all authority in local code."""

    def __init__(
        self,
        *,
        gateway: StructuredPlanningGateway | None,
        clock: Clock = _utc_now,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        self._gateway = gateway
        self._clock = clock
        self._uuid_factory = uuid_factory

    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: Collection[StandardToolName],
    ) -> SupervisorPlanningResult:
        validated = SupervisorPlanningRequest.model_validate(request.model_dump(mode="json"))
        available = frozenset(available_tools)
        if not available:
            raise ValueError("Supervisor planning requires at least one available tool")
        now = self._now()
        if self._gateway is None:
            return self._fallback(
                validated,
                available_tools=available,
                created_at=now,
                warning="LLM_PLANNING_FALLBACK:NO_GATEWAY",
            )
        try:
            self._assert_llm_outbound_safe(validated.user_goal)
            intent = self._create_intent(validated, created_at=now)
            plan = self._create_llm_plan(intent, available_tools=available, created_at=now)
        except (AgentPlanPolicyError, LlmGatewayError, ValidationError, ValueError) as exc:
            return self._fallback(
                validated,
                available_tools=available,
                created_at=now,
                warning=f"LLM_PLANNING_FALLBACK:{type(exc).__name__}",
            )
        return SupervisorPlanningResult(intent=intent, plan=plan)

    def _create_intent(
        self,
        request: SupervisorPlanningRequest,
        *,
        created_at: datetime,
    ) -> AgentIntent:
        if self._gateway is None:
            raise AgentPlanPolicyError("LLM gateway is unavailable")
        response = self._gateway.complete_json(
            LlmStructuredRequest(
                purpose=LlmTaskPurpose.INTENT,
                prompt_version=INTENT_PROMPT_VERSION,
                system_instruction=(
                    "Select requested output names only from the supplied allowlist. "
                    "Do not restate the user goal or generate battery values, identifiers, "
                    "tool inputs, or decisions."
                ),
                user_instruction=(
                    f"User goal: {request.user_goal}\n"
                    f"Allowed requested outputs: {list(request.requested_outputs)}"
                ),
                response_schema=_IntentDraft.model_json_schema(),
                max_output_tokens=500,
            )
        )
        if response.purpose is not LlmTaskPurpose.INTENT:
            raise AgentPlanPolicyError("LLM returned a mismatched intent response")
        draft = _IntentDraft.model_validate(response.payload)
        requested = tuple(
            item for item in draft.requested_outputs if item in request.requested_outputs
        )
        if not requested:
            raise AgentPlanPolicyError("LLM intent did not retain an allowed requested output")
        return AgentIntent(
            intent_id=str(self._uuid_factory()),
            project_id=request.project_id,
            goal=request.user_goal,
            dataset_ids=request.dataset_ids,
            requested_outputs=requested,
            created_at=created_at,
        )

    def _create_llm_plan(
        self,
        intent: AgentIntent,
        *,
        available_tools: frozenset[StandardToolName],
        created_at: datetime,
    ) -> AgentPlan:
        if self._gateway is None:
            raise AgentPlanPolicyError("LLM gateway is unavailable")
        response = self._gateway.complete_json(
            LlmStructuredRequest(
                purpose=LlmTaskPurpose.PLAN,
                prompt_version=PLAN_PROMPT_VERSION,
                system_instruction=(
                    "Create a dependency-ordered tool plan of at most 12 steps. "
                    "Use only the supplied role-tool pairs. Inputs must be references, never "
                    "battery values. Do not create a plan hash."
                ),
                user_instruction=self._plan_instruction(intent, available_tools),
                response_schema=_PlanDraft.model_json_schema(),
                max_output_tokens=2_000,
            )
        )
        if response.purpose is not LlmTaskPurpose.PLAN:
            raise AgentPlanPolicyError("LLM returned a mismatched plan response")
        draft = _PlanDraft.model_validate(response.payload)
        steps = self._enforce_plan_policy(
            draft.steps,
            intent=intent,
            available_tools=available_tools,
        )
        return AgentPlan.build(
            plan_version=SUPERVISOR_PLANNER_VERSION,
            intent_id=intent.intent_id,
            steps=steps,
            planning_mode=AgentPlanningMode.LLM,
            created_at=created_at,
        )

    def _fallback(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: frozenset[StandardToolName],
        created_at: datetime,
        warning: str,
    ) -> SupervisorPlanningResult:
        intent = AgentIntent(
            intent_id=str(self._uuid_factory()),
            project_id=request.project_id,
            goal=request.user_goal,
            dataset_ids=request.dataset_ids,
            requested_outputs=request.requested_outputs,
            created_at=created_at,
        )
        fixed_steps = self._fixed_steps(available_tools)
        if not fixed_steps:
            raise AgentPlanPolicyError("no safe fixed workflow can be built from available tools")
        steps = self._enforce_plan_policy(
            fixed_steps,
            intent=intent,
            available_tools=available_tools,
        )
        plan = AgentPlan.build(
            plan_version=SUPERVISOR_PLANNER_VERSION,
            intent_id=intent.intent_id,
            steps=steps,
            planning_mode=AgentPlanningMode.FIXED_FALLBACK,
            created_at=created_at,
        )
        return SupervisorPlanningResult(intent=intent, plan=plan, warnings=(warning,))

    @staticmethod
    def _enforce_plan_policy(
        steps: tuple[AgentPlanStep, ...],
        *,
        intent: AgentIntent,
        available_tools: frozenset[StandardToolName],
    ) -> tuple[AgentPlanStep, ...]:
        secured: list[AgentPlanStep] = []
        prior_steps: dict[str, StandardToolName] = {}
        for step in steps:
            try:
                tool_name = StandardToolName(step.tool_name)
            except ValueError as exc:
                raise AgentPlanPolicyError("LLM selected an unknown tool") from exc
            if tool_name not in available_tools:
                raise AgentPlanPolicyError("LLM selected a tool unavailable in this application")
            if tool_name not in ROLE_TOOL_ALLOWLIST[step.role]:
                raise AgentPlanPolicyError("LLM selected a tool outside the role allowlist")
            SupervisorPlanner._validate_planned_inputs(
                step,
                tool_name=tool_name,
                intent=intent,
                prior_steps=prior_steps,
            )
            payload = step.model_dump(mode="json")
            payload["requires_approval"] = (
                step.requires_approval or tool_name in FORMAL_APPROVAL_TOOLS
            )
            secured.append(AgentPlanStep.model_validate(payload))
            prior_steps[step.step_id] = tool_name
        return tuple(secured)

    @staticmethod
    def _validate_planned_inputs(
        step: AgentPlanStep,
        *,
        tool_name: StandardToolName,
        intent: AgentIntent,
        prior_steps: dict[str, StandardToolName],
    ) -> None:
        modes = PLANNABLE_INPUT_MODES.get(tool_name)
        if modes is None:
            raise AgentPlanPolicyError("LLM selected a tool without a safe plan compiler")
        if frozenset(step.input_references) not in modes:
            raise AgentPlanPolicyError("LLM tool inputs do not match a safe tool input mode")
        if set(step.depends_on) - set(prior_steps):
            raise AgentPlanPolicyError("LLM plan depends on an unknown or future step")
        for input_name, reference in step.input_references.items():
            if reference in ALLOWED_CONTEXT_REFERENCES.get(input_name, frozenset()):
                continue
            dataset_match = DATASET_REFERENCE.fullmatch(reference)
            if input_name == "record_batch_id" and dataset_match is not None:
                if int(dataset_match.group("index")) >= len(intent.dataset_ids):
                    raise AgentPlanPolicyError("LLM referenced a missing intent dataset")
                continue
            step_match = STEP_RESULT_REFERENCE.fullmatch(reference)
            if step_match is None:
                raise AgentPlanPolicyError("LLM tool inputs must contain trusted references only")
            referenced_step_id = step_match.group("step_id")
            if referenced_step_id not in step.depends_on:
                raise AgentPlanPolicyError(
                    "LLM step result references must be declared as dependencies"
                )
            source_tool = prior_steps.get(referenced_step_id)
            if source_tool not in EXPECTED_RESULT_SOURCE_TOOLS.get(
                input_name, frozenset()
            ):
                raise AgentPlanPolicyError(
                    "LLM step result reference has an incompatible source tool"
                )

    @staticmethod
    def _fixed_steps(
        available_tools: frozenset[StandardToolName],
    ) -> tuple[AgentPlanStep, ...]:
        project_tools = {
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            StandardToolName.GENERATE_AUDITED_REPORT,
        }
        if (
            StandardToolName.VALIDATE_BATTERY_DATA not in available_tools
            and project_tools <= available_tools
        ):
            return SupervisorPlanner._project_advanced_fixed_steps()
        templates = (
            (
                "validate",
                AgentRole.DATA_QUALITY,
                StandardToolName.VALIDATE_BATTERY_DATA,
                {
                    "records": "context.validation_records",
                    "data_version": "context.data_version",
                    "feature_version": "context.feature_version",
                    "provenance": "context.provenance",
                    "validated_at": "context.validated_at",
                },
                (),
                False,
            ),
            (
                "features",
                AgentRole.DATA_QUALITY,
                StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                {"record_batch_id": "intent.dataset_ids[0]"},
                (),
                False,
            ),
            (
                "predict",
                AgentRole.LIFETIME,
                StandardToolName.PREDICT_CYCLE_LIFE,
                {"upstream_result_id": "step.features.result_id"},
                ("features",),
                False,
            ),
            (
                "calibration",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {"calibration_cohort_id": "context.calibration_cohort_id"},
                (),
                False,
            ),
            (
                "interval",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {
                    "prediction_result_id": "step.predict.result_id",
                    "calibration_result_id": "step.calibration.result_id",
                },
                ("predict", "calibration"),
                False,
            ),
            (
                "decision",
                AgentRole.SUPERVISOR,
                StandardToolName.MAKE_BATCH_DECISION,
                {
                    "prediction_interval_result_id": "step.interval.result_id",
                    "calibration_result_id": "step.calibration.result_id",
                    "quality_result_id": "step.validate.result_id",
                    "policy_id": "context.decision_policy_id",
                },
                ("validate", "calibration", "interval"),
                True,
            ),
        )
        steps: list[AgentPlanStep] = []
        for step_id, role, tool_name, references, dependencies, approval in templates:
            if tool_name not in available_tools:
                break
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    role=role,
                    tool_name=tool_name.value,
                    depends_on=dependencies,
                    input_references=references,
                    requires_approval=approval,
                    failure_policy=(
                        AgentFailurePolicy.STOP
                        if step_id == "validate"
                        else AgentFailurePolicy.REPLAN
                    ),
                )
            )
        return tuple(steps)

    @staticmethod
    def _project_advanced_fixed_steps() -> tuple[AgentPlanStep, ...]:
        templates = (
            (
                "advanced-input",
                AgentRole.DATA_QUALITY,
                StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                {"record_batch_id": "intent.dataset_ids[0]"},
                (),
            ),
            (
                "rul-point",
                AgentRole.LIFETIME,
                StandardToolName.PREDICT_CYCLE_LIFE,
                {
                    "upstream_result_id": "step.advanced-input.result_id",
                    "route_role": "context.rul_point_route_role",
                },
                ("advanced-input",),
            ),
            (
                "rul-coverage",
                AgentRole.LIFETIME,
                StandardToolName.PREDICT_CYCLE_LIFE,
                {
                    "upstream_result_id": "step.advanced-input.result_id",
                    "route_role": "context.rul_coverage_route_role",
                },
                ("advanced-input",),
            ),
            (
                "soh",
                AgentRole.LIFETIME,
                StandardToolName.PREDICT_SOH_TRAJECTORY,
                {
                    "upstream_result_id": "step.advanced-input.result_id",
                    "route_role": "context.soh_route_role",
                },
                ("advanced-input",),
            ),
            (
                "rul-calibration",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {
                    "operation": "context.conformal_calibrate_operation",
                    "task": "context.rul_task",
                    "route_role": "context.rul_coverage_route_role",
                    "alpha": "context.conformal_alpha",
                    "calibration_sample_result_ids": (
                        "context.rul_calibration_sample_result_ids"
                    ),
                },
                (),
            ),
            (
                "rul-interval",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {
                    "operation": "context.conformal_issue_operation",
                    "task": "context.rul_task",
                    "route_role": "context.rul_coverage_route_role",
                    "prediction_result_id": "step.rul-coverage.result_id",
                    "calibration_result_id": "step.rul-calibration.result_id",
                },
                ("rul-coverage", "rul-calibration"),
            ),
            (
                "soh-calibration",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {
                    "operation": "context.conformal_calibrate_operation",
                    "task": "context.soh_task",
                    "route_role": "context.soh_route_role",
                    "alpha": "context.conformal_alpha",
                    "calibration_sample_result_ids": (
                        "context.soh_calibration_sample_result_ids"
                    ),
                },
                (),
            ),
            (
                "soh-band",
                AgentRole.LIFETIME,
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                {
                    "operation": "context.conformal_issue_operation",
                    "task": "context.soh_task",
                    "route_role": "context.soh_route_role",
                    "prediction_result_id": "step.soh.result_id",
                    "calibration_result_id": "step.soh-calibration.result_id",
                },
                ("soh", "soh-calibration"),
            ),
            (
                "report",
                AgentRole.SUPERVISOR,
                StandardToolName.GENERATE_AUDITED_REPORT,
                {
                    "rul_result_id": "step.rul-point.result_id",
                    "soh_result_id": "step.soh.result_id",
                    "rul_conformal_result_id": "step.rul-interval.result_id",
                    "soh_conformal_result_id": "step.soh-band.result_id",
                },
                ("rul-point", "soh", "rul-interval", "soh-band"),
            ),
        )
        return tuple(
            AgentPlanStep(
                step_id=step_id,
                role=role,
                tool_name=tool_name.value,
                depends_on=dependencies,
                input_references=references,
                failure_policy=AgentFailurePolicy.STOP,
            )
            for step_id, role, tool_name, references, dependencies in templates
        )

    @staticmethod
    def _plan_instruction(
        intent: AgentIntent,
        available_tools: frozenset[StandardToolName],
    ) -> str:
        allowed_pairs = [
            f"{role.value}:{tool.value}"
            for role, tools in ROLE_TOOL_ALLOWLIST.items()
            for tool in sorted(tools & available_tools, key=lambda item: item.value)
        ]
        dataset_aliases = [f"dataset_{index}" for index, _ in enumerate(intent.dataset_ids)]
        return (
            f"Goal: {intent.goal}\n"
            f"Dataset aliases: {dataset_aliases}\n"
            f"Requested outputs: {list(intent.requested_outputs)}\n"
            f"Allowed role-tool pairs: {allowed_pairs}"
        )

    @staticmethod
    def _assert_llm_outbound_safe(user_goal: str) -> None:
        if any(pattern.search(user_goal) for pattern in SENSITIVE_OUTBOUND_PATTERNS):
            raise AgentPlanPolicyError(
                "user goal contains content prohibited from external LLM transmission"
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Supervisor clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


class FixedAdvancedSupervisorPlanner:
    """Create the product-owned fixed nine-step workflow without an LLM."""

    def __init__(
        self,
        *,
        clock: Clock = _utc_now,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        self._clock = clock
        self._uuid_factory = uuid_factory

    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: Collection[StandardToolName],
    ) -> SupervisorPlanningResult:
        validated = SupervisorPlanningRequest.model_validate(
            request.model_dump(mode="json")
        )
        if len(validated.dataset_ids) != 1:
            raise AgentPlanPolicyError(
                "fixed advanced planning requires exactly one record batch"
            )
        if validated.requested_outputs != (ADVANCED_SINGLE_CELL_OUTPUT,):
            raise AgentPlanPolicyError(
                "fixed advanced planning requires the advanced single-cell output"
            )
        available = frozenset(available_tools)
        if not available >= FIXED_ADVANCED_TOOLS:
            raise AgentPlanPolicyError(
                "fixed advanced planning requires the complete advanced tool set"
            )
        created_at = self._now()
        intent = AgentIntent(
            intent_id=str(self._uuid_factory()),
            project_id=validated.project_id,
            goal=validated.user_goal,
            dataset_ids=validated.dataset_ids,
            requested_outputs=validated.requested_outputs,
            created_at=created_at,
        )
        steps = SupervisorPlanner._enforce_plan_policy(
            SupervisorPlanner._project_advanced_fixed_steps(),
            intent=intent,
            available_tools=available,
        )
        if tuple(step.step_id for step in steps) != FIXED_ADVANCED_STEP_IDS:
            raise AgentPlanPolicyError("fixed advanced workflow identity changed")
        return SupervisorPlanningResult(
            intent=intent,
            plan=AgentPlan.build(
                plan_version=SUPERVISOR_PLANNER_VERSION,
                intent_id=intent.intent_id,
                steps=steps,
                planning_mode=AgentPlanningMode.FIXED_FALLBACK,
                created_at=created_at,
            ),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Supervisor clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
