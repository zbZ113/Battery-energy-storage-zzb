"""Validation-first fixed tool workflow for Feishu and Aily adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.core import ToolResult
from quanxin_life.tools import StandardToolName


class FeishuWorkflowRejected(RuntimeError):
    """Machine-readable refusal with optional already-audited validation evidence."""

    def __init__(
        self,
        reason_code: str,
        *,
        validation_result: ToolResult | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.validation_result = validation_result


class FeishuAnalysisTask(StrEnum):
    PREDICT_CYCLE_LIFE = "predict_cycle_life"
    PREDICT_SOH_TRAJECTORY = "predict_soh_trajectory"
    COMPARE_OPERATION_SCENARIOS = "compare_operation_scenarios"
    PROJECT_STORAGE_LIFETIME = "project_storage_lifetime"
    INGEST_OBSERVED_SOH = "ingest_observed_soh"
    UPDATE_TRAJECTORY = "update_trajectory"


class FeishuRouteAuthorizer(Protocol):
    def authorize(
        self,
        *,
        tool_name: StandardToolName,
        validation_result: ToolResult,
    ) -> None: ...


class FeishuAnalysisInputBinder(Protocol):
    def bind_analysis_input(
        self,
        *,
        task: FeishuAnalysisTask,
        validation_result: ToolResult,
        validation_input: Mapping[str, object],
        requested_analysis_input: Mapping[str, object],
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class FeishuWorkflowOutcome:
    validation_result: ToolResult
    analysis_result: ToolResult


_TASK_TO_TOOL = {
    FeishuAnalysisTask.PREDICT_CYCLE_LIFE: StandardToolName.PREDICT_CYCLE_LIFE,
    FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: StandardToolName.PREDICT_SOH_TRAJECTORY,
    FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: (
        StandardToolName.COMPARE_OPERATION_SCENARIOS
    ),
    FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME: (
        StandardToolName.PROJECT_STORAGE_LIFETIME
    ),
    FeishuAnalysisTask.INGEST_OBSERVED_SOH: StandardToolName.INGEST_NEWLY_OBSERVED_SOH,
    FeishuAnalysisTask.UPDATE_TRAJECTORY: StandardToolName.UPDATE_CELL_PARAMETERS,
}
_MODEL_TOOLS = frozenset(
    {StandardToolName.PREDICT_CYCLE_LIFE, StandardToolName.PREDICT_SOH_TRAJECTORY}
)


class FeishuAnalysisWorkflow:
    """Invoke only audited tools, with data validation as a hard prerequisite."""

    def __init__(
        self,
        service: ToolInvocationService,
        *,
        route_authorizer: FeishuRouteAuthorizer,
        input_binder: FeishuAnalysisInputBinder,
    ) -> None:
        if service.audit_ledger is None:
            raise ValueError("Feishu workflow requires an audit ledger")
        if not callable(getattr(input_binder, "bind_analysis_input", None)):
            raise ValueError("Feishu workflow requires an analysis input binder")
        self._service = service
        self._route_authorizer = route_authorizer
        self._input_binder = input_binder

    def validate(
        self,
        *,
        validation_input: Mapping[str, object],
    ) -> ToolResult:
        validation_result = self._service.invoke_for_agent(
            ToolInvocation(
                tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
                input_value=dict(validation_input),
            ),
            allowed_tool_names={StandardToolName.VALIDATE_BATTERY_DATA},
        )
        blocked = validation_result.values.get("blocked")
        if not isinstance(blocked, bool):
            raise FeishuWorkflowRejected("DATA_VALIDATION_RESULT_INVALID")
        if blocked:
            raise FeishuWorkflowRejected(
                "DATA_VALIDATION_BLOCKED",
                validation_result=validation_result,
            )
        return validation_result

    def run(
        self,
        *,
        task: FeishuAnalysisTask,
        validation_input: Mapping[str, object],
        analysis_input: Mapping[str, object],
        validation_result: ToolResult | None = None,
        before_analysis: Callable[[], None] | None = None,
    ) -> FeishuWorkflowOutcome:
        try:
            tool_name = _TASK_TO_TOOL[task]
        except KeyError as exc:  # pragma: no cover - exhaustive enum map
            raise FeishuWorkflowRejected("ANALYSIS_TASK_NOT_SUPPORTED") from exc
        validation_result = self.validate_result(
            validation_input=validation_input,
            validation_result=validation_result,
        )
        if tool_name in _MODEL_TOOLS:
            try:
                self._route_authorizer.authorize(
                    tool_name=tool_name,
                    validation_result=validation_result,
                )
            except FeishuWorkflowRejected as exc:
                raise FeishuWorkflowRejected(
                    exc.reason_code,
                    validation_result=validation_result,
                ) from exc
        try:
            bound_analysis_input = self._input_binder.bind_analysis_input(
                task=task,
                validation_result=validation_result,
                validation_input=dict(validation_input),
                requested_analysis_input=dict(analysis_input),
            )
        except FeishuWorkflowRejected:
            raise
        except Exception as exc:
            raise FeishuWorkflowRejected("ANALYSIS_INPUT_BINDING_FAILED") from exc
        if not isinstance(bound_analysis_input, Mapping):
            raise FeishuWorkflowRejected("ANALYSIS_INPUT_BINDING_INVALID")
        if before_analysis is not None:
            before_analysis()
        analysis_result = self._service.invoke_for_agent(
            ToolInvocation(tool_name=tool_name, input_value=dict(bound_analysis_input)),
            allowed_tool_names={tool_name},
        )
        return FeishuWorkflowOutcome(
            validation_result=validation_result,
            analysis_result=analysis_result,
        )

    def validate_result(
        self,
        *,
        validation_input: Mapping[str, object],
        validation_result: ToolResult | None,
    ) -> ToolResult:
        if validation_result is None:
            return self.validate(validation_input=validation_input)
        ledger = self._service.audit_ledger
        if ledger is None:  # pragma: no cover - guarded by __init__
            raise FeishuWorkflowRejected("DATA_VALIDATION_RESULT_INVALID")
        try:
            registered = ledger.resolve_registered_result(validation_result.result_id)
            expected_input_hash = self._service.registry.canonical_input_hash(
                StandardToolName.VALIDATE_BATTERY_DATA,
                validation_input,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise FeishuWorkflowRejected("DATA_VALIDATION_RESULT_INVALID") from exc
        blocked = registered.values.get("blocked")
        if (
            registered != validation_result
            or registered.tool_name != StandardToolName.VALIDATE_BATTERY_DATA.value
            or registered.input_hash != expected_input_hash
            or not isinstance(blocked, bool)
        ):
            raise FeishuWorkflowRejected("DATA_VALIDATION_RESULT_INVALID")
        if blocked:
            raise FeishuWorkflowRejected(
                "DATA_VALIDATION_BLOCKED",
                validation_result=registered,
            )
        return registered


__all__ = [
    "FeishuAnalysisInputBinder",
    "FeishuAnalysisTask",
    "FeishuAnalysisWorkflow",
    "FeishuRouteAuthorizer",
    "FeishuWorkflowOutcome",
    "FeishuWorkflowRejected",
]
