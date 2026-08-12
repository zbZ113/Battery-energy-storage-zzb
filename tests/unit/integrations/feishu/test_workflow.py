from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
    FeishuWorkflowRejected,
)
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_TOOL_VERSION,
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="uploaded-csv",
            source_kind=SourceKind.OBSERVED,
            uri="feishu://message/om_source/resource/file_source",
            sha256="a" * 64,
            description="Verified Feishu CSV attachment",
            created_at=NOW,
        ),
    )


class _PredictionInput(ValidateBatteryDataToolInput):
    """Test-only input type used to distinguish the prediction registry entry."""


class _ScenarioWorkflowInput(ContractModel):
    scenario_context_id: str


class _RecordingRouteAuthorizer:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.calls: list[tuple[StandardToolName, str]] = []

    def authorize(
        self,
        *,
        tool_name: StandardToolName,
        validation_result: ToolResult,
    ) -> None:
        self.calls.append((tool_name, validation_result.result_id))
        if self.reject:
            raise FeishuWorkflowRejected("MODEL_ROUTE_NOT_ACTIVATED")


class _SameEvidenceInputBinder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def bind_analysis_input(
        self,
        *,
        task: FeishuAnalysisTask,
        validation_result: ToolResult,
        validation_input: dict[str, object],
        requested_analysis_input: dict[str, object],
    ) -> dict[str, object]:
        del task
        self.calls.append(validation_result.result_id)
        evidence_fields = ("data_version", "feature_version", "provenance")
        if any(
            validation_input.get(field) != requested_analysis_input.get(field)
            for field in evidence_fields
        ):
            raise FeishuWorkflowRejected("ANALYSIS_INPUT_NOT_BOUND_TO_VALIDATION")
        return dict(requested_analysis_input)


class _ScenarioInputBinder:
    def bind_analysis_input(
        self,
        *,
        requested_analysis_input: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        return dict(requested_analysis_input)


class _BrokenInputBinder:
    def bind_analysis_input(self, **_: object) -> dict[str, object]:
        raise RuntimeError("untrusted binder detail")


class _InvalidInputBinder:
    def bind_analysis_input(self, **_: object) -> object:
        return ("not", "a", "mapping")


def _service(*, prediction_calls: list[str]) -> ToolInvocationService:
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)

    def predict(input_value: _PredictionInput) -> ToolResult:
        prediction_calls.append(input_value.data_version)
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            tool_version="test-only-prediction-tool-v1",
            model_version="test-only-route-v1",
            data_version=input_value.data_version,
            feature_version=input_value.feature_version,
            input_hash=sha256_canonical(input_value.model_dump(mode="json")),
            values={"source": "tool-generated-test-result"},
            uncertainty=None,
            warnings=["TEST_ONLY_RESULT_NOT_FOR_BUSINESS_USE"],
            provenance=list(input_value.provenance),
            created_at=input_value.validated_at,
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="test-only-prediction-tool-v1",
            input_model=_PredictionInput,
            executor=predict,
        )
    )
    return ToolInvocationService(registry=registry, audit_ledger=AuditLedger())


def _input(*, records: tuple[object, ...] = ()) -> dict[str, object]:
    return {
        "records": records,
        "data_version": "uploaded-data-v1",
        "feature_version": "raw-cycle-v1",
        "provenance": [item.model_dump(mode="json") for item in _provenance()],
        "validated_at": NOW.isoformat(),
    }


def test_validation_block_prevents_prediction_invocation() -> None:
    prediction_calls: list[str] = []
    authorizer = _RecordingRouteAuthorizer()
    workflow = FeishuAnalysisWorkflow(
        _service(prediction_calls=prediction_calls),
        route_authorizer=authorizer,
        input_binder=_SameEvidenceInputBinder(),
    )

    with pytest.raises(
        FeishuWorkflowRejected, match="DATA_VALIDATION_BLOCKED"
    ) as captured:
        workflow.run(
            task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            validation_input=_input(),
            analysis_input=_input(),
        )

    assert prediction_calls == []
    assert authorizer.calls == []
    assert captured.value.validation_result is not None
    assert captured.value.validation_result.tool_name == "validate_battery_data"


def test_inactive_model_route_is_rejected_after_validation_and_before_prediction() -> None:
    prediction_calls: list[str] = []
    service = _service(prediction_calls=prediction_calls)
    authorizer = _RecordingRouteAuthorizer(reject=True)
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=authorizer,
        input_binder=_SameEvidenceInputBinder(),
    )
    valid_input = _input(
        records=(
            {
                "dataset_id": "source",
                "cell_id": "cell",
                "cycle_index": 0,
                "sample_index": 0,
                "time_s": 0,
                "voltage_v": 3,
                "current_a": 0,
                "temperature_c": None,
                "charge_capacity_ah": None,
                "discharge_capacity_ah": None,
                "internal_resistance_ohm": None,
                "diagnostic": False,
                "valid": True,
            },
        )
    )

    with pytest.raises(FeishuWorkflowRejected, match="MODEL_ROUTE_NOT_ACTIVATED"):
        workflow.run(
            task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            validation_input=valid_input,
            analysis_input=valid_input,
        )

    assert prediction_calls == []
    assert len(authorizer.calls) == 1


def test_active_route_executes_only_allowlisted_tool_and_registers_both_results() -> None:
    prediction_calls: list[str] = []
    service = _service(prediction_calls=prediction_calls)
    authorizer = _RecordingRouteAuthorizer()
    binder = _SameEvidenceInputBinder()
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=authorizer,
        input_binder=binder,
    )
    valid_input = _input(
        records=(
            {
                "dataset_id": "source",
                "cell_id": "cell",
                "cycle_index": 0,
                "sample_index": 0,
                "time_s": 0,
                "voltage_v": 3,
                "current_a": 0,
                "temperature_c": None,
                "charge_capacity_ah": None,
                "discharge_capacity_ah": None,
                "internal_resistance_ohm": None,
                "diagnostic": False,
                "valid": True,
            },
        )
    )

    outcome = workflow.run(
        task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        validation_input=valid_input,
        analysis_input=valid_input,
    )

    assert prediction_calls == ["uploaded-data-v1"]
    assert binder.calls == [outcome.validation_result.result_id]
    assert outcome.validation_result.tool_version == DATA_QUALITY_TOOL_VERSION
    assert outcome.analysis_result.tool_name == "predict_cycle_life"
    assert service.audit_ledger is not None
    assert (
        service.audit_ledger.resolve_registered_result(
            outcome.validation_result.result_id
        )
        == outcome.validation_result
    )
    assert (
        service.audit_ledger.resolve_registered_result(outcome.analysis_result.result_id)
        == outcome.analysis_result
    )


@pytest.mark.parametrize(
    ("task", "tool_name"),
    (
        (
            FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
            StandardToolName.COMPARE_OPERATION_SCENARIOS,
        ),
        (
            FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
            StandardToolName.PROJECT_STORAGE_LIFETIME,
        ),
    ),
)
def test_scenario_tasks_run_only_their_registered_allowlisted_tool(
    task: FeishuAnalysisTask,
    tool_name: StandardToolName,
) -> None:
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)

    def execute_scenario(input_value: _ScenarioWorkflowInput) -> ToolResult:
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=tool_name.value,
            tool_version=f"{tool_name.value}-test-v1",
            model_version="blast-candidate-test-v1",
            data_version="uploaded-data-v1",
            feature_version="operation-scenario-contract-v1",
            input_hash=sha256_canonical(input_value.model_dump(mode="json")),
            values={"scenario_context_id": input_value.scenario_context_id},
            warnings=["TEST_ONLY_SCENARIO_RESULT"],
            provenance=list(_provenance()),
            created_at=NOW,
        )

    registry.register(
        ToolDefinition(
            tool_name=tool_name,
            tool_version=f"{tool_name.value}-test-v1",
            input_model=_ScenarioWorkflowInput,
            executor=execute_scenario,
        )
    )
    authorizer = _RecordingRouteAuthorizer()
    workflow = FeishuAnalysisWorkflow(
        ToolInvocationService(registry=registry, audit_ledger=AuditLedger()),
        route_authorizer=authorizer,
        input_binder=_ScenarioInputBinder(),
    )
    valid_input = _input(
        records=(
            {
                "dataset_id": "source",
                "cell_id": "cell",
                "cycle_index": 0,
                "sample_index": 0,
                "time_s": 0,
                "voltage_v": 3,
                "current_a": 0,
                "temperature_c": None,
                "charge_capacity_ah": None,
                "discharge_capacity_ah": None,
                "internal_resistance_ohm": None,
                "diagnostic": False,
                "valid": True,
            },
        )
    )

    outcome = workflow.run(
        task=task,
        validation_input=valid_input,
        analysis_input={"scenario_context_id": "ctx-workflow"},
    )

    assert outcome.analysis_result.tool_name == tool_name.value
    assert authorizer.calls == []


def test_workflow_requires_a_persistent_audit_boundary() -> None:
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)

    with pytest.raises(ValueError, match="audit ledger"):
        FeishuAnalysisWorkflow(
            ToolInvocationService(registry=registry),
            route_authorizer=_RecordingRouteAuthorizer(),
            input_binder=_SameEvidenceInputBinder(),
        )


def test_analysis_input_must_be_bound_to_the_validated_evidence() -> None:
    prediction_calls: list[str] = []
    valid_input = _input(
        records=(
            {
                "dataset_id": "source",
                "cell_id": "cell",
                "cycle_index": 0,
                "sample_index": 0,
                "time_s": 0,
                "voltage_v": 3,
                "current_a": 0,
                "temperature_c": None,
                "charge_capacity_ah": None,
                "discharge_capacity_ah": None,
                "internal_resistance_ohm": None,
                "diagnostic": False,
                "valid": True,
            },
        )
    )
    mismatched = dict(valid_input)
    mismatched["data_version"] = "different-data-version"
    workflow = FeishuAnalysisWorkflow(
        _service(prediction_calls=prediction_calls),
        route_authorizer=_RecordingRouteAuthorizer(),
        input_binder=_SameEvidenceInputBinder(),
    )

    with pytest.raises(
        FeishuWorkflowRejected, match="ANALYSIS_INPUT_NOT_BOUND_TO_VALIDATION"
    ):
        workflow.run(
            task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            validation_input=valid_input,
            analysis_input=mismatched,
        )

    assert prediction_calls == []


@pytest.mark.parametrize(
    ("binder", "reason"),
    [
        (_BrokenInputBinder(), "ANALYSIS_INPUT_BINDING_FAILED"),
        (_InvalidInputBinder(), "ANALYSIS_INPUT_BINDING_INVALID"),
    ],
)
def test_analysis_input_binder_failures_are_closed(
    binder: object,
    reason: str,
) -> None:
    prediction_calls: list[str] = []
    valid_input = _input(
        records=(
            {
                "dataset_id": "source",
                "cell_id": "cell",
                "cycle_index": 0,
                "sample_index": 0,
                "time_s": 0,
                "voltage_v": 3,
                "current_a": 0,
                "temperature_c": None,
                "charge_capacity_ah": None,
                "discharge_capacity_ah": None,
                "internal_resistance_ohm": None,
                "diagnostic": False,
                "valid": True,
            },
        )
    )
    workflow = FeishuAnalysisWorkflow(
        _service(prediction_calls=prediction_calls),
        route_authorizer=_RecordingRouteAuthorizer(),
        input_binder=binder,  # type: ignore[arg-type]
    )

    with pytest.raises(FeishuWorkflowRejected, match=reason):
        workflow.run(
            task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            validation_input=valid_input,
            analysis_input=valid_input,
        )

    assert prediction_calls == []
