from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind
from quanxin_life.core.schemas import JsonMapping
from quanxin_life.scenarios import (
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSegment,
    ScenarioStateReference,
    VerifiedScenarioContext,
)
from quanxin_life.tools import StandardToolName, ToolRegistry
from quanxin_life.tools.blast_scenarios import (
    SCENARIO_FEATURE_VERSION,
    CompareOperationScenariosToolInput,
    ProjectStorageLifetimeToolInput,
    register_compare_operation_scenarios_tool,
    register_project_storage_lifetime_tool,
)

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)


def _cell(*, chemistry: str = "LFP/graphite") -> ScenarioCellDescriptor:
    return ScenarioCellDescriptor(
        chemistry=chemistry,
        nominal_capacity_ah=250.0,
        cell_format="prismatic",
    )


def _scenario(
    *,
    scenario_id: str,
    temperature_c: float = 25.0,
    years: int = 1,
) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=years,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="all-years",
                start_year=0,
                end_year=years,
                temperature_c=temperature_c,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.075,
                soc_upper_bound=0.925,
                dod=0.85,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


class _ContextResolver:
    def __init__(self, context: VerifiedScenarioContext) -> None:
        self._context = context

    def resolve_scenario_context(self, scenario_context_id: str) -> VerifiedScenarioContext:
        if scenario_context_id != self._context.scenario_context_id:
            raise ValueError("scenario context was not found")
        return self._context


def _context(*, chemistry: str = "LFP/graphite") -> VerifiedScenarioContext:
    return VerifiedScenarioContext(
        scenario_context_id="ctx-250ah-reference",
        cell=_cell(chemistry=chemistry),
        trusted_reference_use=False,
        data_version="scenario-context-data-v1",
        provenance=(
            ProvenanceRecord(
                source_id="verified-scenario-context",
                source_kind=SourceKind.OBSERVED,
                uri="memory://verified-scenario-context",
                sha256="a" * 64,
                description="Test-only verified cell metadata context.",
                created_at=NOW,
            ),
        ),
    )


def _service(
    *,
    context: VerifiedScenarioContext | None = None,
    allow_candidate_execution: bool,
) -> tuple[ToolInvocationService, AuditLedger]:
    ledger = AuditLedger()
    registry = ToolRegistry()
    resolver = _ContextResolver(context or _context())
    register_compare_operation_scenarios_tool(
        registry,
        context_resolver=resolver,
        allow_candidate_execution=allow_candidate_execution,
        clock=lambda: NOW,
    )
    register_project_storage_lifetime_tool(
        registry,
        context_resolver=resolver,
        allow_candidate_execution=allow_candidate_execution,
        clock=lambda: NOW,
    )
    return ToolInvocationService(registry=registry, audit_ledger=ledger), ledger


def test_compare_tool_runs_two_scenarios_and_registers_audited_result() -> None:
    service, ledger = _service(allow_candidate_execution=True)
    input_value = CompareOperationScenariosToolInput(
        run_id=str(uuid4()),
        scenario_context_id="ctx-250ah-reference",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_cell(),
        baseline=_scenario(scenario_id="baseline"),
        comparisons=(_scenario(scenario_id="warmer", temperature_c=30.0),),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.COMPARE_OPERATION_SCENARIOS,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    artifact = cast(JsonMapping, result.values["artifact"])
    baseline = cast(JsonMapping, artifact["baseline"])
    comparisons = cast(list[JsonMapping], artifact["comparisons"])
    assert artifact["status"] == "COMPLETED"
    assert result.feature_version == SCENARIO_FEATURE_VERSION
    assert len(cast(list[float], baseline["natural_years"])) == 13
    assert len(comparisons) == 1
    assert baseline["soh"] != comparisons[0]["soh"]
    assert baseline["final_natural_year"] == cast(list[float], baseline["natural_years"])[-1]
    assert baseline["final_equivalent_full_cycles"] == cast(
        list[float], baseline["equivalent_full_cycles"]
    )[-1]
    assert baseline["final_soh"] == cast(list[float], baseline["soh"])[-1]
    assert baseline["operating_segments"] == [
        input_value.baseline.segments[0].model_dump(mode="json")
    ]
    assert comparisons[0]["operating_segments"] == [
        input_value.comparisons[0].segments[0].model_dump(mode="json")
    ]
    assert artifact["run_id"] == input_value.run_id
    assert artifact["result_id"] == result.result_id
    assert "model_class" not in artifact
    assert result.uncertainty == {
        "kind": "DETERMINISTIC_SCENARIO",
        "prediction_interval_included": False,
        "sensitivity_envelope_included": False,
    }
    assert ledger.resolve_registered_result(result.result_id) == result


def test_compare_tool_marks_reviewed_capacity_and_format_reference_mismatch() -> None:
    reference_cell = ScenarioCellDescriptor(
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        cell_format="cylindrical",
    )
    context = VerifiedScenarioContext(
        scenario_context_id="ctx-reviewed-reference-use",
        cell=reference_cell,
        trusted_reference_use=True,
        data_version="scenario-context-data-v1",
        provenance=_context().provenance,
    )
    service, _ = _service(
        context=context,
        allow_candidate_execution=True,
    )
    input_value = CompareOperationScenariosToolInput(
        run_id=str(uuid4()),
        scenario_context_id=context.scenario_context_id,
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=reference_cell,
        baseline=_scenario(scenario_id="baseline"),
        comparisons=(_scenario(scenario_id="comparison"),),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.COMPARE_OPERATION_SCENARIOS,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    assert result.values["artifact"]["status"] == "COMPLETED"
    assert "CAPACITY_REFERENCE_MISMATCH" in result.warnings
    assert "CELL_FORMAT_REFERENCE_MISMATCH" in result.warnings
    assert "REFERENCE_USE_ONLY" in result.warnings


def test_candidate_route_is_rejected_by_default_activation_gate() -> None:
    service, _ = _service(allow_candidate_execution=False)
    input_value = CompareOperationScenariosToolInput(
        run_id=str(uuid4()),
        scenario_context_id="ctx-250ah-reference",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_cell(),
        baseline=_scenario(scenario_id="baseline"),
        comparisons=(_scenario(scenario_id="comparison"),),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.COMPARE_OPERATION_SCENARIOS,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    artifact = cast(JsonMapping, result.values["artifact"])
    assert artifact["status"] == "REJECTED"
    assert artifact["rejection_reasons"] == ["ROUTE_NOT_ACTIVATED"]
    assert "baseline" not in artifact


def test_tool_rejects_non_lfp_trusted_context_without_running_model() -> None:
    service, _ = _service(
        context=_context(chemistry="NMC/graphite"),
        allow_candidate_execution=True,
    )
    input_value = ProjectStorageLifetimeToolInput(
        run_id=str(uuid4()),
        scenario_context_id="ctx-250ah-reference",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_cell(chemistry="NMC/graphite"),
        scenario=_scenario(scenario_id="unsupported"),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.PROJECT_STORAGE_LIFETIME,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    artifact = cast(JsonMapping, result.values["artifact"])
    assert artifact["status"] == "REJECTED"
    assert artifact["rejection_reasons"] == ["CHEMISTRY_NOT_SUPPORTED"]
    assert "projection" not in artifact


def test_lifetime_tool_rejects_unvalidated_current_state_initialization() -> None:
    service, _ = _service(allow_candidate_execution=True)
    input_value = ProjectStorageLifetimeToolInput(
        run_id=str(uuid4()),
        scenario_context_id="ctx-250ah-reference",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_cell(),
        scenario=_scenario(scenario_id="stateful"),
        current_state_reference=ScenarioStateReference(
            result_id=str(uuid4()),
            json_path="values.artifact.current_soh",
        ),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.PROJECT_STORAGE_LIFETIME,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    artifact = cast(JsonMapping, result.values["artifact"])
    assert artifact["status"] == "REJECTED"
    assert artifact["rejection_reasons"] == [
        "UNSUPPORTED_BLAST_STATE_INITIALIZATION"
    ]


def test_lifetime_tool_outputs_only_horizon_milestones_and_trace_ids() -> None:
    service, _ = _service(allow_candidate_execution=True)
    input_value = ProjectStorageLifetimeToolInput(
        run_id=str(uuid4()),
        scenario_context_id="ctx-250ah-reference",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_cell(),
        scenario=_scenario(scenario_id="twenty-years", years=20),
    )

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.PROJECT_STORAGE_LIFETIME,
            input_value=input_value.model_dump(mode="json"),
        )
    )

    artifact = cast(JsonMapping, result.values["artifact"])
    projection = cast(JsonMapping, artifact["projection"])
    assert artifact["status"] == "COMPLETED"
    assert set(cast(dict[str, float], projection["milestone_soh"])) == {"15", "20"}
    assert artifact["run_id"] == input_value.run_id
    assert artifact["result_id"] == result.result_id
    assert artifact["evidence_level"] == "PHYSICS_REFERENCE"
    assert projection["final_natural_year"] == 20.0
    assert projection["final_soh"] == cast(list[float], projection["soh"])[-1]
    assert projection["operating_segments"] == [
        input_value.scenario.segments[0].model_dump(mode="json")
    ]
