"""Integration checks for the explicit competition tool assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from quanxin_life.application import (
    CompetitionToolDependencies,
    create_competition_tool_invocation_service,
    create_competition_tool_registry,
)
from quanxin_life.tools import (
    StandardToolName,
    ToolExecutionScope,
    ToolRegistry,
    create_available_tool_registry,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_project_report import (
    ADVANCED_CELL_REPORT_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
)


def _stub_dependencies() -> CompetitionToolDependencies:
    """Return inert protocol stubs; schema discovery must not execute them."""

    stub = cast(Any, object())
    return CompetitionToolDependencies(
        audit_ledger=stub,
        early_cycle_batch_resolver=stub,
        cycle_life_predictor=stub,
        hybrid_degradation_predictor=stub,
        normalized_calibration_cohort_resolver=stub,
        prediction_difficulty_scale_resolver=None,
        adaptation_cohort_resolver=stub,
        measurement_resolver=stub,
        individual_trajectory_calibrator=None,
        physics_validator=stub,
        experiment_recommendation_context_resolver=stub,
        batch_decision_policy_resolver=stub,
        knowledge_scope_resolver=stub,
        battery_evidence_backend=stub,
    )


def test_competition_registry_discovers_every_standard_tool_once() -> None:
    registry = create_competition_tool_registry(_stub_dependencies())

    schemas = registry.list_schemas()
    names = tuple(schema.tool_name for schema in schemas)

    assert len(schemas) == len(StandardToolName) == 15
    assert len(set(names)) == len(names)
    assert set(names) == set(StandardToolName)
    assert all(schema.input_schema for schema in schemas)


def test_competition_service_wraps_the_registry_from_the_factory(
    monkeypatch: MonkeyPatch,
) -> None:
    sentinel = ToolRegistry()
    dependencies = _stub_dependencies()

    monkeypatch.setattr(
        "quanxin_life.application.assembly.create_competition_tool_registry",
        lambda supplied: sentinel if supplied is dependencies else ToolRegistry(),
    )

    service = create_competition_tool_invocation_service(dependencies)

    assert service.registry is sentinel
    assert service.audit_ledger is dependencies.audit_ledger


def test_available_registry_remains_the_three_tool_core() -> None:
    names = {schema.tool_name for schema in create_available_tool_registry().list_schemas()}

    assert names == {
        StandardToolName.VALIDATE_BATTERY_DATA,
        StandardToolName.AUDIT_DATASET_SPLIT,
        StandardToolName.CHECK_OPERATING_CONDITION,
    }


def test_project_prediction_assembly_exposes_only_project_scoped_numeric_tools() -> None:
    import quanxin_life.application.assembly as module

    stub = cast(Any, object())
    dependencies = module.ProjectPredictionToolDependencies(
        project_audit_ledger=stub,
        project_context_validator=stub,
        agent_run_invocation_resolver=stub,
        advanced_input_batch_resolver=stub,
        advanced_rul_inference_service=stub,
        advanced_soh_inference_service=stub,
    )

    service = module.create_project_prediction_tool_invocation_service(dependencies)

    assert service.registry.list_schemas() == ()
    project_schemas = service.registry.list_schemas(
        execution_scope=ToolExecutionScope.PROJECT
    )
    assert {item.tool_name for item in project_schemas} == {
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        StandardToolName.GENERATE_AUDITED_REPORT,
    }
    schema_by_name = {item.tool_name: item for item in project_schemas}
    assert (
        schema_by_name[StandardToolName.PREDICT_CYCLE_LIFE].tool_version
        == ADVANCED_RUL_PREDICTION_TOOL_VERSION
    )
    assert (
        schema_by_name[StandardToolName.PREDICT_SOH_TRAJECTORY].tool_version
        == ADVANCED_SOH_PREDICTION_TOOL_VERSION
    )
    assert (
        schema_by_name[StandardToolName.CALIBRATE_PREDICTION_INTERVAL].tool_version
        == ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION
    )
    assert (
        schema_by_name[StandardToolName.GENERATE_AUDITED_REPORT].tool_version
        == ADVANCED_CELL_REPORT_TOOL_VERSION
    )
    assert service.project_audit_ledger is dependencies.project_audit_ledger
    assert (
        service.agent_run_invocation_validator
        is dependencies.agent_run_invocation_resolver
    )


def test_advanced_calibration_assembly_wires_service_worker_api_and_agent(
    monkeypatch: MonkeyPatch,
) -> None:
    import quanxin_life.api.advanced_calibration as api_module
    import quanxin_life.application.assembly as module

    stub = cast(Any, object())
    context_service = cast(
        Any,
        SimpleNamespace(revalidate=lambda context: context),
    )
    adapter = cast(Any, object())
    captured: dict[str, object] = {}

    def create_adapter(
        service: object,
        *,
        queue: object,
        context_service: object,
        auth_adapter: object,
    ) -> object:
        captured.update(
            {
                "service": service,
                "queue": queue,
                "context_service": context_service,
                "auth_adapter": auth_adapter,
            }
        )
        return adapter

    monkeypatch.setattr(
        api_module,
        "create_advanced_calibration_http_adapter",
        create_adapter,
    )
    dependencies = module.AdvancedCalibrationAssemblyDependencies(
        session_factory=stub,
        context_service=context_service,
        evidence_resolver=stub,
        runtime_resolver=stub,
        cell_input_resolver=stub,
        project_materializer=stub,
        queue=stub,
        auth_adapter=stub,
        agent_context_delegate=stub,
        target_record_batch_resolver=stub,
    )

    components = module.create_advanced_calibration_components(dependencies)

    assert components.http_adapter is adapter
    assert captured == {
        "service": components.materialization_service,
        "queue": dependencies.queue,
        "context_service": context_service,
        "auth_adapter": dependencies.auth_adapter,
    }
    assert (
        type(components.materialization_service).__name__
        == "AdvancedCalibrationMaterializationService"
    )
    assert (
        type(components.worker).__name__
        == "AdvancedCalibrationMaterializationWorker"
    )
    assert (
        type(components.agent_context_resolver).__name__
        == "AdvancedAgentExecutionContextResolver"
    )


def test_advanced_calibration_assembly_rejects_missing_dependencies() -> None:
    import quanxin_life.application.assembly as module

    stub = cast(Any, object())

    with pytest.raises(TypeError, match="must not be None"):
        module.AdvancedCalibrationAssemblyDependencies(
            session_factory=stub,
            context_service=stub,
            evidence_resolver=stub,
            runtime_resolver=stub,
            cell_input_resolver=None,
            project_materializer=stub,
            queue=stub,
            auth_adapter=stub,
            agent_context_delegate=stub,
            target_record_batch_resolver=stub,
        )
