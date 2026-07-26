"""Integration checks for the explicit competition tool assembly."""

from __future__ import annotations

from typing import Any, cast

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
        cycle_life_predictor=stub,
        hybrid_degradation_predictor=stub,
        normalized_calibration_cohort_resolver=stub,
        prediction_difficulty_scale_resolver=None,
    )

    service = module.create_project_prediction_tool_invocation_service(dependencies)

    assert service.registry.list_schemas() == ()
    assert {
        item.tool_name
        for item in service.registry.list_schemas(
            execution_scope=ToolExecutionScope.PROJECT
        )
    } == {
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
    }
    assert service.project_audit_ledger is dependencies.project_audit_ledger
    assert (
        service.agent_run_invocation_validator
        is dependencies.agent_run_invocation_resolver
    )
