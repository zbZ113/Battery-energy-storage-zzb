from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.core import ProvenanceRecord, SourceKind
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuScenarioContextRow
from quanxin_life.scenarios import (
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSegment,
    VerifiedScenarioContext,
)
from quanxin_life.tools.blast_scenarios import CompareOperationScenariosToolInput

NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)


def _scenario(scenario_id: str) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=15,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="all-years",
                start_year=0,
                end_year=15,
                temperature_c=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def _context(context_id: str) -> VerifiedScenarioContext:
    return VerifiedScenarioContext(
        scenario_context_id=context_id,
        cell=ScenarioCellDescriptor(
            chemistry="LFP/graphite",
            nominal_capacity_ah=250.0,
            cell_format="prismatic",
        ),
        data_version="verified-scenario-data-v1",
        provenance=(
            ProvenanceRecord(
                source_id="verified-batch",
                source_kind=SourceKind.OBSERVED,
                uri="memory://verified-batch",
                sha256="a" * 64,
                description="Verified scenario-context fixture.",
                created_at=NOW,
            ),
        ),
    )


def _analysis_input(context_id: str) -> CompareOperationScenariosToolInput:
    return CompareOperationScenariosToolInput(
        run_id=context_id,
        scenario_context_id=context_id,
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        cell=_context(context_id).cell,
        baseline=_scenario("baseline"),
        comparisons=(_scenario("comparison"),),
    )


def _store() -> tuple[SqlAlchemyFeishuScenarioContextStore, object]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyFeishuScenarioContextStore(session_factory), session_factory


def test_scenario_context_round_trips_typed_input_and_rebinds_only_run_id() -> None:
    store, _session_factory = _store()
    context_id = str(uuid4())
    created = store.create(
        task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        data_batch_id="batch-scenario-1",
        verified_context=_context(context_id),
        analysis_input=_analysis_input(context_id),
        created_by_reference="ou-scenario-user",
        created_at=NOW,
    )
    run_id = str(uuid4())

    resolved_context = store.resolve_scenario_context(context_id)
    resolved_input = store.resolve_analysis_input(
        scenario_context_id=context_id,
        task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        run_id=run_id,
    )

    assert created.scenario_context_id == context_id
    assert created.data_batch_id == "batch-scenario-1"
    assert resolved_context == _context(context_id)
    assert resolved_input["run_id"] == run_id
    assert resolved_input["scenario_context_id"] == context_id
    assert resolved_input["baseline"] == _scenario("baseline").model_dump(mode="json")


def test_scenario_context_detects_persisted_payload_tampering() -> None:
    store, session_factory = _store()
    context_id = str(uuid4())
    store.create(
        task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        data_batch_id="batch-scenario-1",
        verified_context=_context(context_id),
        analysis_input=_analysis_input(context_id),
        created_by_reference="ou-scenario-user",
        created_at=NOW,
    )
    with session_factory() as session:
        row = session.scalar(
            select(FeishuScenarioContextRow).where(
                FeishuScenarioContextRow.id == context_id
            )
        )
        assert row is not None
        row.route_id = "blast-lite-lfp-gr-sony-murata-3ah-2018-v1"
        session.commit()

    with pytest.raises(ValueError, match="integrity"):
        store.resolve_scenario_context(context_id)


def test_scenario_context_rejects_task_or_identity_mismatch() -> None:
    store, _session_factory = _store()
    context_id = str(uuid4())
    with pytest.raises(ValueError, match="scenario context"):
        store.create(
            task=FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
            data_batch_id="batch-scenario-1",
            verified_context=_context(context_id),
            analysis_input=_analysis_input(context_id),
            created_by_reference="ou-scenario-user",
            created_at=NOW,
        )
