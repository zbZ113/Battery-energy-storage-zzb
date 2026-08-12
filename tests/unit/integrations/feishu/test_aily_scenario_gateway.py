from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

import pytest
from sqlalchemy import create_engine

from quanxin_life.api.aily import AilyCompareScenarioContextRequest
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu import (
    SqlAlchemyAilyScenarioContextGateway,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.scenarios import OperationScenario, ScenarioSegment

NOW = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)
CONTEXT_ID = UUID("5ab7c239-85a1-4ff2-bc58-2c15c006f311")
CSV = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"scenario-data,cell-1,1,0,0.0,3.6,1.0,25.0,250.0,249.0,0.02,true,true\n"
    b"scenario-data,cell-1,20,0,0.0,3.5,1.0,25.0,248.0,247.0,0.03,true,true\n"
)


class _ApprovedReferenceUse:
    def authorize_reference_use(self, **_: object) -> bool:
        return True


def _scenario(scenario_id: str) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=1,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="year-1",
                start_year=0,
                end_year=1,
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


def _request(*, data_batch_id: str) -> AilyCompareScenarioContextRequest:
    return AilyCompareScenarioContextRequest(
        task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        data_batch_id=data_batch_id,
        cell_format="prismatic",
        baseline=_scenario("baseline"),
        comparisons=(_scenario("comparison"),),
    )


def _batch_store(*, capacity_ah: float, chemistry: str = "LFP/graphite"):
    store = InMemoryVerifiedEarlyCycleBatchStore()
    digest = sha256(CSV).hexdigest()
    batch_id = store.register_canonical_csv(
        CSV,
        registration=CanonicalCsvBatchRegistration(
            metadata=CellMetadata(
                dataset_id="scenario-data",
                cell_id="cell-1",
                chemistry=chemistry,
                nominal_capacity_ah=capacity_ah,
                source_uri="memory://aily-scenario.csv",
                source_sha256=digest,
                schema_version="cycle-record-v1",
            ),
            feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
            data_version="scenario-data-v1",
            split_version="scenario-cell-v1",
            provenance=(
                ProvenanceRecord(
                    source_id="aily-scenario-source",
                    source_kind=SourceKind.OBSERVED,
                    uri="memory://aily-scenario.csv",
                    sha256=digest,
                    description="Verified canonical scenario fixture.",
                    created_at=NOW,
                ),
            ),
        ),
    )
    return store, batch_id


def _context_store() -> SqlAlchemyFeishuScenarioContextStore:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SqlAlchemyFeishuScenarioContextStore(create_session_factory(engine))


def test_gateway_resolves_exact_250ah_route_from_the_verified_batch() -> None:
    batches, batch_id = _batch_store(capacity_ah=250.0)
    contexts = _context_store()
    gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=contexts,
        batch_store=batches,
        created_by_reference="aily-connector",
        clock=lambda: NOW,
        uuid_factory=lambda: CONTEXT_ID,
    )

    state = gateway.create_scenario_context(_request(data_batch_id=batch_id))

    assert state.scenario_context_id == str(CONTEXT_ID)
    assert state.data_batch_id == batch_id
    persisted = contexts.get(state.scenario_context_id)
    assert persisted.route_id == "blast-lite-lfp-gr-250ah-prismatic-2019-v1"
    assert (
        contexts.resolve_scenario_context(state.scenario_context_id).cell.nominal_capacity_ah
        == 250.0
    )
    assert "route_id" not in state.model_dump(mode="json")


def test_gateway_rejects_nonexact_prismatic_reference_use_by_default() -> None:
    batches, batch_id = _batch_store(capacity_ah=280.0)
    gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=_context_store(),
        batch_store=batches,
        created_by_reference="aily-connector",
        clock=lambda: NOW,
        uuid_factory=lambda: CONTEXT_ID,
    )

    with pytest.raises(ValueError, match="reference route"):
        gateway.create_scenario_context(_request(data_batch_id=batch_id))


def test_gateway_allows_nonexact_prismatic_reference_only_with_reviewed_approval() -> None:
    batches, batch_id = _batch_store(capacity_ah=280.0)
    contexts = _context_store()
    gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=contexts,
        batch_store=batches,
        created_by_reference="aily-connector",
        reference_use_authorizer=_ApprovedReferenceUse(),
        clock=lambda: NOW,
        uuid_factory=lambda: CONTEXT_ID,
    )

    state = gateway.create_scenario_context(_request(data_batch_id=batch_id))

    resolved = contexts.resolve_scenario_context(state.scenario_context_id)
    assert resolved.cell.nominal_capacity_ah == 280.0
    assert resolved.trusted_reference_use is True


def test_gateway_rejects_non_lfp_chemistry_without_creating_a_context() -> None:
    batches, batch_id = _batch_store(capacity_ah=250.0, chemistry="NMC")
    gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=_context_store(),
        batch_store=batches,
        created_by_reference="aily-connector",
        reference_use_authorizer=_ApprovedReferenceUse(),
        clock=lambda: NOW,
        uuid_factory=lambda: CONTEXT_ID,
    )

    with pytest.raises(ValueError, match="reference route"):
        gateway.create_scenario_context(_request(data_batch_id=batch_id))
