from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioProfile,
    ReviewedDefaultScenarioRegistry,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStage,
    FeishuAnalysisJobStatus,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.sibling_planner import (
    ProactiveFeishuSiblingPlanner,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt, FeishuScenarioContextRow
from quanxin_life.scenarios import OperationScenario, ScenarioSegment

NOW = datetime(2026, 8, 13, 13, 0, tzinfo=UTC)
CSV = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"MATR,MATR_b3c34,1,0,0,3.6,1.0,25,1.1,1.05,0.02,true,true\n"
    b"MATR,MATR_b3c34,50,0,0,3.5,1.0,25,1.02,1.0,0.03,true,true\n"
)


class _Queue:
    def __init__(self, *, fail_task_index: int | None = None) -> None:
        self.job_ids: list[str] = []
        self._fail_task_index = fail_task_index
        self._calls = 0

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self._calls += 1
        if self._fail_task_index == self._calls:
            raise RuntimeError("broker unavailable")
        self.job_ids.append(job_id)
        return FeishuJobDispatchReceipt(job_id=job_id, task_id=f"task-{job_id}")


class _FailBeforePersistenceSiblingJobs(SqlAlchemyFeishuSiblingJobService):
    def stage_sibling(
        self,
        *,
        source_job_id: str,
        task: FeishuAnalysisTask,
        scenario_context_id: str | None = None,
        default_scenario_profile_id: str | None = None,
        default_scenario_profile_version: str | None = None,
        default_scenario_profile_sha256: str | None = None,
    ) -> str:
        del (
            source_job_id,
            task,
            scenario_context_id,
            default_scenario_profile_id,
            default_scenario_profile_version,
            default_scenario_profile_sha256,
        )
        raise RuntimeError("database unavailable before sibling persistence")


def _scenario(scenario_id: str, *, temperature_c: float) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=25,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="years-1-25",
                start_year=0,
                end_year=25,
                temperature_c=temperature_c,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=300.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def _profile(digest: str) -> ReviewedDefaultScenarioProfile:
    return ReviewedDefaultScenarioProfile(
        profile_id="matr-b3c34-default",
        profile_version="matr-b3c34-default-v1",
        review_status="APPROVED",
        task_type="compare_operation_scenarios",
        dataset_id="MATR",
        cell_id="MATR_b3c34",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        allowed_cutoff_cycles=(50,),
        allowed_source_sha256s=(digest,),
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        source_cell_format="cylindrical",
        reference_cell_format="prismatic",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        trusted_reference_use=True,
        reference_use_reason_code="MATR_TO_LARGE_FORMAT_LFP_REFERENCE_ONLY",
        initial_state_policy="BOL_ONLY",
        baseline=_scenario("baseline", temperature_c=25.0),
        comparisons=(_scenario("temperature-35c", temperature_c=35.0),),
    )


def _fixture(
    *,
    registry: ReviewedDefaultScenarioRegistry | None,
    fail_task_index: int | None = None,
    fail_before_persistence: bool = False,
    validation_result_id: str | None = "3a3c972b-a23e-42c3-af76-e39038806f14",
) -> tuple[
    ProactiveFeishuSiblingPlanner,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuScenarioContextStore,
    str,
    object,
]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    jobs = SqlAlchemyFeishuJobStore(sessions, lease_seconds=60)
    contexts = SqlAlchemyFeishuScenarioContextStore(sessions)
    batches = InMemoryVerifiedEarlyCycleBatchStore()
    digest = sha256(CSV).hexdigest()
    batch_id = batches.register_canonical_csv(
        CSV,
        registration=CanonicalCsvBatchRegistration(
            metadata=CellMetadata(
                dataset_id="MATR",
                cell_id="MATR_b3c34",
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.1,
                source_uri="memory://MATR_b3c34-cutoff-50.csv",
                source_sha256=digest,
                schema_version="cycle-record-v1",
            ),
            feature_config=EarlyCycleFeatureConfig(
                cutoff_cycle=50,
                feature_version="cyclepatch-multichannel-v1",
            ),
            data_version="matr-three-batch-v1",
            split_version="matr-cell-split-v1",
            provenance=(
                ProvenanceRecord(
                    source_id="MATR_b3c34-cutoff-50",
                    source_kind=SourceKind.OBSERVED,
                    uri="memory://MATR_b3c34-cutoff-50.csv",
                    sha256=digest,
                    description="Reviewed MATR early-cycle upload.",
                    created_at=NOW,
                ),
            ),
        ),
    )
    source_job_id = str(uuid4())
    with sessions() as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-root-planner",
                event_type="im.message.receive_v1",
                payload_sha256="a" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=source_job_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.RUNNING.value,
                job_stage=FeishuAnalysisJobStage.VALIDATING_DATA.value,
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=source_job_id,
                chat_id="oc-planner",
                sender_id="ou-planner",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id=batch_id,
                cell_reference="MATR_b3c34",
                input_file_sha256=digest,
                validation_result_id=validation_result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
        session.commit()
    queue = _Queue(fail_task_index=fail_task_index)
    sibling_jobs_type = (
        _FailBeforePersistenceSiblingJobs
        if fail_before_persistence
        else SqlAlchemyFeishuSiblingJobService
    )
    planner = ProactiveFeishuSiblingPlanner(
        batch_store=batches,
        context_store=contexts,
        sibling_jobs=sibling_jobs_type(
            jobs,
            queue=queue,
            clock=lambda: NOW,
        ),
        default_scenarios=registry,
        clock=lambda: NOW,
    )
    return planner, jobs, contexts, source_job_id, sessions


def test_planner_creates_soh_and_reviewed_scenario_siblings_idempotently() -> None:
    digest = sha256(CSV).hexdigest()
    planner, jobs, contexts, source_job_id, sessions = _fixture(
        registry=ReviewedDefaultScenarioRegistry((_profile(digest),))
    )

    planner.plan_validated_siblings(job=jobs.get(source_job_id))
    planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        children = tuple(
            session.scalars(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.source_job_id == source_job_id)
                .order_by(FeishuEventReceipt.task_type)
            ).all()
        )
    assert [row.task_type for row in children] == [
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY.value,
    ]
    scenario = children[0]
    assert scenario.scenario_context_id is not None
    assert scenario.default_scenario_profile_sha256 == _profile(digest).profile_sha256
    context = contexts.get(scenario.scenario_context_id)
    assert context.data_batch_id == jobs.get(source_job_id).record_batch_id
    assert contexts.resolve_scenario_context(context.scenario_context_id).cell.cell_format == (
        "cylindrical"
    )


def test_planner_without_profile_creates_structured_parameter_request() -> None:
    planner, jobs, _contexts, source_job_id, sessions = _fixture(registry=None)

    planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        children = tuple(
            session.scalars(
                select(FeishuEventReceipt).where(
                    FeishuEventReceipt.source_job_id == source_job_id
                )
            ).all()
        )
    scenario = next(
        row
        for row in children
        if row.task_type == FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value
    )
    assert scenario.scenario_context_id is None
    assert scenario.job_last_error_code is None
    assert scenario.job_task_id is not None


def test_planner_isolates_one_sibling_dispatch_failure() -> None:
    digest = sha256(CSV).hexdigest()
    planner, jobs, _contexts, source_job_id, sessions = _fixture(
        registry=ReviewedDefaultScenarioRegistry((_profile(digest),)),
        fail_task_index=1,
    )

    planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        children = tuple(
            session.scalars(
                select(FeishuEventReceipt).where(
                    FeishuEventReceipt.source_job_id == source_job_id
                )
            ).all()
        )
    assert len(children) == 2
    assert sum(row.job_task_id is not None for row in children) == 1
    assert sum(row.job_last_error_code == "DISPATCH_RETRYABLE" for row in children) == 1


def test_planner_propagates_failure_before_sibling_persistence() -> None:
    planner, jobs, _contexts, source_job_id, sessions = _fixture(
        registry=None,
        fail_before_persistence=True,
    )

    with pytest.raises(RuntimeError, match="before sibling persistence"):
        planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        assert session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.source_job_id == source_job_id
            )
        ) is None


def test_planner_rejects_replay_jobs_before_creating_children_or_contexts() -> None:
    digest = sha256(CSV).hexdigest()
    planner, jobs, _contexts, source_job_id, sessions = _fixture(
        registry=ReviewedDefaultScenarioRegistry((_profile(digest),))
    )
    with sessions() as session:
        source = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.job_id == source_job_id
            )
        )
        assert source is not None
        source.event_type = "feishu.analysis_job.replay_v1"
        session.commit()

    with pytest.raises(ValueError, match="original Feishu file event"):
        planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        assert session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.source_job_id == source_job_id
            )
        ) is None
        assert session.scalar(select(FeishuScenarioContextRow.id)) is None


def test_planner_rejects_unvalidated_roots_before_creating_any_state() -> None:
    digest = sha256(CSV).hexdigest()
    planner, jobs, _contexts, source_job_id, sessions = _fixture(
        registry=ReviewedDefaultScenarioRegistry((_profile(digest),)),
        validation_result_id=None,
    )

    with pytest.raises(ValueError, match="validated original Feishu file event"):
        planner.plan_validated_siblings(job=jobs.get(source_job_id))

    with sessions() as session:
        assert session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.source_job_id == source_job_id
            )
        ) is None
        assert session.scalar(select(FeishuScenarioContextRow.id)) is None
