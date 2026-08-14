from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.api.aily import AilyCreateAnalysisTaskRequest
from quanxin_life.core import (
    AgentRunStatus,
    ProvenanceRecord,
    SourceKind,
)
from quanxin_life.infrastructure.feishu_queue import CeleryFeishuJobQueue
from quanxin_life.integrations.feishu.aily_tasks import (
    SqlAlchemyAilyAnalysisTaskGateway,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStage,
    FeishuAnalysisJobStatus,
    FeishuJobDeliveryReceipt,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.scenarios import (
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSegment,
    VerifiedScenarioContext,
)
from quanxin_life.tools.blast_scenarios import CompareOperationScenariosToolInput

NOW = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
BATCH_ID = "canonical-csv-" + "1" * 64
SOURCE_RUN_ID = "6996ec78-e8ac-49a9-b5c0-59991a552b3d"


class _AsyncResult:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


class _CeleryApp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult:
        self.calls.append({"name": name, "kwargs": kwargs, "queue": queue})
        return _AsyncResult(f"task-{len(self.calls)}")


class _DataIdentityResolver:
    def __init__(self, source_job_id: str) -> None:
        self.source_job_id = source_job_id
        self.calls: list[tuple[str, str]] = []
        self.active = True

    def resolve_source_job(
        self,
        *,
        source_run_id: str,
        data_batch_id: str,
    ) -> None:
        self.calls.append((source_run_id, data_batch_id))
        if (
            not self.active
            or source_run_id != self.source_job_id
            or data_batch_id != BATCH_ID
        ):
            raise ValueError("Aily data identity is not authorized")


def _scenario() -> OperationScenario:
    return OperationScenario(
        scenario_id="baseline",
        scenario_version="baseline-v1",
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


def _fixture(
    *,
    data_identity_resolver: _DataIdentityResolver | None = None,
    context_source_run_id: str = SOURCE_RUN_ID,
) -> tuple[
    SqlAlchemyAilyAnalysisTaskGateway,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuScenarioContextStore,
    _CeleryApp,
    object,
    str,
]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    with session_factory.begin() as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-aily-source",
                event_type="im.message.receive_v1",
                payload_sha256="f" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=SOURCE_RUN_ID,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=SOURCE_RUN_ID,
                chat_id="oc-aily-source",
                sender_id="ou-aily-source",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id=BATCH_ID,
                cell_reference="cell-1",
                input_file_sha256="1" * 64,
                validation_result_id=str(uuid4()),
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
    contexts = SqlAlchemyFeishuScenarioContextStore(session_factory)
    context_id = str(uuid4())
    cell = ScenarioCellDescriptor(
        chemistry="LFP/graphite",
        nominal_capacity_ah=250.0,
        cell_format="prismatic",
    )
    contexts.create(
        task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        data_batch_id=BATCH_ID,
        verified_context=VerifiedScenarioContext(
            scenario_context_id=context_id,
            cell=cell,
            trusted_reference_use=True,
            data_version="scenario-data-v1",
            provenance=(
                ProvenanceRecord(
                    source_id="scenario-batch",
                    source_kind=SourceKind.OBSERVED,
                    uri="record-batch:scenario-batch",
                    sha256="1" * 64,
                    description="Verified scenario input batch.",
                    created_at=NOW,
                ),
            ),
        ),
        analysis_input=CompareOperationScenariosToolInput(
            run_id=context_id,
            scenario_context_id=context_id,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            cell=cell,
            baseline=_scenario(),
            comparisons=(
                _scenario().model_copy(
                    update={
                        "scenario_id": "comparison",
                        "scenario_version": "comparison-v1",
                    }
                ),
            ),
        ),
        created_by_reference=context_source_run_id,
        created_at=NOW,
    )
    jobs = SqlAlchemyFeishuJobStore(session_factory, lease_seconds=60)
    app = _CeleryApp()
    gateway = SqlAlchemyAilyAnalysisTaskGateway(
        job_store=jobs,
        scenario_context_store=contexts,
        data_identity_resolver=(
            data_identity_resolver or _DataIdentityResolver(SOURCE_RUN_ID)
        ),
        queue=CeleryFeishuJobQueue(app=app),
        clock=lambda: NOW,
    )
    return gateway, jobs, contexts, app, session_factory, context_id


def _request(context_id: str) -> AilyCreateAnalysisTaskRequest:
    return AilyCreateAnalysisTaskRequest(
        task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        source_run_id=SOURCE_RUN_ID,
        scenario_context_id=context_id,
    )


def test_gateway_persists_reference_only_aily_job_and_enqueues_identity() -> None:
    gateway, _jobs, _contexts, app, session_factory, context_id = _fixture()

    state = gateway.create_analysis_task(_request(context_id))

    assert state.status is AgentRunStatus.PLANNING
    assert state.result_ids == ()
    assert len(app.calls) == 1
    assert app.calls[0]["kwargs"] == {"job_id": state.run_id}
    with session_factory() as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == state.run_id)
        )
        assert row is not None
        assert row.job_origin == FeishuAnalysisJobOrigin.AILY.value
        assert row.scenario_context_id == context_id
        assert row.task_type == FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value
        assert row.message_id is None
        assert row.file_key is None
        assert row.chat_id == "oc-aily-source"
        assert row.sender_id == "ou-aily-source"


def test_gateway_is_idempotent_for_one_immutable_scenario_context() -> None:
    gateway, _jobs, _contexts, app, _session_factory, context_id = _fixture()

    first = gateway.create_analysis_task(_request(context_id))
    second = gateway.create_analysis_task(_request(context_id))

    assert second == first
    assert len(app.calls) == 1


def test_gateway_rejects_a_scenario_context_bound_to_another_source_run() -> None:
    gateway, _jobs, _contexts, app, _session_factory, context_id = _fixture(
        context_source_run_id="f5b1293d-8d85-420d-97e2-1d5a1b1bc1a1"
    )

    with pytest.raises(ValueError, match="source upload"):
        gateway.create_analysis_task(_request(context_id))

    assert app.calls == []


def test_gateway_reauthorizes_the_source_before_returning_task_state() -> None:
    identities = _DataIdentityResolver(SOURCE_RUN_ID)
    gateway, _jobs, _contexts, _app, _session_factory, context_id = _fixture(
        data_identity_resolver=identities
    )
    created = gateway.create_analysis_task(_request(context_id))
    identities.active = False

    with pytest.raises(ValueError, match="authorized"):
        gateway.get_analysis_task(created.run_id)


def test_gateway_maps_only_durable_checkpoint_result_ids() -> None:
    gateway, jobs, _contexts, _app, _session_factory, context_id = _fixture()
    created = gateway.create_analysis_task(_request(context_id))
    claim = jobs.claim(job_id=created.run_id, claimed_at=NOW)
    assert claim.claim_token is not None
    validation_id = str(uuid4())
    analysis_id = str(uuid4())
    report_id = str(uuid4())
    jobs.checkpoint_results(
        job_id=created.run_id,
        claim_token=claim.claim_token,
        validation_result_id=validation_id,
        analysis_result_id=analysis_id,
        report_result_id=report_id,
        updated_at=NOW,
    )
    jobs.finish(
        job_id=created.run_id,
        claim_token=claim.claim_token,
        status=FeishuAnalysisJobStatus.SUCCEEDED,
        stage=FeishuAnalysisJobStage.SUCCEEDED,
        error_code=None,
        delivery=FeishuJobDeliveryReceipt(
            bitable_record_id="rec-aily",
            report_file_key=None,
        ),
        completed_at=NOW,
    )

    restored = gateway.get_analysis_task(created.run_id)

    assert restored.status is AgentRunStatus.COMPLETED
    assert restored.completed_step_ids == ("validate", "analyze", "report")
    assert restored.result_ids == (validation_id, analysis_id, report_id)


def test_gateway_rejects_unauthorized_batches_and_mismatched_contexts() -> None:
    gateway, _jobs, _contexts, app, _session_factory, context_id = _fixture()

    with pytest.raises(ValueError, match="authorized"):
        gateway.create_analysis_task(
            AilyCreateAnalysisTaskRequest(
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
                source_run_id=SOURCE_RUN_ID,
                data_batch_id="batch-1",
            )
        )
    with pytest.raises(ValueError, match="task"):
        gateway.create_analysis_task(
            AilyCreateAnalysisTaskRequest(
                task_type=FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
                source_run_id=SOURCE_RUN_ID,
                scenario_context_id=context_id,
            )
        )
    assert app.calls == []


def test_gateway_hides_non_aily_jobs() -> None:
    gateway, _jobs, _contexts, _app, session_factory, _context_id = _fixture()
    feishu_job_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-feishu-only",
                event_type="card.action.trigger",
                payload_sha256="a" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=feishu_job_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.PENDING.value,
                job_stage=FeishuAnalysisJobStage.RECEIVED.value,
                task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
                run_id=feishu_job_id,
                event_time=NOW,
                job_attempt_count=0,
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )

    with pytest.raises(LookupError, match="Aily"):
        gateway.get_analysis_task(feishu_job_id)


def test_gateway_exposes_staged_recommendation_from_root_only_request() -> None:
    gateway, jobs, _contexts, app, _sessions, _context_id = _fixture()
    recommendation_id = SqlAlchemyFeishuSiblingJobService(
        jobs,
        queue=CeleryFeishuJobQueue(app=app),
        clock=lambda: NOW,
    ).stage_sibling(
        source_job_id=SOURCE_RUN_ID,
        task=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
        recommendation_ruleset_id="reviewed-release-gate",
        recommendation_ruleset_version="reviewed-release-gate-v1",
        recommendation_ruleset_sha256="e" * 64,
    )

    state = gateway.create_analysis_task(
        AilyCreateAnalysisTaskRequest(
            task_type=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
            source_run_id=SOURCE_RUN_ID,
        )
    )

    assert state.run_id == recommendation_id
    assert state.status is AgentRunStatus.PLANNING
    assert gateway.get_analysis_task(recommendation_id) == state
    assert app.calls == []


def test_aily_recommendation_request_rejects_caller_supplied_data_references() -> None:
    with pytest.raises(ValueError, match="recommendation"):
        AilyCreateAnalysisTaskRequest(
            task_type=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
            source_run_id=SOURCE_RUN_ID,
            data_batch_id=BATCH_ID,
        )


@pytest.mark.parametrize(
    "task",
    [
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    ],
)
def test_gateway_persists_project_bound_non_scenario_aily_jobs(
    task: FeishuAnalysisTask,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    source_job_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-source-upload",
                event_type="im.message.receive_v1",
                payload_sha256="a" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=source_job_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=source_job_id,
                chat_id="oc-aily-project",
                sender_id="ou-aily-user",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id=BATCH_ID,
                cell_reference="cell-1",
                input_file_sha256="1" * 64,
                validation_result_id=str(uuid4()),
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
    jobs = SqlAlchemyFeishuJobStore(session_factory, lease_seconds=60)
    app = _CeleryApp()
    identities = _DataIdentityResolver(source_job_id)
    gateway = SqlAlchemyAilyAnalysisTaskGateway(
        job_store=jobs,
        scenario_context_store=SqlAlchemyFeishuScenarioContextStore(session_factory),
        data_identity_resolver=identities,
        queue=CeleryFeishuJobQueue(app=app),
        clock=lambda: NOW,
    )

    state = gateway.create_analysis_task(
        AilyCreateAnalysisTaskRequest(
            task_type=task,
            source_run_id=source_job_id,
            data_batch_id=BATCH_ID,
        )
    )

    persisted = jobs.get(state.run_id)
    assert persisted.job_origin is FeishuAnalysisJobOrigin.AILY
    assert persisted.source_job_id == source_job_id
    assert persisted.record_batch_id == BATCH_ID
    assert persisted.chat_id == "oc-aily-project"
    assert persisted.sender_id == "ou-aily-user"
    assert persisted.task_type is task
    assert identities.calls == [(source_job_id, BATCH_ID)]
    assert app.calls == [
        {
            "name": "quanxin_life.feishu.analysis.execute.v1",
            "kwargs": {"job_id": state.run_id},
            "queue": "agent-runs",
        }
    ]
