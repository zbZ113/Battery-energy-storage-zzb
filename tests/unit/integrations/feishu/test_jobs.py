from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.core import sha256_canonical
from quanxin_life.integrations.feishu.events import FeishuReceiptClaimStatus
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStage,
    FeishuAnalysisJobStatus,
    FeishuJobClaimStatus,
    FeishuJobDeliveryReceipt,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobReplayService,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.routing import (
    FeishuEventReference,
    FeishuInboundEventKind,
)
from quanxin_life.integrations.feishu.sqlalchemy_receipts import (
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt

NOW = datetime(2026, 8, 11, 4, 0, tzinfo=UTC)
SCENARIO_CONTEXT_ID = "ea28ad37-d072-4d42-9d2a-fbc4ab5158db"


class _Queue:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_ids.append(job_id)
        return FeishuJobDispatchReceipt(job_id=job_id, task_id=f"task-{job_id}")


class _FailOnceQueue(_Queue):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 1

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("broker unavailable")
        return super().enqueue(job_id=job_id)


def _fixture() -> tuple[
    SqlAlchemyFeishuReceiptStore,
    SqlAlchemyFeishuJobStore,
    object,
]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    return (
        SqlAlchemyFeishuReceiptStore(session_factory, lease_seconds=30),
        SqlAlchemyFeishuJobStore(session_factory, lease_seconds=30),
        session_factory,
    )


def _reference() -> FeishuEventReference:
    return FeishuEventReference(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        kind=FeishuInboundEventKind.FILE,
        chat_id="oc-job",
        user_id="ou-job",
        message_id="om-job",
        file_key="file-job",
        file_name="cell.csv",
        receive_id_type="chat_id",
        event_time=NOW,
    )


def _scenario_reference() -> FeishuEventReference:
    return FeishuEventReference(
        event_id="evt-scenario-job-1",
        event_type="card.action.trigger",
        kind=FeishuInboundEventKind.CARD_ACTION,
        chat_id="oc-job",
        user_id="ou-job",
        message_id="om-scenario-card",
        receive_id_type="chat_id",
        event_time=NOW,
        action_value={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "scenario_context_id": SCENARIO_CONTEXT_ID,
        },
    )


def test_router_persists_sanitized_reference_and_dispatches_identity_once() -> None:
    receipts, jobs, session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.status is FeishuReceiptClaimStatus.NEW
    assert claim.claim_token is not None
    queue = _Queue()
    router = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    )

    first = router.route_event(event=_reference(), claim_token=claim.claim_token)
    second = router.route_event(event=_reference(), claim_token=claim.claim_token)

    assert first is FeishuEventRouteStatus.ENQUEUED
    assert second is FeishuEventRouteStatus.ALREADY_ENQUEUED
    assert len(queue.job_ids) == 1
    with session_factory() as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.event_id == "evt-job-1"
            )
        )
        assert row is not None
        assert row.job_id == queue.job_ids[0]
        assert row.run_id == row.job_id
        assert row.job_status == FeishuAnalysisJobStatus.PENDING.value
        assert row.job_stage == "RECEIVED"
        assert row.message_id == "om-job"
        assert row.file_key == "file-job"
        assert row.file_name == "cell.csv"
        assert row.chat_id == "oc-job"
        assert row.sender_id == "ou-job"
        assert row.receive_id_type == "chat_id"
        assert row.event_time == NOW
        assert row.job_task_id == f"task-{row.job_id}"


def test_sibling_service_stages_idempotent_jobs_from_persisted_batch() -> None:
    _receipts, jobs, session_factory = _fixture()
    source_job_id = _seed_completed_source_job(session_factory)
    queue = _Queue()
    service = SqlAlchemyFeishuSiblingJobService(
        jobs,
        queue=queue,
        clock=lambda: NOW,
    )

    first = service.stage_sibling(
        source_job_id=source_job_id,
        task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    )
    second = service.stage_sibling(
        source_job_id=source_job_id,
        task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    )

    assert first == second
    assert queue.job_ids == [first]
    sibling = jobs.get(first)
    assert sibling.source_job_id == source_job_id
    assert sibling.task_type is FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY
    assert sibling.record_batch_id == "canonical-csv-" + "b" * 64
    assert sibling.message_id is None
    assert sibling.file_key is None
    assert sibling.input_file_sha256 == "c" * 64
    assert sibling.validation_result_id is None


def test_sibling_dispatch_recovers_after_broker_failure() -> None:
    _receipts, jobs, session_factory = _fixture()
    source_job_id = _seed_completed_source_job(session_factory)
    queue = _FailOnceQueue()
    service = SqlAlchemyFeishuSiblingJobService(
        jobs,
        queue=queue,
        clock=lambda: NOW,
    )

    sibling_id = service.stage_sibling(
        source_job_id=source_job_id,
        task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    )

    pending = jobs.list_undispatched_siblings(source_job_id=source_job_id)
    assert pending == (sibling_id,)
    assert service.dispatch_pending(source_job_id=source_job_id) == (pending[0],)
    assert jobs.get(pending[0]).job_status is FeishuAnalysisJobStatus.PENDING


def test_sibling_store_rejects_replay_job_as_a_proactive_source() -> None:
    _receipts, jobs, session_factory = _fixture()
    source_job_id = _seed_completed_source_job(session_factory)
    with session_factory() as session:
        source = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.job_id == source_job_id
            )
        )
        assert source is not None
        source.event_type = "feishu.analysis_job.replay_v1"
        session.commit()

    with pytest.raises(ValueError, match="validated root Feishu file job"):
        SqlAlchemyFeishuSiblingJobService(
            jobs,
            queue=_Queue(),
            clock=lambda: NOW,
        ).stage_sibling(
            source_job_id=source_job_id,
            task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
        )


def test_sibling_store_rejects_a_root_without_a_validation_result() -> None:
    _receipts, jobs, session_factory = _fixture()
    source_job_id = _seed_completed_source_job(
        session_factory,
        validation_result_id=None,
    )

    with pytest.raises(ValueError, match="validated root Feishu file job"):
        SqlAlchemyFeishuSiblingJobService(
            jobs,
            queue=_Queue(),
            clock=lambda: NOW,
        ).stage_sibling(
            source_job_id=source_job_id,
            task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
        )


def test_sibling_store_rejects_a_rejected_root() -> None:
    _receipts, jobs, session_factory = _fixture()
    source_job_id = _seed_completed_source_job(session_factory)
    with session_factory() as session:
        source = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.job_id == source_job_id
            )
        )
        assert source is not None
        source.job_status = FeishuAnalysisJobStatus.REJECTED.value
        source.job_stage = FeishuAnalysisJobStage.REJECTED.value
        source.job_last_error_code = "DATA_VALIDATION_BLOCKED"
        session.commit()

    with pytest.raises(ValueError, match="validated root Feishu file job"):
        SqlAlchemyFeishuSiblingJobService(
            jobs,
            queue=_Queue(),
            clock=lambda: NOW,
        ).stage_sibling(
            source_job_id=source_job_id,
            task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
        )


def _seed_completed_source_job(
    session_factory: object,
    *,
    validation_result_id: str | None = "3a3c972b-a23e-42c3-af76-e39038806f14",
) -> str:
    source_job_id = "3a3c972b-a23e-42c3-af76-e39038806f13"
    with session_factory() as session:
        session.add(
            FeishuEventReceipt(
                id="source-receipt",
                event_id="source-event",
                event_type="im.message.receive_v1",
                payload_sha256="d" * 64,
                status=FeishuReceiptClaimStatus.PROCESSED.value,
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=source_job_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=source_job_id,
                chat_id="oc-job",
                sender_id="ou-job",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id="canonical-csv-" + "b" * 64,
                cell_reference="MATR_b3c34",
                input_file_sha256="c" * 64,
                validation_result_id=validation_result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
        session.commit()
    return source_job_id


def test_expired_worker_claim_is_recovered_with_a_new_fence() -> None:
    receipts, jobs, _session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    router = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    )
    router.route_event(event=_reference(), claim_token=claim.claim_token)
    job_id = queue.job_ids[0]

    first = jobs.claim(job_id=job_id, claimed_at=NOW)
    assert first.status is FeishuJobClaimStatus.CLAIMED
    assert first.claim_token is not None

    busy = jobs.claim(job_id=job_id, claimed_at=NOW + timedelta(seconds=10))
    assert busy.status is FeishuJobClaimStatus.IN_PROGRESS

    recovered = jobs.claim(job_id=job_id, claimed_at=NOW + timedelta(seconds=31))
    assert recovered.status is FeishuJobClaimStatus.CLAIMED
    assert recovered.claim_token is not None
    assert recovered.claim_token != first.claim_token
    assert recovered.attempt == 2


def test_job_heartbeat_renews_only_the_live_fenced_claim() -> None:
    receipts, jobs, _session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    router = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    )
    router.route_event(event=_reference(), claim_token=claim.claim_token)
    job_id = queue.job_ids[0]
    owned = jobs.claim(job_id=job_id, claimed_at=NOW)
    assert owned.claim_token is not None

    renewed = jobs.renew_claim(
        job_id=job_id,
        claim_token=owned.claim_token,
        renewed_at=NOW + timedelta(seconds=10),
    )

    assert renewed == NOW + timedelta(seconds=40)


def test_router_persists_only_scenario_context_reference_for_card_action() -> None:
    receipts, jobs, session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-scenario-job-1",
        event_type="card.action.trigger",
        payload_sha256="b" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    router = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask(event.action_value["task_type"]),
        clock=lambda: NOW,
    )

    status = router.route_event(
        event=_scenario_reference(),
        claim_token=claim.claim_token,
    )

    assert status is FeishuEventRouteStatus.ENQUEUED
    with session_factory() as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.event_id == "evt-scenario-job-1"
            )
        )
        assert row is not None
        assert row.scenario_context_id == SCENARIO_CONTEXT_ID
        assert row.file_key is None
        assert row.file_name is None
        assert row.task_type == FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value


def test_live_worker_checkpoints_delivery_references_for_retry_replay() -> None:
    receipts, jobs, _session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=_reference(), claim_token=claim.claim_token)
    job_id = queue.job_ids[0]
    owned = jobs.claim(job_id=job_id, claimed_at=NOW)
    assert owned.claim_token is not None

    jobs.checkpoint_delivery(
        job_id=job_id,
        claim_token=owned.claim_token,
        scenario_image_key="img-result",
        analysis_image_key="img-analysis",
        analysis_image_renderer_version="feishu-soh-plot-v1",
        analysis_image_sha256="a" * 64,
        result_card_message_id="om-result-card",
        report_file_key="file-report",
        report_message_id="om-report-file",
        report_card_message_id="om-report-card",
        bitable_record_id="rec-run",
        updated_at=NOW,
    )

    snapshot = jobs.get(job_id)
    assert snapshot.scenario_image_key == "img-result"
    assert snapshot.analysis_image_key == "img-analysis"
    assert snapshot.analysis_image_renderer_version == "feishu-soh-plot-v1"
    assert snapshot.analysis_image_sha256 == "a" * 64
    assert snapshot.result_card_message_id == "om-result-card"
    assert snapshot.report_file_key == "file-report"
    assert snapshot.report_message_id == "om-report-file"
    assert snapshot.report_card_message_id == "om-report-card"
    assert snapshot.bitable_record_id == "rec-run"


def test_job_store_rejects_partial_analysis_image_provenance() -> None:
    receipts, jobs, _session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=_reference(), claim_token=claim.claim_token)
    owned = jobs.claim(job_id=queue.job_ids[0], claimed_at=NOW)
    assert owned.claim_token is not None

    with pytest.raises(ValueError, match="image provenance is incomplete"):
        jobs.checkpoint_delivery(
            job_id=queue.job_ids[0],
            claim_token=owned.claim_token,
            analysis_image_key="img-analysis",
            updated_at=NOW,
        )


def test_route_rejection_replay_creates_one_new_auditable_job() -> None:
    receipts, jobs, session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    original_queue = _Queue()
    SqlAlchemyFeishuJobRouter(
        jobs,
        queue=original_queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=_reference(), claim_token=claim.claim_token)
    source_job_id = original_queue.job_ids[0]
    owned = jobs.claim(job_id=source_job_id, claimed_at=NOW)
    assert owned.claim_token is not None
    jobs.checkpoint_batch(
        job_id=source_job_id,
        claim_token=owned.claim_token,
        record_batch_id="canonical-csv-" + "b" * 64,
        cell_reference="MATR_b3c34",
        input_file_sha256="c" * 64,
        updated_at=NOW,
    )
    jobs.checkpoint_results(
        job_id=source_job_id,
        claim_token=owned.claim_token,
        validation_result_id="d" * 64,
        updated_at=NOW,
    )
    jobs.checkpoint_delivery(
        job_id=source_job_id,
        claim_token=owned.claim_token,
        result_card_message_id="om-original-rejection",
        bitable_record_id="rec-original-rejection",
        updated_at=NOW,
    )
    jobs.finish(
        job_id=source_job_id,
        claim_token=owned.claim_token,
        status=FeishuAnalysisJobStatus.REJECTED,
        stage=FeishuAnalysisJobStage.REJECTED,
        error_code="MODEL_ROUTE_NOT_ACTIVATED",
        delivery=FeishuJobDeliveryReceipt(
            bitable_record_id="rec-original-rejection",
            report_file_key=None,
        ),
        completed_at=NOW,
    )
    replay_queue = _Queue()
    service = SqlAlchemyFeishuJobReplayService(
        jobs,
        queue=replay_queue,
        clock=lambda: NOW + timedelta(minutes=1),
    )

    replay_job_id = service.replay_rejected(
        source_job_id=source_job_id,
        replay_key="project-binding-activated-v1",
    )
    duplicate_job_id = service.replay_rejected(
        source_job_id=source_job_id,
        replay_key="project-binding-activated-v1",
    )

    assert duplicate_job_id == replay_job_id
    assert replay_queue.job_ids == [replay_job_id]
    original = jobs.get(source_job_id)
    replay = jobs.get(replay_job_id)
    assert original.job_status is FeishuAnalysisJobStatus.REJECTED
    assert original.job_last_error_code == "MODEL_ROUTE_NOT_ACTIVATED"
    assert original.validation_result_id == "d" * 64
    assert original.result_card_message_id == "om-original-rejection"
    assert original.bitable_record_id == "rec-original-rejection"
    assert replay.job_origin is FeishuAnalysisJobOrigin.FEISHU
    assert replay.job_status is FeishuAnalysisJobStatus.PENDING
    assert replay.job_stage == FeishuAnalysisJobStage.RECEIVED.value
    assert replay.job_attempt_count == 0
    assert replay.job_completed_at is None
    assert replay.message_id == original.message_id
    assert replay.file_key == original.file_key
    assert replay.file_name == original.file_name
    assert replay.chat_id == original.chat_id
    assert replay.sender_id == original.sender_id
    assert replay.receive_id_type == original.receive_id_type
    assert replay.record_batch_id is None
    assert replay.validation_result_id is None
    assert replay.analysis_result_id is None
    assert replay.report_result_id is None
    assert replay.result_card_message_id is None
    assert replay.bitable_record_id is None
    with session_factory() as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.job_id == replay_job_id
            )
        )
        assert row is not None
        assert row.event_type == "feishu.analysis_job.replay_v1"
        assert row.event_id.startswith(f"replay:{source_job_id}:")
        assert len(row.payload_sha256) == 64


def test_replay_rejects_non_route_terminal_reason() -> None:
    receipts, jobs, _session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=_reference(), claim_token=claim.claim_token)
    source_job_id = queue.job_ids[0]
    owned = jobs.claim(job_id=source_job_id, claimed_at=NOW)
    assert owned.claim_token is not None
    jobs.finish(
        job_id=source_job_id,
        claim_token=owned.claim_token,
        status=FeishuAnalysisJobStatus.REJECTED,
        stage=FeishuAnalysisJobStage.REJECTED,
        error_code="ATTACHMENT_REJECTED",
        delivery=None,
        completed_at=NOW,
    )

    with pytest.raises(ValueError, match="MODEL_ROUTE_NOT_ACTIVATED"):
        SqlAlchemyFeishuJobReplayService(
            jobs,
            queue=_Queue(),
            clock=lambda: NOW + timedelta(minutes=1),
        ).replay_rejected(
            source_job_id=source_job_id,
            replay_key="project-binding-activated-v1",
        )


def _claimed_job() -> tuple[SqlAlchemyFeishuJobStore, str, str]:
    receipts, jobs, _ = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    staged = jobs.stage(
        event=_reference(),
        claim_token=claim.claim_token,
        task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        staged_at=NOW,
    )
    owned = jobs.claim(job_id=staged.job_id, claimed_at=NOW)
    assert owned.claim_token is not None
    return jobs, staged.job_id, owned.claim_token


def test_live_worker_checkpoints_sanitized_csv_mapping_evidence() -> None:
    jobs, job_id, claim_token = _claimed_job()
    evidence = {
        "conflicts": [],
        "missing_fields": ["dataset_id", "cell_id"],
        "raw_sha256": "c" * 64,
        "reason_code": "UNREVIEWED_LAYOUT",
        "unit_required": [],
        "unknown_fields": ["Cell Name", "Cycle Number"],
        "value_errors": [],
    }

    jobs.checkpoint_csv_mapping(
        job_id=job_id,
        claim_token=claim_token,
        status="REJECTED",
        evidence=evidence,
        evidence_sha256=sha256_canonical(evidence),
        updated_at=NOW,
    )

    snapshot = jobs.get(job_id)
    assert snapshot.csv_mapping_status == "REJECTED"
    assert snapshot.csv_mapping_evidence == evidence
    assert snapshot.csv_mapping_evidence_sha256 == sha256_canonical(evidence)


def test_csv_mapping_checkpoint_rejects_mismatched_evidence_hash() -> None:
    jobs, job_id, claim_token = _claimed_job()
    evidence = {
        "conflicts": [],
        "missing_fields": [],
        "raw_sha256": "c" * 64,
        "reason_code": "UNREVIEWED_LAYOUT",
        "unit_required": [],
        "unknown_fields": [],
        "value_errors": [],
    }

    with pytest.raises(ValueError, match="evidence SHA-256"):
        jobs.checkpoint_csv_mapping(
            job_id=job_id,
            claim_token=claim_token,
            status="REJECTED",
            evidence=evidence,
            evidence_sha256="d" * 64,
            updated_at=NOW,
        )


@pytest.mark.parametrize(
    "unsafe_key",
    ("rows", "message_text", "payload", "values"),
)
def test_csv_mapping_checkpoint_rejects_non_metadata_evidence(
    unsafe_key: str,
) -> None:
    jobs, job_id, claim_token = _claimed_job()
    evidence = {
        "field_names": [],
        "raw_sha256": "c" * 64,
        "reason_code": "UNREVIEWED_LAYOUT",
        unsafe_key: ["secret-cell-value"],
    }

    with pytest.raises(ValueError, match="mapping evidence contract"):
        jobs.checkpoint_csv_mapping(
            job_id=job_id,
            claim_token=claim_token,
            status="REJECTED",
            evidence=evidence,
            evidence_sha256=sha256_canonical(evidence),
            updated_at=NOW,
        )


def test_job_read_rejects_tampered_csv_mapping_evidence_hash() -> None:
    receipts, jobs, session_factory = _fixture()
    claim = receipts.claim(
        event_id="evt-job-1",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    staged = jobs.stage(
        event=_reference(),
        claim_token=claim.claim_token,
        task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        staged_at=NOW,
    )
    owned = jobs.claim(job_id=staged.job_id, claimed_at=NOW)
    assert owned.claim_token is not None
    evidence = {
        "conflicts": [],
        "missing_fields": [],
        "raw_sha256": "c" * 64,
        "reason_code": "UNREVIEWED_LAYOUT",
        "unit_required": [],
        "unknown_fields": [],
        "value_errors": [],
    }
    jobs.checkpoint_csv_mapping(
        job_id=staged.job_id,
        claim_token=owned.claim_token,
        status="REJECTED",
        evidence=evidence,
        evidence_sha256=sha256_canonical(evidence),
        updated_at=NOW,
    )
    with session_factory() as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.job_id == staged.job_id
            )
        )
        assert row is not None
        row.csv_mapping_evidence_sha256 = "f" * 64
        session.commit()

    with pytest.raises(ValueError, match="evidence SHA-256"):
        jobs.get(staged.job_id)
