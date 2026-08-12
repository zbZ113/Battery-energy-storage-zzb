from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.integrations.feishu.events import FeishuReceiptClaimStatus
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuJobClaimStatus,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
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
        result_card_message_id="om-result-card",
        report_file_key="file-report",
        report_message_id="om-report-file",
        report_card_message_id="om-report-card",
        bitable_record_id="rec-run",
        updated_at=NOW,
    )

    snapshot = jobs.get(job_id)
    assert snapshot.scenario_image_key == "img-result"
    assert snapshot.result_card_message_id == "om-result-card"
    assert snapshot.report_file_key == "file-report"
    assert snapshot.report_message_id == "om-report-file"
    assert snapshot.report_card_message_id == "om-report-card"
    assert snapshot.bitable_record_id == "rec-run"
