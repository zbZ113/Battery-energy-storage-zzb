from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.attachments import FeishuAttachmentPolicy
from quanxin_life.integrations.feishu.client import (
    FeishuApiError,
    FeishuErrorKind,
    FeishuHttpResponse,
    FeishuTransportError,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuAnalysisJobWorker,
    FeishuJobDeliveryProgress,
    FeishuJobDeliveryReceipt,
    FeishuJobDispatchReceipt,
    FeishuJobRetryableError,
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
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
    FeishuWorkflowRejected,
)
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 11, 5, 0, tzinfo=UTC)
HEADER = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
)
ROW_1 = b"feishu-data,cell-1,1,0,0.0,3.6,1.0,25.0,1.2,1.1,0.02,true,true\n"
ROW_20 = b"feishu-data,cell-1,20,0,0.0,3.5,1.0,25.0,1.1,1.0,0.03,true,true\n"
VALID_CSV = HEADER + ROW_1 + ROW_20
DUPLICATE_CSV = HEADER + ROW_1 + ROW_1 + ROW_20


class _Queue:
    def __init__(self) -> None:
        self.job_id: str | None = None

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_id = job_id
        return FeishuJobDispatchReceipt(job_id=job_id, task_id="task-job")


class _Client:
    def __init__(self, response: FeishuHttpResponse | Exception) -> None:
        self.response = response
        self.calls = 0

    def download_message_resource_response(self, **_: str) -> FeishuHttpResponse:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _RouteAuthorizer:
    def __init__(self, *, active: bool) -> None:
        self.active = active

    def authorize(self, **_: object) -> None:
        if not self.active:
            raise FeishuWorkflowRejected("MODEL_ROUTE_NOT_ACTIVATED")


class _InputBinder:
    def bind_analysis_input(
        self,
        *,
        requested_analysis_input: Mapping[str, object],
        **_: object,
    ) -> Mapping[str, object]:
        return dict(requested_analysis_input)


class _Delivery:
    def __init__(self, *, fail_success_once: bool = False) -> None:
        self.rejections: list[tuple[str, str, str | None]] = []
        self.successes: list[tuple[str, str, str]] = []
        self.seen_result_card_message_ids: list[str | None] = []
        self.fail_success_once = fail_success_once

    def deliver_rejection(
        self,
        *,
        job: object,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt:
        del checkpoint
        run_id = job.run_id
        self.rejections.append((run_id, reason_code, primary_result_id))
        return FeishuJobDeliveryReceipt(
            bitable_record_id="rec-rejected",
            report_file_key=None,
        )

    def deliver_success(
        self,
        *,
        job: object,
        analysis_result: ToolResult,
        report_result: ToolResult,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt:
        self.seen_result_card_message_ids.append(job.result_card_message_id)
        if self.fail_success_once:
            self.fail_success_once = False
            checkpoint(
                FeishuJobDeliveryProgress(
                    result_card_message_id="om-checkpointed-result-card"
                )
            )
            raise FeishuTransportError("delivery timeout")
        run_id = job.run_id
        self.successes.append(
            (run_id, analysis_result.result_id, report_result.result_id)
        )
        return FeishuJobDeliveryReceipt(
            bitable_record_id="rec-success",
            report_file_key="file-report",
        )


def _event() -> FeishuEventReference:
    return FeishuEventReference(
        event_id="evt-worker",
        event_type="im.message.receive_v1",
        kind=FeishuInboundEventKind.FILE,
        chat_id="oc-worker",
        user_id="ou-worker",
        message_id="om-worker",
        file_key="file-worker",
        file_name="battery.csv",
        receive_id_type="chat_id",
        event_time=NOW,
    )


def _job() -> tuple[SqlAlchemyFeishuJobStore, str]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(session_factory)
    claim = receipts.claim(
        event_id="evt-worker",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    jobs = SqlAlchemyFeishuJobStore(session_factory, lease_seconds=60)
    queue = _Queue()
    routed = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=_event(), claim_token=claim.claim_token)
    assert routed is FeishuEventRouteStatus.ENQUEUED
    assert queue.job_id is not None
    return jobs, queue.job_id


def _registration(job: object, payload_sha256: str) -> CanonicalCsvBatchRegistration:
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="feishu-data",
            cell_id="cell-1",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.2,
            source_uri=(
                f"feishu://message/{job.message_id}/"
                f"resource/{job.file_key}"
            ),
            source_sha256=payload_sha256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="feishu-canonical-v1",
        split_version="feishu-single-cell-v1",
        provenance=(
            ProvenanceRecord(
                source_id=job.file_key,
                source_kind=SourceKind.OBSERVED,
                uri=(
                    f"feishu://message/{job.message_id}/"
                    f"resource/{job.file_key}"
                ),
                sha256=payload_sha256,
                description="Verified Feishu canonical CSV attachment",
                created_at=NOW,
            ),
        ),
    )


def _tool_service(
    *,
    prediction_calls: list[str],
    analysis_rejection_reason: str | None = None,
) -> tuple[ToolInvocationService, AuditLedger]:
    ledger = AuditLedger()
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)

    def predict(value: ValidateBatteryDataToolInput) -> ToolResult:
        prediction_calls.append(value.data_version)
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            tool_version="advanced-rul-prediction-tool-v1",
            model_version="controlled-fake-route-v1",
            data_version=value.data_version,
            feature_version=value.feature_version,
            input_hash=registry.canonical_input_hash(
                StandardToolName.PREDICT_CYCLE_LIFE,
                value,
            ),
            values=(
                {
                    "artifact": {
                        "status": "REJECTED",
                        "rejection_reasons": [analysis_rejection_reason],
                    }
                }
                if analysis_rejection_reason is not None
                else {
                    "artifact": {
                        "cycle_life_prediction": {"predicted_cycle": 24},
                        "derived_remaining_cycles": 4,
                    }
                }
            ),
            warnings=["CONTROLLED_FAKE_TOOLRESULT_NOT_PRODUCTION"],
            provenance=list(value.provenance),
            created_at=NOW,
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="advanced-rul-prediction-tool-v1",
            input_model=ValidateBatteryDataToolInput,
            executor=predict,
        )
    )
    return ToolInvocationService(registry=registry, audit_ledger=ledger), ledger


def _worker(
    *,
    jobs: SqlAlchemyFeishuJobStore,
    client_response: FeishuHttpResponse | Exception,
    route_active: bool,
    prediction_calls: list[str],
    delivery: _Delivery,
    analysis_input_error: Exception | None = None,
    analysis_rejection_reason: str | None = None,
    max_attempts: int = 3,
) -> FeishuAnalysisJobWorker:
    service, ledger = _tool_service(
        prediction_calls=prediction_calls,
        analysis_rejection_reason=analysis_rejection_reason,
    )
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=_RouteAuthorizer(active=route_active),
        input_binder=_InputBinder(),
    )

    def report_result(_job: object, result: ToolResult) -> ToolResult:
        report = ToolResult(
            result_id=str(uuid4()),
            tool_name="generate_audited_report",
            tool_version="audited-report-tool-v1",
            model_version="audited-markdown-v1",
            data_version=result.data_version,
            feature_version=result.feature_version,
            input_hash=sha256(result.result_id.encode()).hexdigest(),
            values={"upstream_result_ids": [result.result_id]},
            warnings=["CONTROLLED_FAKE_REPORT_NOT_PRODUCTION"],
            provenance=result.provenance,
            created_at=NOW,
        )
        return ledger.register_result(report)

    def analysis_input(
        _job: object,
        batch: object,
    ) -> dict[str, object]:
        if analysis_input_error is not None:
            raise analysis_input_error
        return ValidateBatteryDataToolInput(
            records=batch.records,
            data_version=batch.data_version,
            feature_version=batch.feature_config.feature_version,
            provenance=batch.provenance,
            validated_at=NOW,
        ).model_dump(mode="json")

    return FeishuAnalysisJobWorker(
        jobs,
        client=_Client(client_response),
        attachment_policy=FeishuAttachmentPolicy(),
        batch_store=InMemoryVerifiedEarlyCycleBatchStore(),
        registration_resolver=lambda job, attachment, now: _registration(
            job, attachment.sha256
        ),
        analysis_input_factory=analysis_input,
        workflow=workflow,
        result_resolver=ledger,
        report_result_factory=report_result,
        delivery=delivery,
        clock=lambda: NOW,
        heartbeat_interval_seconds=30,
        max_attempts=max_attempts,
    )


def _response(payload: bytes) -> FeishuHttpResponse:
    return FeishuHttpResponse(
        status_code=200,
        headers={
            "content-type": "text/csv; charset=utf-8",
            "content-length": str(len(payload)),
        },
        body=payload,
    )


def test_worker_rejects_disguised_csv_before_registration_or_model() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    predictions: list[str] = []
    worker = _worker(
        jobs=jobs,
        client_response=_response(b"MZnot-a-csv"),
        route_active=True,
        prediction_calls=predictions,
        delivery=delivery,
    )

    status = worker.execute(job_id=job_id)

    assert status is FeishuAnalysisJobStatus.REJECTED
    assert predictions == []
    assert delivery.rejections[0][1] == "ATTACHMENT_REJECTED"
    assert jobs.get(job_id).job_stage == "REJECTED"


def test_worker_stops_after_audited_data_validation_rejection() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    predictions: list[str] = []
    worker = _worker(
        jobs=jobs,
        client_response=_response(DUPLICATE_CSV),
        route_active=True,
        prediction_calls=predictions,
        delivery=delivery,
    )

    status = worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert status is FeishuAnalysisJobStatus.REJECTED
    assert predictions == []
    assert snapshot.validation_result_id is not None
    assert snapshot.analysis_result_id is None
    assert delivery.rejections[0] == (
        snapshot.run_id,
        "DATA_VALIDATION_BLOCKED",
        snapshot.validation_result_id,
    )


def test_worker_rejects_inactive_route_before_prediction() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    predictions: list[str] = []
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=False,
        prediction_calls=predictions,
        delivery=delivery,
    )

    status = worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert status is FeishuAnalysisJobStatus.REJECTED
    assert predictions == []
    assert snapshot.validation_result_id is not None
    assert delivery.rejections[0][1] == "MODEL_ROUTE_NOT_ACTIVATED"


def test_worker_success_checkpoints_audited_results_before_delivery() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    predictions: list[str] = []
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=True,
        prediction_calls=predictions,
        delivery=delivery,
    )

    status = worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert status is FeishuAnalysisJobStatus.SUCCEEDED
    assert predictions == ["feishu-canonical-v1"]
    assert snapshot.validation_result_id is not None
    assert snapshot.analysis_result_id is not None
    assert snapshot.report_result_id is not None
    assert snapshot.cell_reference == "cell-1"
    assert snapshot.report_file_key == "file-report"
    assert snapshot.bitable_record_id == "rec-success"
    assert delivery.successes == [
        (snapshot.run_id, snapshot.analysis_result_id, snapshot.report_result_id)
    ]


def test_worker_marks_transport_timeout_retryable_without_delivery() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    worker = _worker(
        jobs=jobs,
        client_response=FeishuTransportError("timeout"),
        route_active=True,
        prediction_calls=[],
        delivery=delivery,
    )

    with pytest.raises(FeishuJobRetryableError, match="DOWNLOAD_RETRYABLE"):
        worker.execute(job_id=job_id)

    assert jobs.get(job_id).job_status is FeishuAnalysisJobStatus.RETRYABLE
    assert delivery.rejections == []


def test_worker_marks_feishu_rate_limit_retryable_without_delivery() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    worker = _worker(
        jobs=jobs,
        client_response=FeishuApiError(
            kind=FeishuErrorKind.RATE_LIMIT,
            status_code=429,
            api_code=99991400,
        ),
        route_active=True,
        prediction_calls=[],
        delivery=delivery,
    )

    with pytest.raises(FeishuJobRetryableError, match="DOWNLOAD_RETRYABLE"):
        worker.execute(job_id=job_id)

    assert jobs.get(job_id).job_status is FeishuAnalysisJobStatus.RETRYABLE


def test_worker_restart_resumes_checkpointed_results_without_rerunning_tool() -> None:
    jobs, job_id = _job()
    delivery = _Delivery(fail_success_once=True)
    predictions: list[str] = []
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=True,
        prediction_calls=predictions,
        delivery=delivery,
    )

    with pytest.raises(FeishuJobRetryableError, match="DELIVERY_RETRYABLE"):
        worker.execute(job_id=job_id)
    checkpoint = jobs.get(job_id)
    assert checkpoint.analysis_result_id is not None
    assert checkpoint.report_result_id is not None
    assert checkpoint.result_card_message_id == "om-checkpointed-result-card"

    status = worker.execute(job_id=job_id)

    assert status is FeishuAnalysisJobStatus.SUCCEEDED
    assert predictions == ["feishu-canonical-v1"]
    assert delivery.seen_result_card_message_ids == [
        None,
        "om-checkpointed-result-card",
    ]


def test_worker_rejects_permanent_message_or_file_reference_error() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    worker = _worker(
        jobs=jobs,
        client_response=FeishuApiError(
            kind=FeishuErrorKind.PERMANENT,
            status_code=404,
            api_code=234001,
        ),
        route_active=True,
        prediction_calls=[],
        delivery=delivery,
    )

    status = worker.execute(job_id=job_id)

    assert status is FeishuAnalysisJobStatus.REJECTED
    assert delivery.rejections[0][1] == "ATTACHMENT_RESOURCE_NOT_FOUND"


def test_worker_persists_unexpected_error_as_retryable_before_redelivery() -> None:
    jobs, job_id = _job()
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=True,
        prediction_calls=[],
        delivery=_Delivery(),
        analysis_input_error=RuntimeError("unexpected input factory failure"),
    )

    with pytest.raises(FeishuJobRetryableError, match="UNEXPECTED_WORKER_ERROR"):
        worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert snapshot.job_status is FeishuAnalysisJobStatus.RETRYABLE
    assert snapshot.job_last_error_code == "UNEXPECTED_WORKER_ERROR"


def test_worker_terminally_fails_unexpected_error_at_attempt_limit() -> None:
    jobs, job_id = _job()
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=True,
        prediction_calls=[],
        delivery=_Delivery(),
        analysis_input_error=RuntimeError("unexpected input factory failure"),
        max_attempts=1,
    )

    status = worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert status is FeishuAnalysisJobStatus.FAILED
    assert snapshot.job_status is FeishuAnalysisJobStatus.FAILED
    assert snapshot.job_last_error_code == "UNEXPECTED_WORKER_ERROR"


def test_worker_delivers_audited_analysis_rejection_without_success_report() -> None:
    jobs, job_id = _job()
    delivery = _Delivery()
    worker = _worker(
        jobs=jobs,
        client_response=_response(VALID_CSV),
        route_active=True,
        prediction_calls=[],
        delivery=delivery,
        analysis_rejection_reason="ROUTE_NOT_ACTIVATED",
    )

    status = worker.execute(job_id=job_id)

    snapshot = jobs.get(job_id)
    assert status is FeishuAnalysisJobStatus.REJECTED
    assert snapshot.analysis_result_id is not None
    assert snapshot.report_result_id is None
    assert delivery.rejections == [
        (snapshot.run_id, "ROUTE_NOT_ACTIVATED", snapshot.analysis_result_id)
    ]
