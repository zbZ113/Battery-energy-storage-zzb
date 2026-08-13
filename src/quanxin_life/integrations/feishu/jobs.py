"""Durable Feishu analysis jobs stored on the existing receipt identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from threading import Event, Thread
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.application.battery_csv_mapping import (
    BatteryCsvMappingError,
    BatteryCsvMappingRejectionEvidence,
    BatteryCsvMappingResult,
    BatteryCsvMappingSuccessEvidence,
)
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    VerifiedEarlyCycleBatchStore,
)
from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.persistence.database import SessionFactory
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

from .attachments import (
    FeishuAttachmentError,
    FeishuAttachmentPolicy,
    VerifiedFeishuAttachment,
)
from .bitable import FeishuBitableWriter
from .cards import (
    AuditedCardBuilder,
    AuditedResultAuthorizer,
    FeishuCardStatus,
    build_status_card,
)
from .client import (
    FeishuApiError,
    FeishuClientError,
    FeishuErrorKind,
    FeishuHttpResponse,
    FeishuTransportError,
)
from .delivery_contract import (
    FeishuDeliveryResultContractError,
    validate_delivery_results,
)
from .events import FeishuReceiptClaimStatus
from .report_delivery import (
    FeishuReportDelivery,
    FeishuReportDeliveryReceipt,
)
from .routing import FeishuEventReference, FeishuInboundEventKind
from .scenario_plot import FeishuScenarioPlotArtifact
from .workflow import FeishuAnalysisTask, FeishuAnalysisWorkflow, FeishuWorkflowRejected


class FeishuAnalysisJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYABLE = "RETRYABLE"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class FeishuAnalysisJobOrigin(StrEnum):
    FEISHU = "FEISHU"
    AILY = "AILY"


class FeishuAnalysisJobStage(StrEnum):
    RECEIVED = "RECEIVED"
    DOWNLOADING = "DOWNLOADING"
    VALIDATING_FILE = "VALIDATING_FILE"
    REGISTERING_DATA = "REGISTERING_DATA"
    VALIDATING_DATA = "VALIDATING_DATA"
    WAITING_FOR_ROUTE = "WAITING_FOR_ROUTE"
    RUNNING_TOOL = "RUNNING_TOOL"
    DELIVERING_RESULT = "DELIVERING_RESULT"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


_SCENARIO_TASKS = frozenset(
    {
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    }
)


class FeishuJobClaimStatus(StrEnum):
    CLAIMED = "CLAIMED"
    IN_PROGRESS = "IN_PROGRESS"
    TERMINAL = "TERMINAL"


class FeishuJobOwnershipError(RuntimeError):
    """Raised when a stale worker attempts to mutate a fenced job."""


class FeishuJobBusyError(RuntimeError):
    """Raised so Celery retries while another worker owns a live job lease."""


class FeishuJobRetryableError(RuntimeError):
    """Raised after durable retry state is written so Celery redelivers the job."""


class FeishuProjectModelRejected(RuntimeError):
    """Machine-readable refusal from project-bound Feishu model orchestration."""

    def __init__(self, reason_code: str, message: str | None = None) -> None:
        super().__init__(message or reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class FeishuJobDispatchReceipt:
    job_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class FeishuJobClaim:
    status: FeishuJobClaimStatus
    job_id: str
    claim_token: str | None = None
    attempt: int | None = None


@dataclass(frozen=True, slots=True)
class FeishuAnalysisJobRecord:
    job_id: str
    run_id: str
    event_id: str
    task_type: FeishuAnalysisTask
    job_status: FeishuAnalysisJobStatus
    job_stage: str
    message_id: str | None
    file_key: str | None
    file_name: str | None
    chat_id: str | None
    sender_id: str | None
    receive_id_type: str | None
    event_time: datetime
    scenario_context_id: str | None
    record_batch_id: str | None
    cell_reference: str | None
    input_file_sha256: str | None
    csv_mapping_status: str | None
    csv_mapping_evidence: dict[str, object] | None
    csv_mapping_evidence_sha256: str | None
    validation_result_id: str | None
    prepared_input_result_id: str | None
    analysis_result_id: str | None
    report_result_id: str | None
    scenario_image_key: str | None
    analysis_image_key: str | None
    analysis_image_renderer_version: str | None
    analysis_image_sha256: str | None
    result_card_message_id: str | None
    report_file_key: str | None
    report_message_id: str | None
    report_card_message_id: str | None
    bitable_record_id: str | None
    job_last_error_code: str | None
    job_attempt_count: int
    job_created_at: datetime
    job_updated_at: datetime
    job_completed_at: datetime | None
    job_origin: FeishuAnalysisJobOrigin = FeishuAnalysisJobOrigin.FEISHU
    job_request_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FeishuJobDeliveryReceipt:
    bitable_record_id: str | None
    report_file_key: str | None


@dataclass(frozen=True, slots=True)
class FeishuProjectModelExecution:
    record_batch_id: str
    prepared_input_result_id: str
    analysis_result: ToolResult
    report_result: ToolResult
    slots_committed: bool = False


@dataclass(frozen=True, slots=True)
class FeishuJobDeliveryProgress:
    scenario_image_key: str | None = None
    analysis_image_key: str | None = None
    analysis_image_renderer_version: str | None = None
    analysis_image_sha256: str | None = None
    result_card_message_id: str | None = None
    report_file_key: str | None = None
    report_message_id: str | None = None
    report_card_message_id: str | None = None
    bitable_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class _StagedJob:
    job_id: str
    dispatched: bool


class FeishuJobQueue(Protocol):
    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt: ...


class FeishuResourceClient(Protocol):
    def download_message_resource_response(
        self,
        *,
        message_id: str,
        file_key: str,
        resource_type: str,
    ) -> FeishuHttpResponse: ...


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class FeishuProjectModelExecutorPort(Protocol):
    def execute(
        self,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
        *,
        claim_token: str | None = None,
    ) -> FeishuProjectModelExecution: ...


class BatteryCsvNormalizerPort(Protocol):
    def normalize(self, payload: bytes) -> BatteryCsvMappingResult: ...


class FeishuScenarioInputResolver(Protocol):
    def resolve_data_batch_id(self, scenario_context_id: str) -> str: ...

    def resolve_analysis_input(
        self,
        *,
        scenario_context_id: str,
        task: FeishuAnalysisTask,
        run_id: str,
    ) -> Mapping[str, object]: ...


class FeishuJobDelivery(Protocol):
    def deliver_rejection(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt: ...

    def deliver_success(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        analysis_result: ToolResult,
        report_result: ToolResult,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt: ...


class OriginAwareAnalysisJobDelivery:
    """Route durable job delivery by its persisted integration origin."""

    def __init__(
        self,
        *,
        feishu_delivery: FeishuJobDelivery,
        aily_delivery: FeishuJobDelivery,
    ) -> None:
        self._feishu_delivery = feishu_delivery
        self._aily_delivery = aily_delivery

    def deliver_rejection(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt:
        return self._delivery_for(job).deliver_rejection(
            job=job,
            reason_code=reason_code,
            primary_result_id=primary_result_id,
            checkpoint=checkpoint,
        )

    def deliver_success(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        analysis_result: ToolResult,
        report_result: ToolResult,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None],
    ) -> FeishuJobDeliveryReceipt:
        return self._delivery_for(job).deliver_success(
            job=job,
            analysis_result=analysis_result,
            report_result=report_result,
            checkpoint=checkpoint,
        )

    def _delivery_for(self, job: FeishuAnalysisJobRecord) -> FeishuJobDelivery:
        if job.job_origin is FeishuAnalysisJobOrigin.FEISHU:
            return self._feishu_delivery
        if job.job_origin is FeishuAnalysisJobOrigin.AILY:
            return self._aily_delivery
        raise ValueError("analysis job origin is unsupported")


class FeishuDeliveryClient(Protocol):
    def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: dict[str, object],
    ) -> dict[str, object]: ...

    def upload_image(
        self,
        *,
        payload: bytes,
        content_type: str,
        filename: str,
    ) -> dict[str, object]: ...


class FeishuScenarioPlotRenderer(Protocol):
    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact: ...


class FeishuAnalysisPlotRenderer(Protocol):
    renderer_version: str

    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact: ...


class SqlAlchemyFeishuJobStore:
    """Persist sanitized references and a fenced worker lease on receipt rows."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        lease_seconds: int = 300,
    ) -> None:
        if lease_seconds < 1 or lease_seconds > 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        self._session_factory = session_factory
        self._lease = timedelta(seconds=lease_seconds)

    def stage(
        self,
        *,
        event: FeishuEventReference,
        claim_token: str,
        task: FeishuAnalysisTask,
        staged_at: datetime,
    ) -> _StagedJob:
        now = _utc(staged_at)
        scenario_context_id = _scenario_context_reference(event=event, task=task)
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.event_id == event.event_id)
                .with_for_update()
            )
            if row is None:
                raise ValueError("Feishu receipt was not found")
            if (
                row.status != FeishuReceiptClaimStatus.IN_PROGRESS.value
                or row.claim_token != claim_token
                or row.lease_expires_at is None
                or _database_utc(row.lease_expires_at) <= now
                or row.event_type != event.event_type
            ):
                raise FeishuJobOwnershipError("Feishu receipt claim is stale")
            if row.job_id is not None:
                self._require_same_reference(row, event=event, task=task)
                session.rollback()
                return _StagedJob(
                    job_id=row.job_id,
                    dispatched=row.job_task_id is not None,
                )
            job_id = str(uuid4())
            row.job_id = job_id
            row.job_origin = FeishuAnalysisJobOrigin.FEISHU.value
            row.job_request_sha256 = None
            row.run_id = job_id
            row.job_status = FeishuAnalysisJobStatus.PENDING.value
            row.job_stage = FeishuAnalysisJobStage.RECEIVED.value
            row.task_type = task.value
            row.message_id = event.message_id
            row.file_key = event.file_key
            row.file_name = event.file_name
            row.chat_id = event.chat_id
            row.sender_id = event.user_id
            row.receive_id_type = event.receive_id_type
            row.event_time = event.event_time or now
            row.scenario_context_id = scenario_context_id
            row.job_attempt_count = 0
            row.job_created_at = now
            row.job_updated_at = now
            session.commit()
            return _StagedJob(job_id=job_id, dispatched=False)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def stage_aily(
        self,
        *,
        task: FeishuAnalysisTask,
        scenario_context_id: str,
        staged_at: datetime,
    ) -> _StagedJob:
        if task not in _SCENARIO_TASKS:
            raise ValueError("Aily durable analysis supports only scenario tasks")
        context_id = validate_feishu_job_id(scenario_context_id)
        now = _utc(staged_at)
        request_sha256 = sha256_canonical(
            {
                "job_origin": FeishuAnalysisJobOrigin.AILY.value,
                "task_type": task.value,
                "scenario_context_id": context_id,
            }
        )
        session = self._session_factory()
        try:
            existing = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_request_sha256 == request_sha256)
                .with_for_update()
            )
            if existing is not None:
                staged = self._existing_aily_job(
                    existing,
                    task=task,
                    scenario_context_id=context_id,
                    request_sha256=request_sha256,
                )
                session.rollback()
                return staged
            job_id = str(uuid4())
            session.add(
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id=f"aily:{request_sha256}",
                    event_type="aily.analysis_task.create_v1",
                    payload_sha256=request_sha256,
                    status=FeishuReceiptClaimStatus.PROCESSED.value,
                    attempt_count=1,
                    received_at=now,
                    processed_at=now,
                    job_id=job_id,
                    job_origin=FeishuAnalysisJobOrigin.AILY.value,
                    job_request_sha256=request_sha256,
                    job_status=FeishuAnalysisJobStatus.PENDING.value,
                    job_stage=FeishuAnalysisJobStage.RECEIVED.value,
                    task_type=task.value,
                    run_id=job_id,
                    event_time=now,
                    scenario_context_id=context_id,
                    job_attempt_count=0,
                    job_created_at=now,
                    job_updated_at=now,
                )
            )
            session.commit()
            return _StagedJob(job_id=job_id, dispatched=False)
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(FeishuEventReceipt).where(
                    FeishuEventReceipt.job_request_sha256 == request_sha256
                )
            )
            if existing is None:
                raise
            return self._existing_aily_job(
                existing,
                task=task,
                scenario_context_id=context_id,
                request_sha256=request_sha256,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def stage_replay(
        self,
        *,
        source_job_id: str,
        replay_key: str,
        staged_at: datetime,
    ) -> _StagedJob:
        """Create one new job from an explicitly replayable terminal rejection."""

        checked_source_job_id = validate_feishu_job_id(source_job_id)
        checked_replay_key = _replay_key(replay_key)
        now = _utc(staged_at)
        replay_sha256 = sha256_canonical(
            {
                "schema_version": "feishu-analysis-job-replay-v1",
                "source_job_id": checked_source_job_id,
                "replay_key": checked_replay_key,
            }
        )
        replay_event_id = f"replay:{checked_source_job_id}:{replay_sha256}"
        session = self._session_factory()
        try:
            existing = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.event_id == replay_event_id)
                .with_for_update()
            )
            if existing is not None:
                staged = self._existing_replay_job(
                    existing,
                    replay_sha256=replay_sha256,
                )
                session.rollback()
                return staged
            source = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_id == checked_source_job_id)
                .with_for_update()
            )
            self._require_replayable_route_rejection(source)
            assert source is not None
            job_id = str(uuid4())
            session.add(
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id=replay_event_id,
                    event_type="feishu.analysis_job.replay_v1",
                    payload_sha256=replay_sha256,
                    status=FeishuReceiptClaimStatus.PROCESSED.value,
                    attempt_count=1,
                    received_at=now,
                    processed_at=now,
                    job_id=job_id,
                    job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                    job_request_sha256=None,
                    job_status=FeishuAnalysisJobStatus.PENDING.value,
                    job_stage=FeishuAnalysisJobStage.RECEIVED.value,
                    task_type=source.task_type,
                    run_id=job_id,
                    message_id=source.message_id,
                    file_key=source.file_key,
                    file_name=source.file_name,
                    chat_id=source.chat_id,
                    sender_id=source.sender_id,
                    receive_id_type=source.receive_id_type,
                    event_time=source.event_time,
                    scenario_context_id=None,
                    job_attempt_count=0,
                    job_created_at=now,
                    job_updated_at=now,
                )
            )
            session.commit()
            return _StagedJob(job_id=job_id, dispatched=False)
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(FeishuEventReceipt).where(
                    FeishuEventReceipt.event_id == replay_event_id
                )
            )
            if existing is None:
                raise
            return self._existing_replay_job(
                existing,
                replay_sha256=replay_sha256,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_dispatched(
        self,
        *,
        job_id: str,
        task_id: str,
        dispatched_at: datetime,
    ) -> None:
        now = _utc(dispatched_at)
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_id == validate_feishu_job_id(job_id))
                .with_for_update()
            )
            if row is None:
                raise ValueError("Feishu analysis job was not found")
            checked_task_id = task_id.strip() if isinstance(task_id, str) else ""
            if not checked_task_id or len(checked_task_id) > 200:
                raise ValueError("task_id is invalid")
            if row.job_task_id is not None and row.job_task_id != checked_task_id:
                raise FeishuJobOwnershipError("Feishu job owns another dispatch")
            row.job_task_id = checked_task_id
            row.job_updated_at = now
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def claim(self, *, job_id: str, claimed_at: datetime) -> FeishuJobClaim:
        checked_job_id = validate_feishu_job_id(job_id)
        now = _utc(claimed_at)
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_id == checked_job_id)
                .with_for_update()
            )
            if row is None or row.job_status is None:
                raise ValueError("Feishu analysis job was not found")
            status = FeishuAnalysisJobStatus(row.job_status)
            if status in {
                FeishuAnalysisJobStatus.SUCCEEDED,
                FeishuAnalysisJobStatus.REJECTED,
                FeishuAnalysisJobStatus.FAILED,
            }:
                session.rollback()
                return FeishuJobClaim(
                    status=FeishuJobClaimStatus.TERMINAL,
                    job_id=checked_job_id,
                )
            if (
                status is FeishuAnalysisJobStatus.RUNNING
                and row.job_lease_expires_at is not None
                and _database_utc(row.job_lease_expires_at) > now
            ):
                session.rollback()
                return FeishuJobClaim(
                    status=FeishuJobClaimStatus.IN_PROGRESS,
                    job_id=checked_job_id,
                )
            token = uuid4().hex + uuid4().hex
            row.job_status = FeishuAnalysisJobStatus.RUNNING.value
            row.job_claim_token = token
            row.job_attempt_count += 1
            row.job_lease_expires_at = now + self._lease
            row.job_updated_at = now
            attempt = row.job_attempt_count
            session.commit()
            return FeishuJobClaim(
                status=FeishuJobClaimStatus.CLAIMED,
                job_id=checked_job_id,
                claim_token=token,
                attempt=attempt,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def renew_claim(
        self,
        *,
        job_id: str,
        claim_token: str,
        renewed_at: datetime,
    ) -> datetime:
        checked_job_id = validate_feishu_job_id(job_id)
        checked_token = _claim_token(claim_token)
        now = _utc(renewed_at)
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_id == checked_job_id)
                .with_for_update()
            )
            if (
                row is None
                or row.job_status != FeishuAnalysisJobStatus.RUNNING.value
                or row.job_claim_token != checked_token
                or row.job_lease_expires_at is None
                or _database_utc(row.job_lease_expires_at) <= now
            ):
                raise FeishuJobOwnershipError("Feishu analysis job claim is stale")
            expiry = now + self._lease
            row.job_lease_expires_at = expiry
            row.job_updated_at = now
            session.commit()
            return expiry
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, job_id: str) -> FeishuAnalysisJobRecord:
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt).where(
                    FeishuEventReceipt.job_id == validate_feishu_job_id(job_id)
                )
            )
            if row is None:
                raise ValueError("Feishu analysis job was not found")
            return self._record(row)
        finally:
            session.close()

    def update_stage(
        self,
        *,
        job_id: str,
        claim_token: str,
        stage: FeishuAnalysisJobStage,
        updated_at: datetime,
    ) -> None:
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values={"job_stage": stage.value},
        )

    def checkpoint_batch(
        self,
        *,
        job_id: str,
        claim_token: str,
        record_batch_id: str,
        cell_reference: str,
        input_file_sha256: str,
        csv_mapping_evidence: Mapping[str, object] | None = None,
        csv_mapping_evidence_sha256: str | None = None,
        updated_at: datetime,
    ) -> None:
        values: dict[str, object] = {
            "record_batch_id": record_batch_id,
            "cell_reference": cell_reference,
            "input_file_sha256": input_file_sha256,
        }
        if csv_mapping_evidence is not None or csv_mapping_evidence_sha256 is not None:
            if csv_mapping_evidence is None or csv_mapping_evidence_sha256 is None:
                raise ValueError("CSV mapping batch evidence is incomplete")
            detached = BatteryCsvMappingSuccessEvidence.model_validate(
                csv_mapping_evidence
            ).model_dump(mode="json")
            if sha256_canonical(detached) != csv_mapping_evidence_sha256:
                raise ValueError("CSV mapping evidence SHA-256 does not match")
            values.update(
                {
                    "csv_mapping_status": "MAPPED",
                    "csv_mapping_evidence_json": detached,
                    "csv_mapping_evidence_sha256": csv_mapping_evidence_sha256,
                }
            )
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values=values,
        )

    def checkpoint_csv_mapping(
        self,
        *,
        job_id: str,
        claim_token: str,
        status: str,
        evidence: Mapping[str, object],
        evidence_sha256: str,
        updated_at: datetime,
    ) -> None:
        if status not in {"MAPPED", "REJECTED"}:
            raise ValueError("CSV mapping status is invalid")
        try:
            if status == "MAPPED":
                detached = BatteryCsvMappingSuccessEvidence.model_validate(
                    evidence
                ).model_dump(mode="json")
            else:
                detached = BatteryCsvMappingRejectionEvidence.model_validate(
                    evidence
                ).model_dump(mode="json")
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("CSV mapping evidence contract is invalid") from exc
        if sha256_canonical(detached) != evidence_sha256:
            raise ValueError("CSV mapping evidence SHA-256 does not match")
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values={
                "csv_mapping_status": status,
                "csv_mapping_evidence_json": detached,
                "csv_mapping_evidence_sha256": evidence_sha256,
            },
        )

    def checkpoint_results(
        self,
        *,
        job_id: str,
        claim_token: str,
        validation_result_id: str | None = None,
        prepared_input_result_id: str | None = None,
        analysis_result_id: str | None = None,
        report_result_id: str | None = None,
        updated_at: datetime,
    ) -> None:
        values: dict[str, object] = {}
        if validation_result_id is not None:
            values["validation_result_id"] = validation_result_id
        if prepared_input_result_id is not None:
            values["prepared_input_result_id"] = prepared_input_result_id
        if analysis_result_id is not None:
            values["analysis_result_id"] = analysis_result_id
        if report_result_id is not None:
            values["report_result_id"] = report_result_id
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values=values,
        )

    def checkpoint_delivery(
        self,
        *,
        job_id: str,
        claim_token: str,
        updated_at: datetime,
        scenario_image_key: str | None = None,
        analysis_image_key: str | None = None,
        analysis_image_renderer_version: str | None = None,
        analysis_image_sha256: str | None = None,
        result_card_message_id: str | None = None,
        report_file_key: str | None = None,
        report_message_id: str | None = None,
        report_card_message_id: str | None = None,
        bitable_record_id: str | None = None,
    ) -> None:
        references = {
            "scenario_image_key": scenario_image_key,
            "analysis_image_key": analysis_image_key,
            "result_card_message_id": result_card_message_id,
            "report_file_key": report_file_key,
            "report_message_id": report_message_id,
            "report_card_message_id": report_card_message_id,
            "bitable_record_id": bitable_record_id,
        }
        image_provenance = (
            analysis_image_key,
            analysis_image_renderer_version,
            analysis_image_sha256,
        )
        if any(value is not None for value in image_provenance):
            if any(value is None for value in image_provenance):
                raise ValueError("analysis image provenance is incomplete")
            assert analysis_image_renderer_version is not None
            assert analysis_image_sha256 is not None
            _delivery_reference(
                analysis_image_renderer_version,
                field_name="analysis_image_renderer_version",
            )
            _sha256_reference(
                analysis_image_sha256,
                field_name="analysis_image_sha256",
            )
            references.update(
                {
                    "analysis_image_renderer_version": analysis_image_renderer_version,
                    "analysis_image_sha256": analysis_image_sha256,
                }
            )
        values = {
            field_name: _delivery_reference(value, field_name=field_name)
            for field_name, value in references.items()
            if value is not None
        }
        if not values:
            raise ValueError("delivery checkpoint requires at least one reference")
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values=values,
        )

    def mark_retryable(
        self,
        *,
        job_id: str,
        claim_token: str,
        error_code: str,
        updated_at: datetime,
    ) -> None:
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=updated_at,
            values={
                "job_status": FeishuAnalysisJobStatus.RETRYABLE.value,
                "job_last_error_code": _error_code(error_code),
                "job_claim_token": None,
                "job_lease_expires_at": None,
            },
        )

    def finish(
        self,
        *,
        job_id: str,
        claim_token: str,
        status: FeishuAnalysisJobStatus,
        stage: FeishuAnalysisJobStage,
        error_code: str | None,
        delivery: FeishuJobDeliveryReceipt | None,
        completed_at: datetime,
    ) -> None:
        if status not in {
            FeishuAnalysisJobStatus.SUCCEEDED,
            FeishuAnalysisJobStatus.REJECTED,
            FeishuAnalysisJobStatus.FAILED,
        }:
            raise ValueError("finish requires a terminal Feishu job status")
        values: dict[str, object] = {
            "job_status": status.value,
            "job_stage": stage.value,
            "job_last_error_code": (
                _error_code(error_code) if error_code is not None else None
            ),
            "job_claim_token": None,
            "job_lease_expires_at": None,
            "job_completed_at": _utc(completed_at),
        }
        if delivery is not None:
            values["bitable_record_id"] = delivery.bitable_record_id
            values["report_file_key"] = delivery.report_file_key
        self._mutate_owned(
            job_id=job_id,
            claim_token=claim_token,
            updated_at=completed_at,
            values=values,
        )

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool:
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt).where(FeishuEventReceipt.run_id == run_id)
            )
            if row is None:
                return False
            return result_id in {
                row.validation_result_id,
                row.prepared_input_result_id,
                row.analysis_result_id,
                row.report_result_id,
            }
        finally:
            session.close()

    def _mutate_owned(
        self,
        *,
        job_id: str,
        claim_token: str,
        updated_at: datetime,
        values: Mapping[str, object],
    ) -> None:
        checked_job_id = validate_feishu_job_id(job_id)
        checked_token = _claim_token(claim_token)
        now = _utc(updated_at)
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuEventReceipt)
                .where(FeishuEventReceipt.job_id == checked_job_id)
                .with_for_update()
            )
            if (
                row is None
                or row.job_status != FeishuAnalysisJobStatus.RUNNING.value
                or row.job_claim_token != checked_token
                or row.job_lease_expires_at is None
                or _database_utc(row.job_lease_expires_at) <= now
            ):
                raise FeishuJobOwnershipError("Feishu analysis job claim is stale")
            for field_name, value in values.items():
                setattr(row, field_name, value)
            row.job_updated_at = now
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _record(row: FeishuEventReceipt) -> FeishuAnalysisJobRecord:
        required = {
            "job_id": row.job_id,
            "run_id": row.run_id,
            "task_type": row.task_type,
            "job_status": row.job_status,
            "job_stage": row.job_stage,
            "event_time": row.event_time,
            "job_created_at": row.job_created_at,
            "job_updated_at": row.job_updated_at,
        }
        if any(value is None for value in required.values()):
            raise ValueError("Feishu analysis job is incomplete")
        assert row.job_id is not None
        assert row.run_id is not None
        assert row.task_type is not None
        assert row.job_status is not None
        assert row.job_stage is not None
        assert row.event_time is not None
        assert row.job_created_at is not None
        assert row.job_updated_at is not None
        mapping_fields = (
            row.csv_mapping_status,
            row.csv_mapping_evidence_json,
            row.csv_mapping_evidence_sha256,
        )
        if any(item is not None for item in mapping_fields) and any(
            item is None for item in mapping_fields
        ):
            raise ValueError("Feishu CSV mapping evidence is incomplete")
        mapping_evidence = (
            dict(row.csv_mapping_evidence_json)
            if row.csv_mapping_evidence_json is not None
            else None
        )
        if mapping_evidence is not None:
            if row.csv_mapping_status not in {"MAPPED", "REJECTED"}:
                raise ValueError("Feishu CSV mapping status is invalid")
            evidence_model = (
                BatteryCsvMappingSuccessEvidence
                if row.csv_mapping_status == "MAPPED"
                else BatteryCsvMappingRejectionEvidence
            )
            mapping_evidence = evidence_model.model_validate(
                mapping_evidence
            ).model_dump(mode="json")
            if sha256_canonical(mapping_evidence) != row.csv_mapping_evidence_sha256:
                raise ValueError("Feishu CSV mapping evidence SHA-256 does not match")
        return FeishuAnalysisJobRecord(
            job_id=row.job_id,
            run_id=row.run_id,
            event_id=row.event_id,
            task_type=FeishuAnalysisTask(row.task_type),
            job_status=FeishuAnalysisJobStatus(row.job_status),
            job_stage=row.job_stage,
            message_id=row.message_id,
            file_key=row.file_key,
            file_name=row.file_name,
            chat_id=row.chat_id,
            sender_id=row.sender_id,
            receive_id_type=row.receive_id_type,
            event_time=_database_utc(row.event_time),
            scenario_context_id=row.scenario_context_id,
            record_batch_id=row.record_batch_id,
            cell_reference=row.cell_reference,
            input_file_sha256=row.input_file_sha256,
            csv_mapping_status=row.csv_mapping_status,
            csv_mapping_evidence=mapping_evidence,
            csv_mapping_evidence_sha256=row.csv_mapping_evidence_sha256,
            validation_result_id=row.validation_result_id,
            prepared_input_result_id=row.prepared_input_result_id,
            analysis_result_id=row.analysis_result_id,
            report_result_id=row.report_result_id,
            scenario_image_key=row.scenario_image_key,
            analysis_image_key=row.analysis_image_key,
            analysis_image_renderer_version=row.analysis_image_renderer_version,
            analysis_image_sha256=row.analysis_image_sha256,
            result_card_message_id=row.result_card_message_id,
            report_file_key=row.report_file_key,
            report_message_id=row.report_message_id,
            report_card_message_id=row.report_card_message_id,
            bitable_record_id=row.bitable_record_id,
            job_last_error_code=row.job_last_error_code,
            job_attempt_count=row.job_attempt_count,
            job_created_at=_database_utc(row.job_created_at),
            job_updated_at=_database_utc(row.job_updated_at),
            job_completed_at=(
                _database_utc(row.job_completed_at)
                if row.job_completed_at is not None
                else None
            ),
            job_origin=(
                FeishuAnalysisJobOrigin(row.job_origin)
                if row.job_origin is not None
                else FeishuAnalysisJobOrigin.FEISHU
            ),
            job_request_sha256=row.job_request_sha256,
        )

    @staticmethod
    def _existing_aily_job(
        row: FeishuEventReceipt,
        *,
        task: FeishuAnalysisTask,
        scenario_context_id: str,
        request_sha256: str,
    ) -> _StagedJob:
        if (
            row.job_id is None
            or row.job_origin != FeishuAnalysisJobOrigin.AILY.value
            or row.job_request_sha256 != request_sha256
            or row.task_type != task.value
            or row.scenario_context_id != scenario_context_id
        ):
            raise FeishuJobOwnershipError(
                "Aily request conflicts with its persisted sanitized reference"
            )
        return _StagedJob(
            job_id=row.job_id,
            dispatched=row.job_task_id is not None,
        )

    @staticmethod
    def _existing_replay_job(
        row: FeishuEventReceipt,
        *,
        replay_sha256: str,
    ) -> _StagedJob:
        if (
            row.job_id is None
            or row.event_type != "feishu.analysis_job.replay_v1"
            or row.payload_sha256 != replay_sha256
            or row.job_origin != FeishuAnalysisJobOrigin.FEISHU.value
        ):
            raise FeishuJobOwnershipError(
                "Feishu replay conflicts with its persisted sanitized reference"
            )
        return _StagedJob(
            job_id=row.job_id,
            dispatched=row.job_task_id is not None,
        )

    @staticmethod
    def _require_replayable_route_rejection(
        row: FeishuEventReceipt | None,
    ) -> None:
        required_references = (
            row.message_id if row is not None else None,
            row.file_key if row is not None else None,
            row.file_name if row is not None else None,
            row.chat_id if row is not None else None,
            row.sender_id if row is not None else None,
            row.receive_id_type if row is not None else None,
            row.event_time if row is not None else None,
        )
        if (
            row is None
            or row.job_origin != FeishuAnalysisJobOrigin.FEISHU.value
            or row.job_status != FeishuAnalysisJobStatus.REJECTED.value
            or row.job_stage != FeishuAnalysisJobStage.REJECTED.value
            or row.job_last_error_code != "MODEL_ROUTE_NOT_ACTIVATED"
            or row.task_type != FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value
            or row.scenario_context_id is not None
            or row.job_completed_at is None
            or any(value is None for value in required_references)
        ):
            raise ValueError(
                "replay requires a completed MODEL_ROUTE_NOT_ACTIVATED Feishu file job"
            )

    @staticmethod
    def _require_same_reference(
        row: FeishuEventReceipt,
        *,
        event: FeishuEventReference,
        task: FeishuAnalysisTask,
    ) -> None:
        expected = (
            event.message_id,
            event.file_key,
            event.file_name,
            event.chat_id,
            event.user_id,
            event.receive_id_type,
            event.action_value.get("scenario_context_id"),
            task.value,
        )
        actual = (
            row.message_id,
            row.file_key,
            row.file_name,
            row.chat_id,
            row.sender_id,
            row.receive_id_type,
            row.scenario_context_id,
            row.task_type,
        )
        if actual != expected:
            raise FeishuJobOwnershipError(
                "Feishu event conflicts with its persisted sanitized reference"
            )


class SqlAlchemyFeishuJobRouter:
    """Persist first, then dispatch only a job identity through the shared broker."""

    def __init__(
        self,
        store: SqlAlchemyFeishuJobStore,
        *,
        queue: FeishuJobQueue,
        task_resolver: Callable[[FeishuEventReference], FeishuAnalysisTask],
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._queue = queue
        self._task_resolver = task_resolver
        self._clock = clock

    def route_event(
        self,
        *,
        event: FeishuEventReference,
        claim_token: str,
    ) -> FeishuEventRouteStatus:
        task = self._task_resolver(event)
        if not isinstance(task, FeishuAnalysisTask):
            raise TypeError("task_resolver must return FeishuAnalysisTask")
        staged = self._store.stage(
            event=event,
            claim_token=claim_token,
            task=task,
            staged_at=self._clock(),
        )
        if staged.dispatched:
            return FeishuEventRouteStatus.ALREADY_ENQUEUED
        receipt = self._queue.enqueue(job_id=staged.job_id)
        if receipt.job_id != staged.job_id:
            raise ValueError("Feishu queue returned a mismatched job identity")
        self._store.mark_dispatched(
            job_id=staged.job_id,
            task_id=receipt.task_id,
            dispatched_at=self._clock(),
        )
        return FeishuEventRouteStatus.ENQUEUED


class SqlAlchemyFeishuJobReplayService:
    """Idempotently replay one reviewed terminal route rejection as a new job."""

    def __init__(
        self,
        store: SqlAlchemyFeishuJobStore,
        *,
        queue: FeishuJobQueue,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._queue = queue
        self._clock = clock

    def replay_rejected(self, *, source_job_id: str, replay_key: str) -> str:
        staged = self._store.stage_replay(
            source_job_id=source_job_id,
            replay_key=replay_key,
            staged_at=self._clock(),
        )
        if staged.dispatched:
            return staged.job_id
        receipt = self._queue.enqueue(job_id=staged.job_id)
        if receipt.job_id != staged.job_id:
            raise ValueError("Feishu queue returned a mismatched replay job identity")
        self._store.mark_dispatched(
            job_id=staged.job_id,
            task_id=receipt.task_id,
            dispatched_at=self._clock(),
        )
        return staged.job_id


class FeishuAnalysisJobDelivery:
    """Deliver cards, Bitable metadata and reports from audited identities only."""

    def __init__(
        self,
        *,
        client: FeishuDeliveryClient,
        card_builder: AuditedCardBuilder,
        bitable_writer: FeishuBitableWriter,
        report_delivery: FeishuReportDelivery,
        result_authorizer: AuditedResultAuthorizer,
        report_link_factory: (
            Callable[[FeishuReportDeliveryReceipt], str] | None
        ) = None,
        scenario_plotter: FeishuScenarioPlotRenderer | None = None,
        analysis_plotter: FeishuAnalysisPlotRenderer | None = None,
    ) -> None:
        self._client = client
        self._card_builder = card_builder
        self._bitable_writer = bitable_writer
        self._report_delivery = report_delivery
        self._result_authorizer = result_authorizer
        self._report_link_factory = report_link_factory
        self._scenario_plotter = scenario_plotter
        self._analysis_plotter = analysis_plotter

    def deliver_rejection(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None] | None = None,
    ) -> FeishuJobDeliveryReceipt:
        chat_id, receive_id_type = _delivery_target(job)
        if job.result_card_message_id is None:
            card = build_status_card(
                status=FeishuCardStatus.REJECTED,
                run_id=job.run_id,
                result_id=primary_result_id,
                reason_code=reason_code,
                task_type=job.task_type.value,
            )
            sent = self._client.send_message(
                receive_id=chat_id,
                receive_id_type=receive_id_type,
                msg_type="interactive",
                content=card,
            )
            _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(
                    result_card_message_id=_delivery_reference(
                        sent.get("message_id"),
                        field_name="message_id",
                    )
                ),
            )
        if job.bitable_record_id is None:
            fields = self._base_bitable_fields(job, task_status="REJECTED")
            fields["warnings"] = reason_code
            if primary_result_id is not None:
                fields["primary_result_id"] = primary_result_id
            written = self._bitable_writer.upsert(fields)
            bitable_record_id = written.record_id
            _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(bitable_record_id=bitable_record_id),
            )
        else:
            bitable_record_id = job.bitable_record_id
        return FeishuJobDeliveryReceipt(
            bitable_record_id=bitable_record_id,
            report_file_key=None,
        )

    def deliver_success(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        analysis_result: ToolResult,
        report_result: ToolResult,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None] | None = None,
    ) -> FeishuJobDeliveryReceipt:
        chat_id, receive_id_type = _delivery_target(job)
        authorization = validate_delivery_results(
            task=job.task_type,
            expected_analysis_result_id=job.analysis_result_id,
            expected_report_result_id=job.report_result_id,
            analysis_result=analysis_result,
            report_result=report_result,
            authorizer=self._result_authorizer,
        )
        if job.result_card_message_id is None:
            if job.task_type in _SCENARIO_TASKS:
                image_key = job.scenario_image_key
                if image_key is None:
                    if self._scenario_plotter is None:
                        raise ValueError("scenario delivery requires a plot renderer")
                    plot = self._scenario_plotter.render(analysis_result)
                    if plot.source_result_id != analysis_result.result_id:
                        raise ValueError(
                            "scenario plot source does not match the analysis result"
                        )
                    if sha256(plot.payload).hexdigest() != plot.sha256:
                        raise ValueError("scenario plot SHA-256 verification failed")
                    uploaded_image = self._client.upload_image(
                        payload=plot.payload,
                        content_type=plot.media_type,
                        filename=plot.filename,
                    )
                    image_key = _delivery_reference(
                        uploaded_image.get("image_key"),
                        field_name="image_key",
                    )
                    _emit_delivery_progress(
                        checkpoint,
                        FeishuJobDeliveryProgress(scenario_image_key=image_key),
                    )
                result_card = self._card_builder.build_result_card(
                    run_id=job.run_id,
                    result_id=analysis_result.result_id,
                    image_key=image_key,
                )
            elif job.task_type is FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY:
                image_key = job.analysis_image_key
                stored_provenance = (
                    job.analysis_image_key,
                    job.analysis_image_renderer_version,
                    job.analysis_image_sha256,
                )
                if any(value is not None for value in stored_provenance) and any(
                    value is None for value in stored_provenance
                ):
                    raise ValueError("analysis image provenance is incomplete")
                if image_key is None:
                    if self._analysis_plotter is None:
                        raise ValueError("SOH delivery requires a plot renderer")
                    plot = self._analysis_plotter.render(analysis_result)
                    if plot.source_result_id != analysis_result.result_id:
                        raise ValueError(
                            "analysis plot source does not match the analysis result"
                        )
                    if sha256(plot.payload).hexdigest() != plot.sha256:
                        raise ValueError("analysis plot SHA-256 verification failed")
                    renderer_version = getattr(
                        self._analysis_plotter,
                        "renderer_version",
                        "",
                    )
                    _delivery_reference(
                        renderer_version,
                        field_name="analysis_image_renderer_version",
                    )
                    _sha256_reference(
                        plot.sha256,
                        field_name="analysis_image_sha256",
                    )
                    uploaded_image = self._client.upload_image(
                        payload=plot.payload,
                        content_type=plot.media_type,
                        filename=plot.filename,
                    )
                    image_key = _delivery_reference(
                        uploaded_image.get("image_key"),
                        field_name="image_key",
                    )
                    _emit_delivery_progress(
                        checkpoint,
                        FeishuJobDeliveryProgress(
                            analysis_image_key=image_key,
                            analysis_image_renderer_version=renderer_version,
                            analysis_image_sha256=plot.sha256,
                        ),
                    )
                else:
                    expected_renderer_version = getattr(
                        self._analysis_plotter,
                        "renderer_version",
                        "",
                    )
                    if (
                        job.analysis_image_renderer_version != expected_renderer_version
                    ):
                        raise ValueError(
                            "analysis image provenance renderer version mismatch"
                        )
                    _sha256_reference(
                        job.analysis_image_sha256,
                        field_name="analysis_image_sha256",
                    )
                result_card = self._card_builder.build_result_card(
                    run_id=job.run_id,
                    result_id=analysis_result.result_id,
                    image_key=image_key,
                )
            else:
                result_card = self._card_builder.build_result_card(
                    run_id=job.run_id,
                    result_id=analysis_result.result_id,
                )
            sent_result_card = self._client.send_message(
                receive_id=chat_id,
                receive_id_type=receive_id_type,
                msg_type="interactive",
                content=result_card,
            )
            _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(
                    result_card_message_id=_delivery_reference(
                        sent_result_card.get("message_id"),
                        field_name="message_id",
                    )
                ),
            )
        report_receipt = self._report_delivery.deliver(
            result_id=report_result.result_id,
            chat_id=chat_id,
            existing_file_key=job.report_file_key,
            existing_message_id=job.report_message_id,
            on_uploaded=lambda file_key: _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(report_file_key=file_key),
            ),
            on_sent=lambda message_id: _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(report_message_id=message_id),
            ),
        )
        if job.report_card_message_id is None:
            report_card = build_status_card(
                status=FeishuCardStatus.REPORT_READY,
                run_id=job.run_id,
                result_id=report_result.result_id,
                task_type="generate_audited_report",
            )
            sent_report_card = self._client.send_message(
                receive_id=chat_id,
                receive_id_type=receive_id_type,
                msg_type="interactive",
                content=report_card,
            )
            _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(
                    report_card_message_id=_delivery_reference(
                        sent_report_card.get("message_id"),
                        field_name="message_id",
                    )
                ),
            )
        if job.bitable_record_id is None:
            fields = self._base_bitable_fields(job, task_status="SUCCEEDED")
            fields.update(
                {
                    "primary_result_id": analysis_result.result_id,
                    "model_route": authorization.route_id,
                    "model_version": analysis_result.model_version,
                    "data_version": analysis_result.data_version,
                    "feature_version": analysis_result.feature_version,
                    "evidence_level": authorization.evidence_level.value,
                }
            )
            fields.update(_scenario_bitable_metadata(job, analysis_result))
            if analysis_result.warnings:
                fields["warnings"] = "\n".join(analysis_result.warnings)
            if self._report_link_factory is not None:
                fields["report_link"] = self._report_link_factory(report_receipt)
            written = self._bitable_writer.upsert(fields)
            bitable_record_id = written.record_id
            _emit_delivery_progress(
                checkpoint,
                FeishuJobDeliveryProgress(bitable_record_id=bitable_record_id),
            )
        else:
            bitable_record_id = job.bitable_record_id
        return FeishuJobDeliveryReceipt(
            bitable_record_id=bitable_record_id,
            report_file_key=report_receipt.file_key,
        )

    @staticmethod
    def _base_bitable_fields(
        job: FeishuAnalysisJobRecord,
        *,
        task_status: str,
    ) -> dict[str, object]:
        fields: dict[str, object] = {
            "run_id": job.run_id,
            "task_type": job.task_type.value,
            "task_status": task_status,
            "created_at_utc": job.job_created_at,
            "updated_at_utc": job.job_updated_at,
        }
        optional = {
            "data_batch_id": job.record_batch_id,
            "input_file_sha256": job.input_file_sha256,
            "cell_reference": job.cell_reference,
            "scenario_context_id": job.scenario_context_id,
        }
        fields.update({key: value for key, value in optional.items() if value is not None})
        return fields


class _FeishuJobHeartbeat:
    def __init__(
        self,
        *,
        job_id: str,
        claim_token: str,
        interval_seconds: int,
        renew: Callable[[str, str], None],
    ) -> None:
        self._job_id = job_id
        self._claim_token = claim_token
        self._interval_seconds = interval_seconds
        self._renew = renew
        self._stop = Event()
        self._error: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name=f"feishu-job-heartbeat-{job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise FeishuJobOwnershipError("Feishu job heartbeat failed") from None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._renew(self._job_id, self._claim_token)
            except BaseException as exc:
                self._error = exc
                self._stop.set()


class FeishuAnalysisJobWorker:
    """Run the sanitized callback job outside the HTTP callback process."""

    def __init__(
        self,
        store: SqlAlchemyFeishuJobStore,
        *,
        client: FeishuResourceClient,
        attachment_policy: FeishuAttachmentPolicy,
        csv_normalizer: BatteryCsvNormalizerPort | None = None,
        batch_store: VerifiedEarlyCycleBatchStore,
        registration_resolver: Callable[
            [FeishuAnalysisJobRecord, VerifiedFeishuAttachment, datetime],
            CanonicalCsvBatchRegistration,
        ],
        analysis_input_factory: Callable[
            [FeishuAnalysisJobRecord, VerifiedEarlyCycleBatch],
            Mapping[str, object],
        ],
        scenario_input_resolver: FeishuScenarioInputResolver | None = None,
        workflow: FeishuAnalysisWorkflow,
        result_resolver: RegisteredResultResolver,
        report_result_factory: Callable[
            [FeishuAnalysisJobRecord, ToolResult], ToolResult
        ],
        project_model_executor: FeishuProjectModelExecutorPort | None = None,
        delivery: FeishuJobDelivery,
        clock: Callable[[], datetime],
        heartbeat_interval_seconds: int = 30,
        max_attempts: int = 3,
    ) -> None:
        if heartbeat_interval_seconds < 1:
            raise ValueError("heartbeat interval must be positive")
        if max_attempts < 1 or max_attempts > 10:
            raise ValueError("max_attempts must be between 1 and 10")
        self._store = store
        self._client = client
        self._attachment_policy = attachment_policy
        self._csv_normalizer = csv_normalizer
        self._batch_store = batch_store
        self._registration_resolver = registration_resolver
        self._analysis_input_factory = analysis_input_factory
        self._scenario_input_resolver = scenario_input_resolver
        self._workflow = workflow
        self._result_resolver = result_resolver
        self._report_result_factory = report_result_factory
        self._project_model_executor = project_model_executor
        self._delivery = delivery
        self._clock = clock
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._max_attempts = max_attempts

    def execute(self, *, job_id: str) -> FeishuAnalysisJobStatus:
        claim = self._store.claim(job_id=job_id, claimed_at=self._now())
        if claim.status is FeishuJobClaimStatus.IN_PROGRESS:
            raise FeishuJobBusyError("Feishu job has an active worker lease")
        if claim.status is FeishuJobClaimStatus.TERMINAL:
            return self._store.get(job_id).job_status
        if claim.claim_token is None:
            raise FeishuJobOwnershipError("Feishu job claim has no fence")
        heartbeat = _FeishuJobHeartbeat(
            job_id=job_id,
            claim_token=claim.claim_token,
            interval_seconds=self._heartbeat_interval_seconds,
            renew=self._renew,
        )
        heartbeat.start()
        unexpected_error: Exception | None = None
        try:
            try:
                return self._execute_owned(
                    job_id=job_id,
                    claim_token=claim.claim_token,
                )
            except (FeishuJobOwnershipError, FeishuJobRetryableError):
                raise
            except Exception as exc:
                unexpected_error = exc
        finally:
            heartbeat.stop()
        if unexpected_error is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("unexpected worker failure was not captured")
        job = self._store.get(job_id)
        terminal = job.job_attempt_count >= self._max_attempts
        self._retry(job, claim.claim_token, "UNEXPECTED_WORKER_ERROR")
        if terminal:
            return FeishuAnalysisJobStatus.FAILED
        raise FeishuJobRetryableError("UNEXPECTED_WORKER_ERROR") from unexpected_error

    def _execute_owned(self, *, job_id: str, claim_token: str) -> FeishuAnalysisJobStatus:
        job = self._store.get(job_id)
        if job.analysis_result_id is not None and job.report_result_id is not None:
            analysis = self._resolve_result(job.analysis_result_id)
            report = self._resolve_result(job.report_result_id)
            return self._deliver_success(job, claim_token, analysis, report)

        registration: CanonicalCsvBatchRegistration | None = None
        if job.task_type in _SCENARIO_TASKS:
            if self._scenario_input_resolver is None or job.scenario_context_id is None:
                return self._reject(job, claim_token, "SCENARIO_CONTEXT_REQUIRED", None)
            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.VALIDATING_DATA,
                updated_at=self._now(),
            )
            try:
                record_batch_id = self._scenario_input_resolver.resolve_data_batch_id(
                    job.scenario_context_id
                )
                batch = self._batch_store.resolve_verified_early_cycle_batch(
                    record_batch_id
                )
                requested_analysis_input = dict(
                    self._scenario_input_resolver.resolve_analysis_input(
                        scenario_context_id=job.scenario_context_id,
                        task=job.task_type,
                        run_id=job.run_id,
                    )
                )
                validation_input = dict(self._analysis_input_factory(job, batch))
            except (KeyError, TypeError, ValueError):
                return self._reject(job, claim_token, "SCENARIO_CONTEXT_REJECTED", None)
            self._store.checkpoint_batch(
                job_id=job_id,
                claim_token=claim_token,
                record_batch_id=record_batch_id,
                cell_reference=batch.metadata.cell_id,
                input_file_sha256=batch.metadata.source_sha256,
                updated_at=self._now(),
            )
            job = self._store.get(job_id)
        else:
            if job.record_batch_id is not None:
                self._store.update_stage(
                    job_id=job_id,
                    claim_token=claim_token,
                    stage=FeishuAnalysisJobStage.VALIDATING_DATA,
                    updated_at=self._now(),
                )
                try:
                    batch = self._batch_store.resolve_verified_early_cycle_batch(
                        job.record_batch_id
                    )
                    registration = CanonicalCsvBatchRegistration(
                        metadata=batch.metadata,
                        feature_config=batch.feature_config,
                        data_version=batch.data_version,
                        split_version=batch.split_version,
                        provenance=batch.provenance,
                    )
                    validation_input = dict(self._analysis_input_factory(job, batch))
                except (KeyError, TypeError, ValueError):
                    return self._reject(
                        job,
                        claim_token,
                        "REGISTERED_DATA_UNAVAILABLE",
                        None,
                    )
                requested_analysis_input = validation_input
                job = self._store.get(job_id)
            else:
                return self._execute_downloaded_file(
                    job=job,
                    claim_token=claim_token,
                )

        return self._execute_validated_job(
            job=job,
            claim_token=claim_token,
            registration=registration,
            validation_input=validation_input,
            requested_analysis_input=requested_analysis_input,
        )

    def _execute_downloaded_file(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
    ) -> FeishuAnalysisJobStatus:
        job_id = job.job_id
        registration: CanonicalCsvBatchRegistration | None = None
        mapping_evidence: dict[str, object] | None = None
        mapping_evidence_sha256: str | None = None
        try:
            if job.message_id is None or job.file_key is None or job.file_name is None:
                return self._reject(job, claim_token, "UNSUPPORTED_INPUT", None)
            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.DOWNLOADING,
                updated_at=self._now(),
            )
            try:
                response = self._client.download_message_resource_response(
                    message_id=job.message_id,
                    file_key=job.file_key,
                    resource_type="file",
                )
            except FeishuTransportError as exc:
                self._retry(job, claim_token, "DOWNLOAD_RETRYABLE")
                raise FeishuJobRetryableError("DOWNLOAD_RETRYABLE") from exc
            except FeishuApiError as exc:
                if exc.kind in {FeishuErrorKind.RATE_LIMIT, FeishuErrorKind.TRANSIENT}:
                    self._retry(job, claim_token, "DOWNLOAD_RETRYABLE")
                    raise FeishuJobRetryableError("DOWNLOAD_RETRYABLE") from exc
                return self._reject(
                    job,
                    claim_token,
                    "ATTACHMENT_RESOURCE_NOT_FOUND",
                    None,
                )
            except FeishuClientError as exc:
                self._retry(job, claim_token, "DOWNLOAD_RETRYABLE")
                raise FeishuJobRetryableError("DOWNLOAD_RETRYABLE") from exc

            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.VALIDATING_FILE,
                updated_at=self._now(),
            )
            try:
                content_type = _response_content_type(response)
                _verify_content_length(response)
                if self._csv_normalizer is None:
                    attachment = self._attachment_policy.verify(
                        filename=job.file_name,
                        content_type=content_type,
                        payload=response.body,
                    )
                else:
                    source_attachment = self._attachment_policy.verify_csv_envelope(
                        filename=job.file_name,
                        content_type=content_type,
                        payload=response.body,
                    )
                    try:
                        mapped = self._csv_normalizer.normalize(source_attachment.payload)
                    except BatteryCsvMappingError as exc:
                        evidence = BatteryCsvMappingRejectionEvidence(
                            raw_sha256=source_attachment.sha256,
                            reason_code=exc.reason_code,
                            **exc.details.model_dump(mode="python"),
                        ).model_dump(mode="json")
                        self._store.checkpoint_csv_mapping(
                            job_id=job_id,
                            claim_token=claim_token,
                            status="REJECTED",
                            evidence=evidence,
                            evidence_sha256=sha256_canonical(evidence),
                            updated_at=self._now(),
                        )
                        return self._reject(
                            job,
                            claim_token,
                            "CSV_MAPPING_REJECTED",
                            None,
                        )
                    except (TypeError, ValueError):
                        evidence = BatteryCsvMappingRejectionEvidence(
                            raw_sha256=source_attachment.sha256,
                            reason_code="INVALID_MAPPING_INPUT",
                        ).model_dump(mode="json")
                        self._store.checkpoint_csv_mapping(
                            job_id=job_id,
                            claim_token=claim_token,
                            status="REJECTED",
                            evidence=evidence,
                            evidence_sha256=sha256_canonical(evidence),
                            updated_at=self._now(),
                        )
                        return self._reject(
                            job,
                            claim_token,
                            "CSV_MAPPING_REJECTED",
                            None,
                        )
                    canonical_attachment = self._attachment_policy.verify(
                        filename=job.file_name,
                        content_type=content_type,
                        payload=mapped.canonical_payload,
                        expected_sha256=mapped.canonical_sha256,
                    )
                    mapping_evidence = BatteryCsvMappingSuccessEvidence(
                        canonical_sha256=mapped.canonical_sha256,
                        column_evidence=mapped.column_evidence,
                        profile_id=mapped.profile_id,
                        profile_sha256=mapped.profile_sha256,
                        profile_version=mapped.profile_version,
                        raw_sha256=mapped.raw_sha256,
                    ).model_dump(mode="json")
                    mapping_evidence_sha256 = sha256_canonical(mapping_evidence)
                    attachment = VerifiedFeishuAttachment(
                        filename=canonical_attachment.filename,
                        content_type=canonical_attachment.content_type,
                        payload=canonical_attachment.payload,
                        sha256=canonical_attachment.sha256,
                        size_bytes=canonical_attachment.size_bytes,
                        source_sha256=source_attachment.sha256,
                        mapping_evidence={
                            "source_upload_sha256": mapped.raw_sha256,
                            "canonical_payload_sha256": mapped.canonical_sha256,
                            "mapping_profile_id": mapped.profile_id,
                            "mapping_profile_version": mapped.profile_version,
                            "mapping_profile_sha256": mapped.profile_sha256,
                            "column_mapping_evidence": [
                                item.model_dump(mode="json")
                                for item in mapped.column_evidence
                            ],
                        },
                    )
            except (FeishuAttachmentError, ValueError):
                return self._reject(job, claim_token, "ATTACHMENT_REJECTED", None)

            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.REGISTERING_DATA,
                updated_at=self._now(),
            )
            try:
                registration = self._registration_resolver(
                    job,
                    attachment,
                    self._now(),
                )
                record_batch_id = self._batch_store.register_canonical_csv(
                    attachment.payload,
                    registration=registration,
                )
                batch = self._batch_store.resolve_verified_early_cycle_batch(
                    record_batch_id
                )
            except (KeyError, TypeError, ValueError):
                return self._reject(
                    job,
                    claim_token,
                    "DATA_REGISTRATION_REJECTED",
                    None,
                )
            self._store.checkpoint_batch(
                job_id=job_id,
                claim_token=claim_token,
                record_batch_id=record_batch_id,
                cell_reference=batch.metadata.cell_id,
                input_file_sha256=attachment.source_sha256 or attachment.sha256,
                csv_mapping_evidence=mapping_evidence,
                csv_mapping_evidence_sha256=mapping_evidence_sha256,
                updated_at=self._now(),
            )
            job = self._store.get(job_id)
            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.VALIDATING_DATA,
                updated_at=self._now(),
            )
            try:
                validation_input = dict(self._analysis_input_factory(job, batch))
            except (TypeError, ValueError):
                return self._reject(job, claim_token, "ANALYSIS_INPUT_REJECTED", None)
            requested_analysis_input = validation_input
        except (FeishuJobOwnershipError, FeishuJobRetryableError):
            raise
        return self._execute_validated_job(
            job=job,
            claim_token=claim_token,
            registration=registration,
            validation_input=validation_input,
            requested_analysis_input=requested_analysis_input,
        )

    def _execute_validated_job(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        registration: CanonicalCsvBatchRegistration | None,
        validation_input: Mapping[str, object],
        requested_analysis_input: Mapping[str, object],
    ) -> FeishuAnalysisJobStatus:
        job_id = job.job_id
        if (
            self._project_model_executor is not None
            and registration is not None
            and job.task_type
            in {
                FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
                FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
            }
        ):
            return self._execute_project_model(
                job=job,
                claim_token=claim_token,
                registration=registration,
                validation_input=validation_input,
            )

        try:
            self._store.update_stage(
                job_id=job_id,
                claim_token=claim_token,
                stage=FeishuAnalysisJobStage.WAITING_FOR_ROUTE,
                updated_at=self._now(),
            )
            outcome = self._workflow.run(
                task=job.task_type,
                validation_input=validation_input,
                analysis_input=requested_analysis_input,
                before_analysis=lambda: self._store.update_stage(
                    job_id=job_id,
                    claim_token=claim_token,
                    stage=FeishuAnalysisJobStage.RUNNING_TOOL,
                    updated_at=self._now(),
                ),
            )
        except FeishuWorkflowRejected as exc:
            primary_result_id = None
            if exc.validation_result is not None:
                primary_result_id = exc.validation_result.result_id
                self._store.checkpoint_results(
                    job_id=job_id,
                    claim_token=claim_token,
                    validation_result_id=primary_result_id,
                    updated_at=self._now(),
                )
                job = self._store.get(job_id)
            return self._reject(job, claim_token, exc.reason_code, primary_result_id)
        except (TypeError, ValueError):
            return self._reject(job, claim_token, "ANALYSIS_INPUT_REJECTED", None)

        self._store.checkpoint_results(
            job_id=job_id,
            claim_token=claim_token,
            validation_result_id=outcome.validation_result.result_id,
            analysis_result_id=outcome.analysis_result.result_id,
            updated_at=self._now(),
        )
        job = self._store.get(job_id)
        rejection_reason = _analysis_rejection_reason(outcome.analysis_result)
        if rejection_reason is not None:
            return self._reject(
                job,
                claim_token,
                rejection_reason,
                outcome.analysis_result.result_id,
            )
        try:
            report = ToolResult.model_validate(
                self._report_result_factory(job, outcome.analysis_result).model_dump(
                    mode="json"
                )
            )
            self._store.checkpoint_results(
                job_id=job_id,
                claim_token=claim_token,
                report_result_id=report.result_id,
                updated_at=self._now(),
            )
            resolved_report = self._resolve_result(report.result_id)
            if resolved_report != report:
                raise ValueError("report ToolResult registration changed")
        except (AttributeError, TypeError, ValueError):
            return self._fail(job, claim_token, "REPORT_RESULT_INVALID")
        job = self._store.get(job_id)
        return self._deliver_success(job, claim_token, outcome.analysis_result, report)

    def _execute_project_model(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        registration: CanonicalCsvBatchRegistration,
        validation_input: Mapping[str, object],
    ) -> FeishuAnalysisJobStatus:
        executor = self._project_model_executor
        if executor is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("project model executor is unavailable")
        self._store.update_stage(
            job_id=job.job_id,
            claim_token=claim_token,
            stage=FeishuAnalysisJobStage.WAITING_FOR_ROUTE,
            updated_at=self._now(),
        )
        try:
            validation = self._workflow.validate(validation_input=validation_input)
        except FeishuWorkflowRejected as exc:
            primary_result_id = None
            if exc.validation_result is not None:
                primary_result_id = exc.validation_result.result_id
                self._store.checkpoint_results(
                    job_id=job.job_id,
                    claim_token=claim_token,
                    validation_result_id=primary_result_id,
                    updated_at=self._now(),
                )
                job = self._store.get(job.job_id)
            return self._reject(job, claim_token, exc.reason_code, primary_result_id)
        self._store.checkpoint_results(
            job_id=job.job_id,
            claim_token=claim_token,
            validation_result_id=validation.result_id,
            updated_at=self._now(),
        )
        job = self._store.get(job.job_id)
        self._store.update_stage(
            job_id=job.job_id,
            claim_token=claim_token,
            stage=FeishuAnalysisJobStage.RUNNING_TOOL,
            updated_at=self._now(),
        )
        try:
            execution = executor.execute(
                job,
                registration,
                claim_token=claim_token,
            )
            analysis = ToolResult.model_validate(
                execution.analysis_result.model_dump(mode="json")
            )
            report = ToolResult.model_validate(
                execution.report_result.model_dump(mode="json")
            )
        except FeishuProjectModelRejected as exc:
            return self._reject(
                job,
                claim_token,
                exc.reason_code,
                validation.result_id,
            )
        except (AttributeError, TypeError, ValueError):
            return self._reject(
                job,
                claim_token,
                "PROJECT_MODEL_EXECUTION_REJECTED",
                validation.result_id,
            )
        if not execution.slots_committed:
            self._store.checkpoint_results(
                job_id=job.job_id,
                claim_token=claim_token,
                prepared_input_result_id=execution.prepared_input_result_id,
                analysis_result_id=analysis.result_id,
                report_result_id=report.result_id,
                updated_at=self._now(),
            )
        job = self._store.get(job.job_id)
        rejection_reason = _analysis_rejection_reason(analysis)
        if rejection_reason is not None:
            return self._reject(job, claim_token, rejection_reason, analysis.result_id)
        return self._deliver_success(job, claim_token, analysis, report)

    def _deliver_success(
        self,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        analysis: ToolResult,
        report: ToolResult,
    ) -> FeishuAnalysisJobStatus:
        self._store.update_stage(
            job_id=job.job_id,
            claim_token=claim_token,
            stage=FeishuAnalysisJobStage.DELIVERING_RESULT,
            updated_at=self._now(),
        )
        try:
            delivery = self._delivery.deliver_success(
                job=job,
                analysis_result=analysis,
                report_result=report,
                checkpoint=lambda progress: self._checkpoint_delivery(
                    job,
                    claim_token,
                    progress,
                ),
            )
        except FeishuDeliveryResultContractError:
            return self._reject(
                job,
                claim_token,
                "DELIVERY_RESULT_CONTRACT_REJECTED",
                analysis.result_id,
            )
        except Exception as exc:
            self._retry(job, claim_token, "DELIVERY_RETRYABLE")
            raise FeishuJobRetryableError("DELIVERY_RETRYABLE") from exc
        self._store.finish(
            job_id=job.job_id,
            claim_token=claim_token,
            status=FeishuAnalysisJobStatus.SUCCEEDED,
            stage=FeishuAnalysisJobStage.SUCCEEDED,
            error_code=None,
            delivery=delivery,
            completed_at=self._now(),
        )
        return FeishuAnalysisJobStatus.SUCCEEDED

    def _reject(
        self,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        reason_code: str,
        primary_result_id: str | None,
    ) -> FeishuAnalysisJobStatus:
        self._store.update_stage(
            job_id=job.job_id,
            claim_token=claim_token,
            stage=FeishuAnalysisJobStage.DELIVERING_RESULT,
            updated_at=self._now(),
        )
        try:
            delivery = self._delivery.deliver_rejection(
                job=job,
                reason_code=reason_code,
                primary_result_id=primary_result_id,
                checkpoint=lambda progress: self._checkpoint_delivery(
                    job,
                    claim_token,
                    progress,
                ),
            )
        except Exception as exc:
            self._retry(job, claim_token, "DELIVERY_RETRYABLE")
            raise FeishuJobRetryableError("DELIVERY_RETRYABLE") from exc
        self._store.finish(
            job_id=job.job_id,
            claim_token=claim_token,
            status=FeishuAnalysisJobStatus.REJECTED,
            stage=FeishuAnalysisJobStage.REJECTED,
            error_code=reason_code,
            delivery=delivery,
            completed_at=self._now(),
        )
        return FeishuAnalysisJobStatus.REJECTED

    def _retry(
        self,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        error_code: str,
    ) -> None:
        if job.job_attempt_count >= self._max_attempts:
            self._store.finish(
                job_id=job.job_id,
                claim_token=claim_token,
                status=FeishuAnalysisJobStatus.FAILED,
                stage=FeishuAnalysisJobStage.FAILED,
                error_code=error_code,
                delivery=None,
                completed_at=self._now(),
            )
            return
        self._store.mark_retryable(
            job_id=job.job_id,
            claim_token=claim_token,
            error_code=error_code,
            updated_at=self._now(),
        )

    def _fail(
        self,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        error_code: str,
    ) -> FeishuAnalysisJobStatus:
        self._store.finish(
            job_id=job.job_id,
            claim_token=claim_token,
            status=FeishuAnalysisJobStatus.FAILED,
            stage=FeishuAnalysisJobStage.FAILED,
            error_code=error_code,
            delivery=None,
            completed_at=self._now(),
        )
        return FeishuAnalysisJobStatus.FAILED

    def _resolve_result(self, result_id: str) -> ToolResult:
        return ToolResult.model_validate(
            self._result_resolver.resolve_registered_result(result_id).model_dump(
                mode="json"
            )
        )

    def _checkpoint_delivery(
        self,
        job: FeishuAnalysisJobRecord,
        claim_token: str,
        progress: FeishuJobDeliveryProgress,
    ) -> None:
        if not isinstance(progress, FeishuJobDeliveryProgress):
            raise TypeError("delivery checkpoint has an invalid type")
        self._store.checkpoint_delivery(
            job_id=job.job_id,
            claim_token=claim_token,
            scenario_image_key=progress.scenario_image_key,
            analysis_image_key=progress.analysis_image_key,
            analysis_image_renderer_version=progress.analysis_image_renderer_version,
            analysis_image_sha256=progress.analysis_image_sha256,
            result_card_message_id=progress.result_card_message_id,
            report_file_key=progress.report_file_key,
            report_message_id=progress.report_message_id,
            report_card_message_id=progress.report_card_message_id,
            bitable_record_id=progress.bitable_record_id,
            updated_at=self._now(),
        )

    def _renew(self, job_id: str, claim_token: str) -> None:
        self._store.renew_claim(
            job_id=job_id,
            claim_token=claim_token,
            renewed_at=self._now(),
        )

    def _now(self) -> datetime:
        return _utc(self._clock())


def validate_feishu_job_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("job_id must be a UUID") from exc
    if str(parsed) != value:
        raise ValueError("job_id must be canonical")
    return value


def _claim_token(value: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("claim_token must be a hexadecimal fence")
    return normalized


def _error_code(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 100:
        raise ValueError("Feishu job error_code is invalid")
    return normalized


def _replay_key(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 200
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("Feishu replay_key is invalid")
    return normalized


def _delivery_reference(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 200
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"Feishu delivery {field_name} is invalid")
    return normalized


def _sha256_reference(value: object, *, field_name: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


def _emit_delivery_progress(
    checkpoint: Callable[[FeishuJobDeliveryProgress], None] | None,
    progress: FeishuJobDeliveryProgress,
) -> None:
    if checkpoint is not None:
        checkpoint(progress)


def _scenario_bitable_metadata(
    job: FeishuAnalysisJobRecord,
    result: ToolResult,
) -> dict[str, object]:
    if job.task_type not in _SCENARIO_TASKS:
        return {}
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("status") != "COMPLETED":
        raise ValueError("scenario Bitable metadata requires a completed result")
    projection = (
        artifact.get("baseline")
        if job.task_type is FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS
        else artifact.get("projection")
    )
    if not isinstance(projection, Mapping):
        raise ValueError("scenario Bitable projection is invalid")
    scenario_id = _delivery_reference(
        projection.get("scenario_id"),
        field_name="scenario_id",
    )
    scenario_version = _delivery_reference(
        projection.get("scenario_version"),
        field_name="scenario_version",
    )
    return {
        "scenario_id": scenario_id,
        "scenario_version": scenario_version,
    }


def _analysis_rejection_reason(result: ToolResult) -> str | None:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("status") != "REJECTED":
        return None
    reasons = artifact.get("rejection_reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], str):
        try:
            return _error_code(reasons[0])
        except ValueError:
            pass
    return "ANALYSIS_REJECTED"


def _response_content_type(response: FeishuHttpResponse) -> str:
    for key, value in response.headers.items():
        if key.lower() == "content-type":
            return value
    raise ValueError("Feishu attachment response has no content type")


def _verify_content_length(response: FeishuHttpResponse) -> None:
    candidate: str | None = None
    for key, value in response.headers.items():
        if key.lower() == "content-length":
            candidate = value
            break
    if candidate is None:
        return
    try:
        declared = int(candidate)
    except ValueError as exc:
        raise ValueError("Feishu attachment content length is invalid") from exc
    if declared < 0 or declared != len(response.body):
        raise ValueError("Feishu attachment content length does not match")


def _delivery_target(job: FeishuAnalysisJobRecord) -> tuple[str, str]:
    if job.chat_id is None or job.receive_id_type is None:
        raise ValueError("Feishu job has no delivery target")
    return job.chat_id, job.receive_id_type


def _scenario_context_reference(
    *,
    event: FeishuEventReference,
    task: FeishuAnalysisTask,
) -> str | None:
    if task not in _SCENARIO_TASKS:
        if "scenario_context_id" in event.action_value:
            raise ValueError("scenario context cannot be attached to a non-scenario task")
        return None
    if event.kind is not FeishuInboundEventKind.CARD_ACTION:
        raise ValueError("scenario analysis requires a Feishu card action")
    if set(event.action_value) != {"task_type", "scenario_context_id"}:
        raise ValueError("scenario card action must contain only task and context references")
    if event.action_value.get("task_type") != task.value:
        raise ValueError("scenario card action task does not match its resolved task")
    context_id = event.action_value.get("scenario_context_id")
    if context_id is None:
        raise ValueError("scenario card action has no context reference")
    return validate_feishu_job_id(context_id)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Feishu job timestamps must include a timezone")
    return value.astimezone(UTC)


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "FeishuAnalysisJobDelivery",
    "FeishuAnalysisJobOrigin",
    "FeishuAnalysisJobRecord",
    "FeishuAnalysisJobStage",
    "FeishuAnalysisJobStatus",
    "FeishuAnalysisJobWorker",
    "FeishuJobBusyError",
    "FeishuJobClaim",
    "FeishuJobClaimStatus",
    "FeishuJobDelivery",
    "FeishuJobDeliveryProgress",
    "FeishuJobDeliveryReceipt",
    "FeishuJobDispatchReceipt",
    "FeishuJobOwnershipError",
    "FeishuJobQueue",
    "FeishuJobRetryableError",
    "FeishuProjectModelExecution",
    "FeishuProjectModelExecutorPort",
    "FeishuProjectModelRejected",
    "FeishuScenarioInputResolver",
    "FeishuScenarioPlotRenderer",
    "OriginAwareAnalysisJobDelivery",
    "SqlAlchemyFeishuJobReplayService",
    "SqlAlchemyFeishuJobRouter",
    "SqlAlchemyFeishuJobStore",
    "validate_feishu_job_id",
]
