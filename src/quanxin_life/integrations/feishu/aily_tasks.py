"""Durable Aily task facade over the shared sanitized analysis-job store."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from quanxin_life.core import AgentRunState, AgentRunStatus, ToolResult

from .analysis_bitable import build_audited_analysis_bitable_fields
from .bitable import BitableWriteResult
from .cards import AuditedResultAuthorizer
from .delivery_contract import validate_delivery_results
from .jobs import (
    AilyDataIdentityResolver,
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
    FeishuBitableCurvePlotRenderer,
    FeishuBitableMediaUploader,
    FeishuJobDeliveryProgress,
    FeishuJobDeliveryReceipt,
    FeishuJobQueue,
    SqlAlchemyFeishuJobStore,
    prepare_audited_bitable_curve_fields,
)
from .scenario_contexts import SqlAlchemyFeishuScenarioContextStore
from .workflow import FeishuAnalysisTask

if TYPE_CHECKING:
    from quanxin_life.api.aily import AilyCreateAnalysisTaskRequest

_SCENARIO_TASKS = frozenset(
    {
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    }
)
_DATA_TASKS = frozenset(
    {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    }
)
_RECOMMENDATION_TASK = FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION
_BITABLE_CURVE_TASKS = _SCENARIO_TASKS | _DATA_TASKS
Clock = Callable[[], datetime]


class AilyBitableWriter(Protocol):
    def upsert(self, fields: Mapping[str, object]) -> BitableWriteResult: ...


class AilyAnalysisJobDelivery:
    """Deliver Aily outcomes as scalar metadata without requiring a Feishu chat."""

    def __init__(
        self,
        *,
        bitable_writer: AilyBitableWriter,
        result_authorizer: AuditedResultAuthorizer,
        report_link_factory: (
            Callable[[FeishuAnalysisJobRecord, ToolResult], str] | None
        ) = None,
        analysis_plotter: FeishuBitableCurvePlotRenderer | None = None,
        bitable_media_uploader: FeishuBitableMediaUploader | None = None,
    ) -> None:
        self._bitable_writer = bitable_writer
        self._result_authorizer = result_authorizer
        self._report_link_factory = report_link_factory
        self._analysis_plotter = analysis_plotter
        self._bitable_media_uploader = bitable_media_uploader

    def deliver_rejection(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: Callable[[FeishuJobDeliveryProgress], None] | None = None,
    ) -> FeishuJobDeliveryReceipt:
        self._require_aily(job)
        if job.bitable_record_id is not None:
            return FeishuJobDeliveryReceipt(
                bitable_record_id=job.bitable_record_id,
                report_file_key=None,
            )
        fields = self._base_fields(job, task_status="REJECTED")
        fields["warnings"] = _safe_text(reason_code, field_name="reason_code")
        if primary_result_id is not None:
            fields["primary_result_id"] = _safe_text(
                primary_result_id,
                field_name="primary_result_id",
            )
        written = self._bitable_writer.upsert(fields)
        _emit_bitable_checkpoint(checkpoint, written.record_id)
        return FeishuJobDeliveryReceipt(
            bitable_record_id=written.record_id,
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
        self._require_aily(job)
        authorization = validate_delivery_results(
            task=job.task_type,
            expected_analysis_result_id=job.analysis_result_id,
            expected_report_result_id=job.report_result_id,
            analysis_result=analysis_result,
            report_result=report_result,
            authorizer=self._result_authorizer,
        )
        should_write_bitable = job.bitable_record_id is None or (
            job.task_type in _BITABLE_CURVE_TASKS
            and job.bitable_curve_file_token is None
            and self._analysis_plotter is not None
            and self._bitable_media_uploader is not None
        )
        if not should_write_bitable:
            return FeishuJobDeliveryReceipt(
                bitable_record_id=job.bitable_record_id,
                report_file_key=None,
            )
        curve_fields = (
            prepare_audited_bitable_curve_fields(
                job=job,
                analysis_result=analysis_result,
                analysis_plotter=self._analysis_plotter,
                bitable_media_uploader=self._bitable_media_uploader,
                checkpoint=checkpoint,
            )
            if job.task_type in _BITABLE_CURVE_TASKS
            else {}
        )
        fields = self._base_fields(job, task_status="SUCCEEDED")
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
        fields.update(build_audited_analysis_bitable_fields(analysis_result))
        if job.task_type in _SCENARIO_TASKS:
            fields.update(_scenario_fields(job, analysis_result))
        fields.update(curve_fields)
        if analysis_result.warnings and job.task_type is not _RECOMMENDATION_TASK:
            fields["warnings"] = "\n".join(analysis_result.warnings)
        if self._report_link_factory is not None:
            fields["report_link"] = self._report_link_factory(job, report_result)
        written = self._bitable_writer.upsert(fields)
        _emit_bitable_checkpoint(checkpoint, written.record_id)
        return FeishuJobDeliveryReceipt(
            bitable_record_id=written.record_id,
            report_file_key=None,
        )

    @staticmethod
    def _base_fields(
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

    @staticmethod
    def _require_aily(job: FeishuAnalysisJobRecord) -> None:
        if job.job_origin is not FeishuAnalysisJobOrigin.AILY:
            raise ValueError("Aily delivery requires an Aily analysis job")


def _scenario_fields(
    job: FeishuAnalysisJobRecord,
    result: ToolResult,
) -> dict[str, object]:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("status") != "COMPLETED":
        raise ValueError("Aily scenario metadata requires a completed result")
    projection = (
        artifact.get("baseline")
        if job.task_type is FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS
        else artifact.get("projection")
    )
    if not isinstance(projection, Mapping):
        raise ValueError("Aily scenario projection metadata is invalid")
    return {
        "scenario_id": _safe_text(
            projection.get("scenario_id"),
            field_name="scenario_id",
        ),
        "scenario_version": _safe_text(
            projection.get("scenario_version"),
            field_name="scenario_version",
        ),
    }


def _safe_text(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 200
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"Aily delivery {field_name} is invalid")
    return normalized


def _emit_bitable_checkpoint(
    checkpoint: Callable[[FeishuJobDeliveryProgress], None] | None,
    record_id: str,
) -> None:
    if checkpoint is not None:
        checkpoint(FeishuJobDeliveryProgress(bitable_record_id=record_id))


class SqlAlchemyAilyAnalysisTaskGateway:
    """Create and resolve reference-only Aily jobs through the shared queue."""

    def __init__(
        self,
        *,
        job_store: SqlAlchemyFeishuJobStore,
        scenario_context_store: SqlAlchemyFeishuScenarioContextStore,
        data_identity_resolver: AilyDataIdentityResolver,
        queue: FeishuJobQueue,
        clock: Clock,
    ) -> None:
        self._job_store = job_store
        self._scenario_context_store = scenario_context_store
        self._data_identity_resolver = data_identity_resolver
        self._queue = queue
        self._clock = clock

    def create_analysis_task(
        self,
        request: AilyCreateAnalysisTaskRequest,
    ) -> AgentRunState:
        task = request.task_type
        scenario_context_id: str | None = None
        if task is _RECOMMENDATION_TASK:
            recommendation = self._job_store.get_recommendation_sibling(
                source_job_id=request.source_run_id
            )
            if recommendation.record_batch_id is None:
                raise ValueError("Aily recommendation data identity is incomplete")
            self._data_identity_resolver.resolve_source_job(
                source_run_id=request.source_run_id,
                data_batch_id=recommendation.record_batch_id,
            )
            return _state(recommendation)
        if task in _SCENARIO_TASKS:
            if request.scenario_context_id is None:
                raise ValueError(
                    "Aily durable scenario analysis requires a persisted context"
                )
            context = self._scenario_context_store.get(request.scenario_context_id)
            if context.task is not task:
                raise ValueError("scenario context task does not match the Aily request")
            if context.created_by_reference != request.source_run_id:
                raise ValueError(
                    "scenario context is not bound to the requested source upload"
                )
            data_batch_id = context.data_batch_id
            scenario_context_id = context.scenario_context_id
        elif task in _DATA_TASKS:
            if request.data_batch_id is None:
                raise ValueError("Aily data analysis requires a persisted batch")
            data_batch_id = request.data_batch_id
        else:
            raise ValueError("Aily durable analysis task is not supported")
        self._data_identity_resolver.resolve_source_job(
            source_run_id=request.source_run_id,
            data_batch_id=data_batch_id,
        )
        staged = self._job_store.stage_aily(
            task=task,
            source_job_id=request.source_run_id,
            record_batch_id=data_batch_id,
            scenario_context_id=scenario_context_id,
            staged_at=self._now(),
        )
        if not staged.dispatched:
            dispatched = self._queue.enqueue(job_id=staged.job_id)
            if dispatched.job_id != staged.job_id:
                raise RuntimeError("Aily queue returned a mismatched job identity")
            self._job_store.mark_dispatched(
                job_id=staged.job_id,
                task_id=dispatched.task_id,
                dispatched_at=self._now(),
            )
        return _state(self._job_store.get(staged.job_id))

    def get_analysis_task(self, run_id: str) -> AgentRunState:
        try:
            job = self._job_store.get(run_id)
        except ValueError as exc:
            raise LookupError("Aily analysis task was not found") from exc
        visible_recommendation = (
            job.job_origin is FeishuAnalysisJobOrigin.FEISHU
            and job.event_type == "feishu.analysis_job.derived_v1"
            and job.task_type is _RECOMMENDATION_TASK
            and job.scenario_context_id is None
        )
        if (
            (
                job.job_origin is not FeishuAnalysisJobOrigin.AILY
                and not visible_recommendation
            )
            or job.job_request_sha256 is None
            or job.source_job_id is None
            or job.record_batch_id is None
            or (
                job.task_type in _SCENARIO_TASKS
                and job.scenario_context_id is None
            )
            or (
                job.task_type in _DATA_TASKS
                and job.scenario_context_id is not None
            )
            or (
                job.task_type is _RECOMMENDATION_TASK
                and job.scenario_context_id is not None
            )
        ):
            raise LookupError("Aily analysis task was not found")
        source_job_id = job.source_job_id
        data_batch_id = job.record_batch_id
        assert source_job_id is not None
        assert data_batch_id is not None
        if job.task_type in _SCENARIO_TASKS:
            assert job.scenario_context_id is not None
            context = self._scenario_context_store.get(job.scenario_context_id)
            if (
                context.task is not job.task_type
                or context.data_batch_id != data_batch_id
                or context.created_by_reference != source_job_id
            ):
                raise ValueError(
                    "Aily scenario task is not bound to its source upload"
                )
        self._data_identity_resolver.resolve_source_job(
            source_run_id=source_job_id,
            data_batch_id=data_batch_id,
        )
        return _state(job)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Aily task clock must include a timezone")
        return value.astimezone(UTC)


def _state(job: FeishuAnalysisJobRecord) -> AgentRunState:
    intent_id = job.scenario_context_id or job.source_job_id
    request_sha256 = job.job_request_sha256
    if intent_id is None or request_sha256 is None:
        raise ValueError("Aily analysis task metadata is incomplete")
    result_ids = tuple(
        result_id
        for result_id in (
            job.validation_result_id,
            job.analysis_result_id,
            job.report_result_id,
        )
        if result_id is not None
    )
    completed_steps = tuple(
        step_id
        for step_id, result_id in (
            ("validate", job.validation_result_id),
            ("analyze", job.analysis_result_id),
            ("report", job.report_result_id),
        )
        if result_id is not None
    )
    warnings = (
        (job.job_last_error_code,) if job.job_last_error_code is not None else ()
    )
    return AgentRunState(
        run_id=job.run_id,
        intent_id=intent_id,
        plan_hash=request_sha256,
        status=_run_status(job.job_status),
        completed_step_ids=completed_steps,
        result_ids=result_ids,
        warnings=warnings,
        updated_at=job.job_updated_at,
    )


def _run_status(status: FeishuAnalysisJobStatus) -> AgentRunStatus:
    return {
        FeishuAnalysisJobStatus.PENDING: AgentRunStatus.PLANNING,
        FeishuAnalysisJobStatus.RUNNING: AgentRunStatus.RUNNING,
        FeishuAnalysisJobStatus.RETRYABLE: AgentRunStatus.RUNNING,
        FeishuAnalysisJobStatus.SUCCEEDED: AgentRunStatus.COMPLETED,
        FeishuAnalysisJobStatus.REJECTED: AgentRunStatus.FALLBACK,
        FeishuAnalysisJobStatus.FAILED: AgentRunStatus.FAILED,
    }[status]


__all__ = [
    "AilyAnalysisJobDelivery",
    "AilyBitableWriter",
    "AilyDataIdentityResolver",
    "SqlAlchemyAilyAnalysisTaskGateway",
]
