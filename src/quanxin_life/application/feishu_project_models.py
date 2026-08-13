"""Project-bound Feishu orchestration for reviewed Advanced model tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import or_, select

from quanxin_life.api.service import ToolInvocation
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit import AuditLedger, ProjectResultLedger
from quanxin_life.core import (
    AdvancedModelRouteRole,
    DatasetStatus,
    ToolResult,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobRecord,
    FeishuProjectModelExecution,
    FeishuProjectModelRejected,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import Dataset, FeishuEventReceipt, RecordBatchBinding
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.audited_report import (
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
    execute_generate_audited_report_tool,
)

Clock = Callable[[], datetime]


class ProjectToolInvocationPort(Protocol):
    def invoke_in_project(
        self,
        invocation: ToolInvocation,
        *,
        context: VerifiedProjectInvocationContext,
    ) -> ToolResult: ...


class FeishuProjectRecordBatchResolver:
    """Map reviewed upload identity to one exact FROZEN project batch."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        registration: CanonicalCsvBatchRegistration,
    ) -> str:
        checked = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        with session_scope(self._session_factory) as session:
            rows = tuple(
                session.scalars(
                    select(RecordBatchBinding)
                    .join(Dataset, Dataset.id == RecordBatchBinding.dataset_id)
                    .where(
                        RecordBatchBinding.project_id == context.project_id,
                        Dataset.project_id == context.project_id,
                        Dataset.status == DatasetStatus.FROZEN.value,
                        Dataset.frozen_at.is_not(None),
                        Dataset.data_version == checked.data_version,
                        Dataset.schema_version == checked.metadata.schema_version,
                        RecordBatchBinding.source_manifest_sha256
                        == checked.metadata.source_sha256,
                        RecordBatchBinding.content_dataset_id
                        == checked.metadata.dataset_id,
                        RecordBatchBinding.dataset_schema_version
                        == checked.metadata.schema_version,
                        RecordBatchBinding.cell_id == checked.metadata.cell_id,
                        RecordBatchBinding.cutoff_cycle
                        == checked.feature_config.cutoff_cycle,
                        RecordBatchBinding.data_version == checked.data_version,
                        RecordBatchBinding.split_version == checked.split_version,
                        RecordBatchBinding.feature_version
                        == checked.feature_config.feature_version,
                    )
                    .order_by(RecordBatchBinding.id)
                ).all()
            )
        if not rows:
            raise FeishuProjectModelRejected(
                "PROJECT_RECORD_BATCH_NOT_FROZEN",
                "exact FROZEN project RecordBatchBinding is unavailable",
            )
        if len(rows) != 1:
            raise FeishuProjectModelRejected(
                "PROJECT_RECORD_BATCH_AMBIGUOUS",
                "project RecordBatchBinding identity is not unique",
            )
        return rows[0].id


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class FeishuProjectResultResolver:
    """Resolve only results bound to one durable job and its live identity."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        global_resolver: RegisteredResultResolver,
        context_service: ProjectInvocationContextService,
        project_ledger: ProjectResultLedger,
    ) -> None:
        self._session_factory = session_factory
        self._global_resolver = global_resolver
        self._context_service = context_service
        self._project_ledger = project_ledger

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        checked_result_id = _result_id(result_id)
        with session_scope(self._session_factory) as session:
            jobs = tuple(
                session.scalars(
                    select(FeishuEventReceipt).where(
                        or_(
                            FeishuEventReceipt.validation_result_id
                            == checked_result_id,
                            FeishuEventReceipt.analysis_result_id
                            == checked_result_id,
                            FeishuEventReceipt.report_result_id
                            == checked_result_id,
                        )
                    )
                ).all()
            )
            if len(jobs) != 1:
                raise ValueError("ToolResult is not uniquely bound to an analysis job")
            job_origin = jobs[0].job_origin
            chat_id = jobs[0].chat_id
            sender_id = jobs[0].sender_id

        context: VerifiedProjectInvocationContext | None = None
        if job_origin == "FEISHU":
            if not chat_id or not sender_id:
                raise ValueError("Feishu ToolResult identity is unavailable")
            try:
                context = self._context_service.resolve_feishu(
                    chat_id=chat_id,
                    sender_open_id=sender_id,
                )
            except (LookupError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "project ToolResult is not authorized by an active Feishu binding"
                ) from exc
        try:
            return self._global_resolver.resolve_registered_result(checked_result_id)
        except ValueError:
            pass
        if job_origin != "FEISHU" or context is None:
            raise ValueError("project ToolResult is not bound to a Feishu job")
        try:
            result = self._project_ledger.resolve_registered_result(
                context,
                checked_result_id,
            )
            binding = self._project_ledger.resolve_binding(
                context,
                checked_result_id,
            )
        except (LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Feishu project ToolResult could not be verified") from exc
        if (
            binding.invocation_source is not ProjectInvocationSource.FEISHU
            or binding.project_id != context.project_id
            or binding.actor_user_id != context.actor_user_id
            or binding.feishu_binding_id != context.feishu_binding_id
        ):
            raise ValueError(
                "project ToolResult is not authorized by the exact Feishu binding"
            )
        return ToolResult.model_validate(result.model_dump(mode="json"))


class FeishuProjectModelExecutor:
    """Invoke reviewed project tools without deriving any business numbers."""

    def __init__(
        self,
        *,
        context_service: ProjectInvocationContextService,
        batch_resolver: FeishuProjectRecordBatchResolver,
        project_tool_service: ProjectToolInvocationPort,
        project_ledger: ProjectResultLedger,
        clock: Clock,
    ) -> None:
        self._context_service = context_service
        self._batch_resolver = batch_resolver
        self._project_tool_service = project_tool_service
        self._project_ledger = project_ledger
        self._clock = clock

    def execute(
        self,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
    ) -> FeishuProjectModelExecution:
        if _task_value(job.task_type) != StandardToolName.PREDICT_CYCLE_LIFE.value:
            raise FeishuProjectModelRejected(
                "PROJECT_MODEL_TASK_NOT_SUPPORTED",
                "Feishu project model task is not supported",
            )
        if not job.chat_id or not job.sender_id:
            raise FeishuProjectModelRejected(
                "FEISHU_PROJECT_BINDING_REQUIRED",
                "Feishu project identity is unavailable",
            )
        try:
            context = self._context_service.resolve_feishu(
                chat_id=job.chat_id,
                sender_open_id=job.sender_id,
            )
            record_batch_id = self._batch_resolver.resolve(context, registration)
            role = _rul_role(registration.feature_config.cutoff_cycle)
            prepared = self._project_tool_service.invoke_in_project(
                ToolInvocation(
                    tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                    input_value={"record_batch_id": record_batch_id},
                ),
                context=context,
            )
            analysis = self._project_tool_service.invoke_in_project(
                ToolInvocation(
                    tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
                    input_value={
                        "upstream_result_id": prepared.result_id,
                        "route_role": role.value,
                    },
                ),
                context=context,
            )
            if (
                analysis.tool_name != StandardToolName.PREDICT_CYCLE_LIFE.value
                or analysis.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
            ):
                raise ValueError("project RUL ToolResult contract is invalid")
            report = self._audited_report(context, analysis)
        except FeishuProjectModelRejected:
            raise
        except (LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise FeishuProjectModelRejected(
                "PROJECT_MODEL_EXECUTION_REJECTED",
                "reviewed project model execution was rejected",
            ) from exc
        return FeishuProjectModelExecution(
            record_batch_id=record_batch_id,
            prepared_input_result_id=prepared.result_id,
            analysis_result=analysis,
            report_result=report,
        )

    def _audited_report(
        self,
        context: VerifiedProjectInvocationContext,
        analysis: ToolResult,
    ) -> ToolResult:
        report_input = GenerateAuditedReportToolInput(
            report_kind=ReportKind.LIFETIME_DECISION,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                    numeric_evidence=(
                        NumericEvidenceReference(
                            result_id=analysis.result_id,
                            json_path=(
                                "values.artifact.cycle_life_prediction.predicted_cycle"
                            ),
                        ),
                        NumericEvidenceReference(
                            result_id=analysis.result_id,
                            json_path="values.artifact.derived_remaining_cycles",
                        ),
                    ),
                ),
            ),
            upstream_result_ids=(analysis.result_id,),
        )
        report = execute_generate_audited_report_tool(
            report_input,
            audit_ledger=AuditLedger((analysis,)),
            clock=self._clock,
        )
        return self._project_ledger.register_result(context, report)


def _rul_role(cutoff_cycle: int) -> AdvancedModelRouteRole:
    if cutoff_cycle == 20:
        return AdvancedModelRouteRole.DEFAULT
    if cutoff_cycle in {50, 100, 150}:
        return AdvancedModelRouteRole.POINT_ACCURACY
    raise FeishuProjectModelRejected(
        "PROJECT_MODEL_CUTOFF_NOT_SUPPORTED",
        "Advanced RUL requires cutoff 20, 50, 100, or 150",
    )


def _task_value(value: object) -> str:
    candidate = getattr(value, "value", value)
    return candidate if isinstance(candidate, str) else ""


def _result_id(value: object) -> str:
    try:
        return str(UUID(value))  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("result_id must be a UUID string") from exc


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "FeishuProjectModelExecutor",
    "FeishuProjectModelRejected",
    "FeishuProjectRecordBatchResolver",
    "FeishuProjectResultResolver",
    "ProjectToolInvocationPort",
]
