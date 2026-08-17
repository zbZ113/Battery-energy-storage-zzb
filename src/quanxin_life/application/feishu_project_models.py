"""Project-bound Feishu orchestration for reviewed Advanced model tools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy import or_, select

from quanxin_life.api.service import ToolInvocation
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    project_model_registration_rejection_code,
)
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
    sha256_canonical,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobRecord,
    FeishuProjectModelExecution,
    FeishuProjectModelRejected,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import Dataset, FeishuEventReceipt, RecordBatchBinding
from quanxin_life.reporting.contracts import AUDITED_REPORT_TOOL_VERSION
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_input import PREPARE_ADVANCED_INPUT_TOOL_VERSION
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
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


class UnregisteredProjectToolExecutionPort(Protocol):
    def execute_in_project_unregistered(
        self,
        invocation: ToolInvocation,
        *,
        context: VerifiedProjectInvocationContext,
    ) -> ToolResult: ...


class FeishuProjectResultSlotCommitter(Protocol):
    def commit_feishu_result_slot(
        self,
        *,
        context: VerifiedProjectInvocationContext,
        job_id: str,
        claim_token: str,
        slot: str,
        expected_task: str,
        run_id: str,
        chat_id: str,
        sender_id: str,
        result: ToolResult,
    ) -> ToolResult: ...


class FeishuProjectBatchProvisioner(Protocol):
    def provision_frozen_binding(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        content_batch_id: str,
        registration: CanonicalCsvBatchRegistration,
        now: datetime,
    ) -> object: ...


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
        registration_sha256 = sha256_canonical(checked.model_dump(mode="json"))
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
                        RecordBatchBinding.registration_sha256
                        == registration_sha256,
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
                            FeishuEventReceipt.prepared_input_result_id
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
            job = jobs[0]
            job_origin = job.job_origin
            chat_id = job.chat_id
            sender_id = job.sender_id
            if job_origin == "AILY":
                source = session.scalar(
                    select(FeishuEventReceipt).where(
                        FeishuEventReceipt.job_id == job.source_job_id
                    )
                )
                if not _is_anchored_aily_project_job(job, source):
                    raise ValueError(
                        "Aily project ToolResult is not bound to an authorized upload"
                    )

        context: VerifiedProjectInvocationContext | None = None
        if job_origin in {"FEISHU", "AILY"}:
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
        if job_origin not in {"FEISHU", "AILY"} or context is None:
            raise ValueError("project ToolResult is not bound to an authorized job")
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
        batch_provisioner: FeishuProjectBatchProvisioner | None = None,
        project_tool_service: ProjectToolInvocationPort,
        project_ledger: ProjectResultLedger,
        clock: Clock,
    ) -> None:
        self._context_service = context_service
        self._batch_resolver = batch_resolver
        self._batch_provisioner = batch_provisioner
        self._project_tool_service = project_tool_service
        self._project_ledger = project_ledger
        self._clock = clock

    def execute(
        self,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
        *,
        claim_token: str | None = None,
    ) -> FeishuProjectModelExecution:
        rejection_code = project_model_registration_rejection_code(registration)
        if rejection_code is not None:
            self.provision_registration(job, registration)
            raise FeishuProjectModelRejected(
                rejection_code,
                "uploaded data is outside the reviewed project model domain",
            )
        if callable(
            getattr(self._project_tool_service, "execute_in_project_unregistered", None)
        ):
            return self._execute_atomic(
                job=job,
                registration=registration,
                claim_token=claim_token,
            )
        return self._execute_legacy(job, registration)

    def provision_registration(
        self,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
    ) -> str | None:
        """Freeze verified upload identity without invoking a numerical model."""

        if self._batch_provisioner is None:
            return None
        content_batch_id = getattr(job, "record_batch_id", None)
        if not isinstance(content_batch_id, str) or not content_batch_id.strip():
            raise FeishuProjectModelRejected(
                "PROJECT_RECORD_BATCH_NOT_FROZEN",
                "verified upload content batch is unavailable",
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
            provisioned = self._batch_provisioner.provision_frozen_binding(
                context,
                content_batch_id=content_batch_id,
                registration=registration,
                now=self._clock(),
            )
            record_batch_id = getattr(provisioned, "record_batch_id", None)
            if not isinstance(record_batch_id, str) or not record_batch_id.strip():
                raise ValueError(
                    "project batch provisioner returned an invalid binding"
                )
            return record_batch_id
        except FeishuProjectModelRejected:
            raise
        except (LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise FeishuProjectModelRejected(
                "PROJECT_MODEL_EXECUTION_REJECTED",
                "verified upload project binding could not be provisioned",
            ) from exc

    def _execute_legacy(
        self,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
    ) -> FeishuProjectModelExecution:
        try:
            task = StandardToolName(_task_value(job.task_type))
        except ValueError as exc:
            raise FeishuProjectModelRejected(
                "PROJECT_MODEL_TASK_NOT_SUPPORTED",
                "Feishu project model task is not supported",
            ) from exc
        if task not in {
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
        }:
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
            record_batch_id = self._resolve_record_batch_id(
                context=context,
                job=job,
                registration=registration,
            )
            prepared = self._project_tool_service.invoke_in_project(
                ToolInvocation(
                    tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                    input_value={"record_batch_id": record_batch_id},
                ),
                context=context,
            )
            role = _project_model_role(
                task=task,
                cutoff_cycle=registration.feature_config.cutoff_cycle,
            )
            analysis = self._project_tool_service.invoke_in_project(
                ToolInvocation(
                    tool_name=task,
                    input_value={
                        "upstream_result_id": prepared.result_id,
                        "route_role": role.value,
                    },
                ),
                context=context,
            )
            expected_version = (
                ADVANCED_RUL_PREDICTION_TOOL_VERSION
                if task is StandardToolName.PREDICT_CYCLE_LIFE
                else ADVANCED_SOH_PREDICTION_TOOL_VERSION
            )
            if (
                analysis.tool_name != task.value
                or analysis.tool_version != expected_version
            ):
                raise ValueError("project model ToolResult contract is invalid")
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
        if analysis.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value:
            paths = (
                "values.artifact.cycle_life_prediction.predicted_cycle",
                "values.artifact.derived_remaining_cycles",
            )
        elif analysis.tool_name == StandardToolName.PREDICT_SOH_TRAJECTORY.value:
            paths = _soh_report_paths(analysis)
        else:  # pragma: no cover - guarded by execute
            raise ValueError("project report task is unsupported")
        report_input = GenerateAuditedReportToolInput(
            report_kind=ReportKind.LIFETIME_DECISION,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                    numeric_evidence=tuple(
                        NumericEvidenceReference(
                            result_id=analysis.result_id,
                            json_path=json_path,
                        )
                        for json_path in paths
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

    def _execute_atomic(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
        claim_token: str | None,
    ) -> FeishuProjectModelExecution:
        if claim_token is None:
            raise FeishuProjectModelRejected(
                "FEISHU_PROJECT_CLAIM_REQUIRED",
                "atomic Feishu project execution requires a live job claim",
            )
        if not callable(
            getattr(self._project_ledger, "commit_feishu_result_slot", None)
        ):
            raise FeishuProjectModelRejected(
                "FEISHU_PROJECT_ATOMIC_COMMIT_UNAVAILABLE",
                "atomic Feishu project result commit is unavailable",
            )
        try:
            task = StandardToolName(_task_value(job.task_type))
            role = _project_model_role(
                task=task,
                cutoff_cycle=registration.feature_config.cutoff_cycle,
            )
            context = self._context_service.resolve_feishu(
                chat_id=job.chat_id or "",
                sender_open_id=job.sender_id or "",
            )
            record_batch_id = self._resolve_record_batch_id(
                context=context,
                job=job,
                registration=registration,
            )
            execute_port = cast(
                UnregisteredProjectToolExecutionPort,
                self._project_tool_service,
            )
            committer = cast(FeishuProjectResultSlotCommitter, self._project_ledger)
            prepared_input = ToolInvocation(
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                input_value={"record_batch_id": record_batch_id},
            )
            prepared = _resolve_or_execute_slot(
                job=job,
                slot="PREPARED",
                result_id=job.prepared_input_result_id,
                expected_tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                expected_tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
                expected_input_hash=sha256_canonical(prepared_input.input_value),
                expected_reference_id=record_batch_id,
                context=context,
                resolver=self._project_ledger,
                committer=committer.commit_feishu_result_slot,
                claim_token=claim_token,
                expected_task=_task_value(job.task_type),
                run_id=job.run_id,
                chat_id=job.chat_id or "",
                sender_id=job.sender_id or "",
                execute=lambda: execute_port.execute_in_project_unregistered(
                    prepared_input,
                    context=context,
                ),
            )
            analysis_input = ToolInvocation(
                tool_name=task,
                input_value={
                    "upstream_result_id": prepared.result_id,
                    "route_role": role.value,
                },
            )
            analysis = _resolve_or_execute_slot(
                job=job,
                slot="ANALYSIS",
                result_id=job.analysis_result_id,
                expected_tool_name=task.value,
                expected_tool_version=(
                    ADVANCED_RUL_PREDICTION_TOOL_VERSION
                    if task is StandardToolName.PREDICT_CYCLE_LIFE
                    else ADVANCED_SOH_PREDICTION_TOOL_VERSION
                ),
                expected_input_hash=sha256_canonical(analysis_input.input_value),
                expected_reference_id=prepared.result_id,
                context=context,
                resolver=self._project_ledger,
                committer=committer.commit_feishu_result_slot,
                claim_token=claim_token,
                expected_task=task.value,
                run_id=job.run_id,
                chat_id=job.chat_id or "",
                sender_id=job.sender_id or "",
                execute=lambda: execute_port.execute_in_project_unregistered(
                    analysis_input,
                    context=context,
                ),
            )
            report_input = self._audited_report_input(analysis)
            report = _resolve_or_execute_slot(
                job=job,
                slot="REPORT",
                result_id=job.report_result_id,
                expected_tool_name=StandardToolName.GENERATE_AUDITED_REPORT.value,
                expected_tool_version=AUDITED_REPORT_TOOL_VERSION,
                expected_input_hash=sha256_canonical(
                    report_input.model_dump(mode="json")
                ),
                expected_reference_id=analysis.result_id,
                context=context,
                resolver=self._project_ledger,
                committer=committer.commit_feishu_result_slot,
                claim_token=claim_token,
                expected_task=task.value,
                run_id=job.run_id,
                chat_id=job.chat_id or "",
                sender_id=job.sender_id or "",
                execute=lambda: self._audited_report_unregistered(
                    analysis,
                    report_input=report_input,
                ),
            )
            expected_version = (
                ADVANCED_RUL_PREDICTION_TOOL_VERSION
                if task is StandardToolName.PREDICT_CYCLE_LIFE
                else ADVANCED_SOH_PREDICTION_TOOL_VERSION
            )
            if analysis.tool_version != expected_version:
                raise ValueError("project model ToolResult contract is invalid")
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
            slots_committed=True,
        )

    def _resolve_record_batch_id(
        self,
        *,
        context: VerifiedProjectInvocationContext,
        job: FeishuAnalysisJobRecord,
        registration: CanonicalCsvBatchRegistration,
    ) -> str:
        provisioned_id: str | None = None
        content_batch_id = getattr(job, "record_batch_id", None)
        if self._batch_provisioner is not None and isinstance(
            content_batch_id, str
        ) and content_batch_id.strip():
            provisioned = self._batch_provisioner.provision_frozen_binding(
                context,
                content_batch_id=content_batch_id,
                registration=registration,
                now=self._clock(),
            )
            candidate = getattr(provisioned, "record_batch_id", None)
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("project batch provisioner returned an invalid binding")
            provisioned_id = candidate
        resolved_id = self._batch_resolver.resolve(context, registration)
        if provisioned_id is not None and resolved_id != provisioned_id:
            raise ValueError("provisioned project batch identity changed")
        return resolved_id

    def _audited_report_input(
        self,
        analysis: ToolResult,
    ) -> GenerateAuditedReportToolInput:
        if analysis.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value:
            paths = (
                "values.artifact.cycle_life_prediction.predicted_cycle",
                "values.artifact.derived_remaining_cycles",
            )
        elif analysis.tool_name == StandardToolName.PREDICT_SOH_TRAJECTORY.value:
            paths = _soh_report_paths(analysis)
        else:
            raise ValueError("project report task is unsupported")
        return GenerateAuditedReportToolInput(
            report_kind=ReportKind.LIFETIME_DECISION,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                    numeric_evidence=tuple(
                        NumericEvidenceReference(
                            result_id=analysis.result_id,
                            json_path=path,
                        )
                        for path in paths
                    ),
                ),
            ),
            upstream_result_ids=(analysis.result_id,),
        )

    def _audited_report_unregistered(
        self,
        analysis: ToolResult,
        *,
        report_input: GenerateAuditedReportToolInput | None = None,
    ) -> ToolResult:
        resolved_input = report_input or self._audited_report_input(analysis)
        return execute_generate_audited_report_tool(
            resolved_input,
            audit_ledger=AuditLedger((analysis,)),
            clock=self._clock,
        )


def _resolve_or_execute_slot(
    *,
    job: FeishuAnalysisJobRecord,
    slot: str,
    result_id: str | None,
    expected_tool_name: str,
    expected_tool_version: str,
    expected_input_hash: str,
    expected_reference_id: str,
    context: VerifiedProjectInvocationContext,
    resolver: ProjectResultLedger,
    committer: Callable[..., ToolResult],
    claim_token: str,
    expected_task: str,
    run_id: str,
    chat_id: str,
    sender_id: str,
    execute: Callable[[], ToolResult],
) -> ToolResult:
    if result_id is not None:
        restored = ToolResult.model_validate(
            resolver.resolve_registered_result(context, result_id).model_dump(
                mode="json"
            )
        )
        _validate_slot_result(
            restored,
            slot=slot,
            expected_tool_name=expected_tool_name,
            expected_tool_version=expected_tool_version,
            expected_input_hash=expected_input_hash,
            expected_reference_id=expected_reference_id,
        )
        return restored
    computed = ToolResult.model_validate(execute().model_dump(mode="json"))
    _validate_slot_result(
        computed,
        slot=slot,
        expected_tool_name=expected_tool_name,
        expected_tool_version=expected_tool_version,
        expected_input_hash=expected_input_hash,
        expected_reference_id=expected_reference_id,
    )
    return ToolResult.model_validate(
        committer(
            context=context,
            job_id=job.job_id,
            claim_token=claim_token,
            slot=slot,
            expected_task=expected_task,
            run_id=run_id,
            chat_id=chat_id,
            sender_id=sender_id,
            result=computed,
        ).model_dump(mode="json")
    )


def _validate_slot_result(
    result: ToolResult,
    *,
    slot: str,
    expected_tool_name: str,
    expected_tool_version: str,
    expected_input_hash: str,
    expected_reference_id: str,
) -> None:
    if (
        result.tool_name != expected_tool_name
        or result.tool_version != expected_tool_version
        or result.input_hash != expected_input_hash
    ):
        raise ValueError("Feishu result slot contract is invalid")
    if slot == "REPORT":
        if result.values.get("upstream_result_ids") != [expected_reference_id]:
            raise ValueError("Feishu report slot is not bound to the analysis result")
        return
    artifact = result.values.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Feishu result slot artifact is invalid")
    reference_field = "record_batch_id" if slot == "PREPARED" else "upstream_result_id"
    if artifact.get(reference_field) != expected_reference_id:
        raise ValueError("Feishu result slot reference is invalid")


def _rul_role(cutoff_cycle: int) -> AdvancedModelRouteRole:
    if cutoff_cycle == 20:
        return AdvancedModelRouteRole.DEFAULT
    if cutoff_cycle in {50, 100, 150}:
        return AdvancedModelRouteRole.POINT_ACCURACY
    raise FeishuProjectModelRejected(
        "PROJECT_MODEL_CUTOFF_NOT_SUPPORTED",
        "Advanced RUL requires cutoff 20, 50, 100, or 150",
    )


def _project_model_role(
    *,
    task: StandardToolName,
    cutoff_cycle: int,
) -> AdvancedModelRouteRole:
    if task is StandardToolName.PREDICT_CYCLE_LIFE:
        return _rul_role(cutoff_cycle)
    if task is StandardToolName.PREDICT_SOH_TRAJECTORY and cutoff_cycle in {
        20,
        50,
        100,
        150,
    }:
        return AdvancedModelRouteRole.MEAN_ACCURACY
    raise FeishuProjectModelRejected(
        "PROJECT_MODEL_CUTOFF_NOT_SUPPORTED",
        "Advanced SOH requires cutoff 20, 50, 100, or 150",
    )


def _soh_report_paths(analysis: ToolResult) -> tuple[str, str]:
    artifact = analysis.values.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Advanced SOH report artifact is invalid")
    cycles = artifact.get("prediction_cycles")
    predicted_soh = artifact.get("predicted_soh")
    if (
        not isinstance(cycles, list)
        or not isinstance(predicted_soh, list)
        or not cycles
        or len(cycles) != len(predicted_soh)
    ):
        raise ValueError("Advanced SOH report trajectory is invalid")
    last_index = len(predicted_soh) - 1
    return (
        "values.artifact.horizon_end_cycle",
        f"values.artifact.predicted_soh.{last_index}",
    )


def _is_anchored_aily_project_job(
    job: FeishuEventReceipt,
    source: FeishuEventReceipt | None,
) -> bool:
    return bool(
        source is not None
        and job.event_type == "aily.analysis_task.create_v2"
        and job.source_job_id == source.job_id
        and job.task_type
        in {
            StandardToolName.PREDICT_CYCLE_LIFE.value,
            StandardToolName.PREDICT_SOH_TRAJECTORY.value,
            StandardToolName.COMPARE_OPERATION_SCENARIOS.value,
            StandardToolName.PROJECT_STORAGE_LIFETIME.value,
        }
        and job.message_id is None
        and job.file_key is None
        and source.job_origin == "FEISHU"
        and source.event_type == "im.message.receive_v1"
        and source.source_job_id is None
        and source.task_type == StandardToolName.PREDICT_CYCLE_LIFE.value
        and source.job_status == "SUCCEEDED"
        and source.validation_result_id is not None
        and job.record_batch_id == source.record_batch_id
        and job.cell_reference == source.cell_reference
        and job.input_file_sha256 == source.input_file_sha256
        and job.chat_id == source.chat_id
        and job.sender_id == source.sender_id
        and job.receive_id_type == source.receive_id_type
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
