"""Explicit production assembly for the audited Feishu and Aily scenario path.

Importing this module performs no network access, service start, model training,
or raw-data processing. The factory binds existing components to one SQL audit
and job boundary so the HTTP process and Celery worker can be constructed from
the same operator-owned settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import SecretStr

from quanxin_life.api.aily import (
    AilyConnectorConfig,
    AilyHttpAdapter,
    create_aily_http_adapter,
)
from quanxin_life.api.feishu import FeishuHttpAdapter, create_feishu_http_adapter
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.feishu_project_models import (
    FeishuProjectModelExecutor,
    FeishuProjectRecordBatchResolver,
    FeishuProjectResultResolver,
    ProjectToolInvocationPort,
)
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    FileSystemVerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import ProjectResultLedger, SqlAuditLedger
from quanxin_life.core import SourceKind, ToolResult
from quanxin_life.infrastructure.feishu_queue import (
    CeleryApplication,
    CeleryFeishuJobQueue,
)
from quanxin_life.integrations.feishu.aily_scenarios import (
    AilyScenarioReferenceUseAuthorizer,
    SqlAlchemyAilyScenarioContextGateway,
)
from quanxin_life.integrations.feishu.aily_tasks import (
    AilyAnalysisJobDelivery,
    SqlAlchemyAilyAnalysisTaskGateway,
)
from quanxin_life.integrations.feishu.attachments import (
    FeishuAttachmentPolicy,
    VerifiedFeishuAttachment,
)
from quanxin_life.integrations.feishu.bitable import FeishuBitableWriter
from quanxin_life.integrations.feishu.cards import AuditedCardBuilder
from quanxin_life.integrations.feishu.client import (
    FeishuClient,
    FeishuClientConfig,
    FeishuTransport,
)
from quanxin_life.integrations.feishu.decryptor import FeishuAesCbcDecryptor
from quanxin_life.integrations.feishu.events import FeishuEventProcessor
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobDelivery,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobWorker,
    OriginAwareAnalysisJobDelivery,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.report_delivery import FeishuReportDelivery
from quanxin_life.integrations.feishu.routing import (
    FeishuEventReference,
    FeishuInboundEventKind,
)
from quanxin_life.integrations.feishu.scenario_authorization import (
    AuditedScenarioResultAuthorizer,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.scenario_plot import FeishuScenarioPlotter
from quanxin_life.integrations.feishu.scenario_reports import (
    FeishuScenarioReportResultFactory,
)
from quanxin_life.integrations.feishu.security import (
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)
from quanxin_life.integrations.feishu.sqlalchemy_receipts import (
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
    FeishuWorkflowRejected,
)
from quanxin_life.persistence.database import SessionFactory
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.tools import StandardToolName, ToolRegistry
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.blast_scenarios import (
    register_compare_operation_scenarios_tool,
    register_project_storage_lifetime_tool,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

Clock = Callable[[], datetime]
FeishuCsvRegistrationResolver = Callable[
    [FeishuAnalysisJobRecord, VerifiedFeishuAttachment, datetime],
    CanonicalCsvBatchRegistration,
]

_SCENARIO_TASKS = frozenset(
    {
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    }
)


class _CeleryApplication(CeleryApplication, Protocol):
    """Structural queue port; task registration remains owned by the runtime."""


class _RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class RegisteredFeishuCsvRegistrationResolver:
    """Resolve only operator-approved canonical CSV bytes by exact SHA-256."""

    def __init__(
        self,
        registrations: Mapping[str, CanonicalCsvBatchRegistration],
    ) -> None:
        validated: dict[str, CanonicalCsvBatchRegistration] = {}
        for payload_sha256, registration in registrations.items():
            copied = CanonicalCsvBatchRegistration.model_validate(
                registration.model_dump(mode="json")
            )
            if copied.metadata.source_sha256 != payload_sha256:
                raise ValueError(
                    "Feishu CSV registration metadata SHA-256 does not match payload"
                )
            observed_hashes = {
                item.sha256
                for item in copied.provenance
                if item.source_kind is SourceKind.OBSERVED
            }
            if payload_sha256 not in observed_hashes:
                raise ValueError(
                    "Feishu CSV registration provenance SHA-256 does not match payload"
                )
            validated[payload_sha256] = copied
        self._registrations = validated

    def __call__(
        self,
        _job: FeishuAnalysisJobRecord,
        attachment: VerifiedFeishuAttachment,
        _received_at: datetime,
    ) -> CanonicalCsvBatchRegistration:
        registration = self._registrations.get(attachment.sha256)
        if registration is None:
            raise ValueError("canonical CSV payload SHA-256 is not registered")
        return CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )


@dataclass(frozen=True, slots=True)
class FeishuAilyAssemblyConfig:
    """Secret-safe operator inputs for the opt-in production bundle."""

    app_id: str
    app_secret: SecretStr
    verification_token: SecretStr
    encrypt_key: SecretStr
    aily_connector_api_key: SecretStr
    bitable_app_token: str
    bitable_table_id: str
    external_https_base_url: str
    data_root: Path
    allow_candidate_scenario_execution: bool = False
    allow_candidate_scenario_results: bool = False

    def __post_init__(self) -> None:
        for field_name in ("app_id", "bitable_app_token", "bitable_table_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        for field_name in (
            "app_secret",
            "verification_token",
            "encrypt_key",
            "aily_connector_api_key",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, SecretStr) or not value.get_secret_value().strip():
                raise ValueError(f"{field_name} must not be blank")
        if not isinstance(self.allow_candidate_scenario_execution, bool):
            raise TypeError("allow_candidate_scenario_execution must be a bool")
        if not isinstance(self.allow_candidate_scenario_results, bool):
            raise TypeError("allow_candidate_scenario_results must be a bool")
        _https_base_url(self.external_https_base_url)


@dataclass(frozen=True, slots=True)
class FeishuAilyComponents:
    """Detached runtime components sharing one SQL ledger and durable job store."""

    audit_ledger: SqlAuditLedger
    tool_service: ToolInvocationService
    batch_store: FileSystemVerifiedEarlyCycleBatchStore
    receipt_store: SqlAlchemyFeishuReceiptStore
    job_store: SqlAlchemyFeishuJobStore
    scenario_context_store: SqlAlchemyFeishuScenarioContextStore
    event_processor: FeishuEventProcessor
    feishu_http_adapter: FeishuHttpAdapter
    aily_http_adapter: AilyHttpAdapter
    aily_task_gateway: SqlAlchemyAilyAnalysisTaskGateway
    worker: FeishuAnalysisJobWorker
    registration_resolver: FeishuCsvRegistrationResolver


@dataclass(frozen=True, slots=True)
class FeishuProjectModelDependencies:
    """Existing competition runtime services reused by the Feishu model path."""

    context_service: ProjectInvocationContextService
    project_ledger: ProjectResultLedger
    project_tool_service: ProjectToolInvocationPort


class _RejectingModelRouteAuthorizer:
    """Keep existing deep-model routes unavailable until their project grant exists."""

    def authorize(
        self,
        *,
        tool_name: StandardToolName,
        validation_result: object,
    ) -> None:
        del tool_name, validation_result
        raise FeishuWorkflowRejected("MODEL_ROUTE_NOT_ACTIVATED")


class _ScenarioInputBinder:
    """Pass only the already persisted typed scenario input to its tool."""

    def bind_analysis_input(
        self,
        *,
        task: FeishuAnalysisTask,
        validation_result: object,
        validation_input: Mapping[str, object],
        requested_analysis_input: Mapping[str, object],
    ) -> Mapping[str, object]:
        del validation_result, validation_input
        if task not in _SCENARIO_TASKS:
            raise FeishuWorkflowRejected("MODEL_ROUTE_NOT_ACTIVATED")
        return dict(requested_analysis_input)


def create_feishu_aily_components(
    *,
    session_factory: SessionFactory,
    celery_app: _CeleryApplication,
    config: FeishuAilyAssemblyConfig,
    feishu_transport: FeishuTransport | None = None,
    registration_resolver: FeishuCsvRegistrationResolver | None = None,
    scenario_reference_use_authorizer: (
        AilyScenarioReferenceUseAuthorizer | None
    ) = None,
    project_model_dependencies: FeishuProjectModelDependencies | None = None,
    clock: Clock | None = None,
) -> FeishuAilyComponents:
    """Assemble Feishu callbacks, Aily facade, worker, tools, and delivery ports."""

    now = clock or _utc_now
    checked_base_url = _https_base_url(config.external_https_base_url)
    batch_store = FileSystemVerifiedEarlyCycleBatchStore(config.data_root)
    audit_ledger = SqlAuditLedger(session_factory, clock=now)
    receipt_store = SqlAlchemyFeishuReceiptStore(session_factory)
    job_store = SqlAlchemyFeishuJobStore(session_factory)
    scenario_context_store = SqlAlchemyFeishuScenarioContextStore(session_factory)
    result_resolver: _RegisteredResultResolver = audit_ledger
    project_model_executor = None
    if project_model_dependencies is not None:
        result_resolver = FeishuProjectResultResolver(
            session_factory=session_factory,
            global_resolver=audit_ledger,
            context_service=project_model_dependencies.context_service,
            project_ledger=project_model_dependencies.project_ledger,
        )
        project_model_executor = FeishuProjectModelExecutor(
            context_service=project_model_dependencies.context_service,
            batch_resolver=FeishuProjectRecordBatchResolver(session_factory),
            project_tool_service=project_model_dependencies.project_tool_service,
            project_ledger=project_model_dependencies.project_ledger,
            clock=now,
        )

    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_compare_operation_scenarios_tool(
        registry,
        context_resolver=scenario_context_store,
        allow_candidate_execution=config.allow_candidate_scenario_execution,
        clock=now,
    )
    register_project_storage_lifetime_tool(
        registry,
        context_resolver=scenario_context_store,
        allow_candidate_execution=config.allow_candidate_scenario_execution,
        clock=now,
    )
    register_generate_audited_report_tool(
        registry,
        audit_ledger=audit_ledger,
        clock=now,
    )
    tool_service = ToolInvocationService(
        registry=registry,
        audit_ledger=audit_ledger,
    )

    client = FeishuClient(
        FeishuClientConfig(
            app_id=config.app_id,
            app_secret=config.app_secret,
        ),
        transport=feishu_transport,
    )
    queue = CeleryFeishuJobQueue(app=celery_app)
    scenario_gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=scenario_context_store,
        batch_store=batch_store,
        created_by_reference="aily-connector-v1",
        reference_use_authorizer=scenario_reference_use_authorizer,
        clock=now,
    )
    aily_task_gateway = SqlAlchemyAilyAnalysisTaskGateway(
        job_store=job_store,
        scenario_context_store=scenario_context_store,
        queue=queue,
        clock=now,
    )

    decryptor = FeishuAesCbcDecryptor(config.encrypt_key)
    event_processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(
            FeishuWebhookSecrets(
                verification_token=config.verification_token,
                encrypt_key=config.encrypt_key,
            )
        ),
        verification_token=config.verification_token,
        receipts=receipt_store,
        decryptor=decryptor,
    )
    job_router = SqlAlchemyFeishuJobRouter(
        job_store,
        queue=queue,
        task_resolver=_resolve_feishu_task,
        clock=now,
    )
    feishu_http_adapter = create_feishu_http_adapter(
        event_processor,
        router=job_router,
        now_factory=now,
    )

    authorizer = AuditedScenarioResultAuthorizer(
        result_resolver=result_resolver,
        allow_candidate_results=config.allow_candidate_scenario_results,
    )
    artifact_exporter = AuditedReportArtifactExporter(result_resolver)
    bitable_writer = FeishuBitableWriter(
        client,
        app_token=config.bitable_app_token,
        table_id=config.bitable_table_id,
    )
    feishu_delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=AuditedCardBuilder(
            result_resolver,
            authorizer=authorizer,
            binding_verifier=job_store,
        ),
        bitable_writer=bitable_writer,
        report_delivery=FeishuReportDelivery(
            artifact_exporter,
            client,
            result_resolver=result_resolver,
        ),
        result_authorizer=authorizer,
        scenario_plotter=FeishuScenarioPlotter(),
    )
    aily_delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable_writer,
        result_authorizer=authorizer,
        report_link_factory=lambda job, report: (
            f"{checked_base_url}/v1/aily/analysis-tasks/{job.run_id}"
            f"/reports/{report.result_id}"
        ),
    )
    workflow = FeishuAnalysisWorkflow(
        tool_service,
        route_authorizer=_RejectingModelRouteAuthorizer(),
        input_binder=_ScenarioInputBinder(),
    )
    scenario_report_factory = FeishuScenarioReportResultFactory(tool_service)

    def build_scenario_report(
        job: FeishuAnalysisJobRecord,
        result: ToolResult,
    ) -> ToolResult:
        return scenario_report_factory(job, result)

    resolved_registration = registration_resolver or _rejecting_registration_resolver
    worker = FeishuAnalysisJobWorker(
        job_store,
        client=client,
        attachment_policy=FeishuAttachmentPolicy(),
        batch_store=batch_store,
        registration_resolver=resolved_registration,
        analysis_input_factory=lambda _job, batch: _validation_input(
            batch,
            validated_at=_utc(now()),
        ),
        scenario_input_resolver=scenario_context_store,
        workflow=workflow,
        result_resolver=result_resolver,
        report_result_factory=build_scenario_report,
        project_model_executor=project_model_executor,
        delivery=OriginAwareAnalysisJobDelivery(
            feishu_delivery=feishu_delivery,
            aily_delivery=aily_delivery,
        ),
        clock=now,
    )
    aily_http_adapter = create_aily_http_adapter(
        AilyConnectorConfig(api_key=config.aily_connector_api_key),
        gateway=aily_task_gateway,
        scenario_context_gateway=scenario_gateway,
        audit_ledger=audit_ledger,
        report_exporter=artifact_exporter,
        result_authorizer=authorizer,
    )
    return FeishuAilyComponents(
        audit_ledger=audit_ledger,
        tool_service=tool_service,
        batch_store=batch_store,
        receipt_store=receipt_store,
        job_store=job_store,
        scenario_context_store=scenario_context_store,
        event_processor=event_processor,
        feishu_http_adapter=feishu_http_adapter,
        aily_http_adapter=aily_http_adapter,
        aily_task_gateway=aily_task_gateway,
        worker=worker,
        registration_resolver=resolved_registration,
    )


def _validation_input(
    batch: VerifiedEarlyCycleBatch,
    *,
    validated_at: datetime,
) -> dict[str, object]:
    return ValidateBatteryDataToolInput(
        records=batch.records,
        data_version=batch.data_version,
        feature_version=batch.feature_config.feature_version,
        provenance=batch.provenance,
        validated_at=validated_at,
    ).model_dump(mode="json")


def _rejecting_registration_resolver(
    _job: FeishuAnalysisJobRecord,
    _attachment: VerifiedFeishuAttachment,
    _received_at: datetime,
) -> CanonicalCsvBatchRegistration:
    raise ValueError("trusted Feishu CSV registration metadata is not configured")


def _resolve_feishu_task(event: FeishuEventReference) -> FeishuAnalysisTask:
    if event.kind is FeishuInboundEventKind.FILE:
        return FeishuAnalysisTask.PREDICT_CYCLE_LIFE
    if event.kind is FeishuInboundEventKind.CARD_ACTION:
        raw_task = event.action_value.get("task_type")
        if not isinstance(raw_task, str):
            raise ValueError("scenario card action has an invalid task reference")
        try:
            task = FeishuAnalysisTask(raw_task)
        except (TypeError, ValueError) as exc:
            raise ValueError("scenario card action has an invalid task reference") from exc
        if task not in _SCENARIO_TASKS:
            raise ValueError("card action may dispatch only a scenario analysis task")
        return task
    raise ValueError("Feishu event does not identify a supported analysis task")


def _https_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("external integration base URL must be a pure HTTPS origin")
    return value.removesuffix("/")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Feishu/Aily assembly clock must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "FeishuAilyAssemblyConfig",
    "FeishuAilyComponents",
    "FeishuCsvRegistrationResolver",
    "FeishuProjectModelDependencies",
    "RegisteredFeishuCsvRegistrationResolver",
    "create_feishu_aily_components",
]
