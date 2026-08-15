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
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

from pydantic import SecretStr

from quanxin_life.api.aily import (
    AilyConnectorConfig,
    AilyHttpAdapter,
    create_aily_http_adapter,
)
from quanxin_life.api.feishu import FeishuHttpAdapter, create_feishu_http_adapter
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.aily_mcp_authorization import (
    AilyMcpIdentityBindings,
    ProjectBoundAilyMcpCallerAuthorizer,
)
from quanxin_life.application.battery_csv_mapping import (
    BatteryCsvMappingProfile,
    ReviewedBatteryCsvNormalizer,
)
from quanxin_life.application.engineering_recommendation_rules import (
    ReviewedEngineeringRecommendationRulesetRegistry,
)
from quanxin_life.application.feishu_engineering_recommendation import (
    FeishuEngineeringRecommendationExecutor,
)
from quanxin_life.application.feishu_project_models import (
    FeishuProjectModelExecutor,
    FeishuProjectRecordBatchResolver,
    FeishuProjectResultResolver,
    ProjectToolInvocationPort,
)
from quanxin_life.application.feishu_recheck_authorization import (
    ProjectBoundRecheckActionAuthorizationVerifier,
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
    AilyDataIdentityResolver,
    SqlAlchemyAilyAnalysisTaskGateway,
)
from quanxin_life.integrations.feishu.analysis_plots import FeishuAnalysisPlotter
from quanxin_life.integrations.feishu.attachments import (
    FeishuAttachmentPolicy,
    VerifiedFeishuAttachment,
)
from quanxin_life.integrations.feishu.bitable import (
    CHINESE_ANALYSIS_BITABLE_PROFILE,
    BitableMediaUploader,
    FeishuBitableWriter,
)
from quanxin_life.integrations.feishu.cards import AuditedCardBuilder
from quanxin_life.integrations.feishu.client import (
    FeishuClient,
    FeishuClientConfig,
    FeishuTransport,
)
from quanxin_life.integrations.feishu.decryptor import FeishuAesCbcDecryptor
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioRegistry,
)
from quanxin_life.integrations.feishu.events import FeishuEventProcessor
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobDelivery,
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
    FeishuAnalysisJobWorker,
    OriginAwareAnalysisJobDelivery,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.recheck_actions import (
    CHINESE_RECHECK_ACTION_FIELD_PROFILE,
    FeishuRecheckActionService,
    RecheckActionReceipt,
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
from quanxin_life.integrations.feishu.scenario_reports import (
    FeishuScenarioReportResultFactory,
)
from quanxin_life.integrations.feishu.security import (
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)
from quanxin_life.integrations.feishu.sibling_planner import (
    ProactiveFeishuSiblingPlanner,
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

if TYPE_CHECKING:
    from quanxin_life.api.aily_mcp import AilyMcpAdapter

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
        resolved = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        mapping_evidence = getattr(attachment, "mapping_evidence", None)
        if mapping_evidence is None:
            return resolved
        metadata_payload = resolved.metadata.model_dump(mode="json")
        metadata_payload["ingestion_parameters"] = {
            **resolved.metadata.ingestion_parameters,
            **mapping_evidence,
        }
        payload = resolved.model_dump(mode="json")
        payload["metadata"] = metadata_payload
        return CanonicalCsvBatchRegistration.model_validate(payload)


@dataclass(frozen=True, slots=True)
class AilyMcpAssemblyConfig:
    """Resolved operator inputs for the optional protected Aily MCP endpoint."""

    endpoint_token: SecretStr
    allowed_source_ips: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    identity_bindings: AilyMcpIdentityBindings

    def __post_init__(self) -> None:
        if not isinstance(self.identity_bindings, AilyMcpIdentityBindings):
            raise TypeError("identity_bindings must be AilyMcpIdentityBindings")
        from quanxin_life.api.aily_mcp import AilyMcpConfig

        AilyMcpConfig(
            endpoint_token=self.endpoint_token,
            allowed_source_ips=self.allowed_source_ips,
            allowed_hosts=self.allowed_hosts,
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
    recheck_table_id: str | None = None
    recheck_permission_reference: str | None = None
    aily_mcp: AilyMcpAssemblyConfig | None = None

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
        recheck_values = (
            self.recheck_table_id,
            self.recheck_permission_reference,
        )
        if any(value is None for value in recheck_values) and any(
            value is not None for value in recheck_values
        ):
            raise ValueError(
                "recheck table and permission reference must be configured together"
            )
        for field_name in ("recheck_table_id", "recheck_permission_reference"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must not be blank")
        if self.aily_mcp is not None and not isinstance(
            self.aily_mcp,
            AilyMcpAssemblyConfig,
        ):
            raise TypeError("aily_mcp must be an AilyMcpAssemblyConfig")
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
    aily_mcp_adapter: AilyMcpAdapter | None
    aily_task_gateway: SqlAlchemyAilyAnalysisTaskGateway
    sibling_job_service: SqlAlchemyFeishuSiblingJobService
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


class _RejectingAilyDataIdentityResolver:
    def resolve_source_job(self, **_: object) -> None:
        raise ValueError("project-bound Aily data identity is not configured")


class _ProjectBoundAilyDataIdentityResolver:
    """Revalidate one completed Feishu upload before any Aily task is staged."""

    def __init__(
        self,
        *,
        job_store: SqlAlchemyFeishuJobStore,
        batch_store: FileSystemVerifiedEarlyCycleBatchStore,
        context_service: ProjectInvocationContextService,
        batch_resolver: FeishuProjectRecordBatchResolver,
    ) -> None:
        self._job_store = job_store
        self._batch_store = batch_store
        self._context_service = context_service
        self._batch_resolver = batch_resolver

    def resolve_source_job(
        self,
        *,
        source_run_id: str,
        data_batch_id: str,
    ) -> None:
        try:
            source = self._job_store.get(source_run_id)
            if (
                source.job_origin is not FeishuAnalysisJobOrigin.FEISHU
                or source.event_type != "im.message.receive_v1"
                or source.task_type is not FeishuAnalysisTask.PREDICT_CYCLE_LIFE
                or source.source_job_id is not None
                or source.job_status is not FeishuAnalysisJobStatus.SUCCEEDED
                or source.record_batch_id != data_batch_id
                or source.validation_result_id is None
                or not source.chat_id
                or not source.sender_id
                or not source.cell_reference
                or not source.input_file_sha256
            ):
                raise ValueError("Aily source upload is not authorized")
            context = self._context_service.resolve_feishu(
                chat_id=source.chat_id,
                sender_open_id=source.sender_id,
            )
            batch = self._batch_store.resolve_verified_early_cycle_batch(
                data_batch_id
            )
            canonical_sha256 = _source_canonical_sha256(source)
            if (
                batch.metadata.cell_id != source.cell_reference
                or batch.metadata.source_sha256 != canonical_sha256
            ):
                raise ValueError("Aily source upload data identity changed")
            registration = CanonicalCsvBatchRegistration(
                metadata=batch.metadata,
                feature_config=batch.feature_config,
                data_version=batch.data_version,
                split_version=batch.split_version,
                provenance=batch.provenance,
            )
            self._batch_resolver.resolve(context, registration)
        except (KeyError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Aily data identity is not authorized") from exc


class _ConfiguredRecheckActionGateway:
    """Inject operator-owned permission evidence into reference-only Aily calls."""

    def __init__(
        self,
        service: FeishuRecheckActionService,
        *,
        permission_reference: str,
    ) -> None:
        self._service = service
        self._permission_reference = permission_reference

    def create_recheck_action(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        responsibility_reference: str,
    ) -> RecheckActionReceipt:
        return self._service.create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference=responsibility_reference,
            permission_reference=self._permission_reference,
        )


def _source_canonical_sha256(source: FeishuAnalysisJobRecord) -> str:
    raw_sha256 = source.input_file_sha256
    if raw_sha256 is None:
        raise ValueError("Aily source upload SHA is unavailable")
    if source.csv_mapping_status is None:
        if source.csv_mapping_evidence is not None:
            raise ValueError("Aily source mapping evidence is inconsistent")
        return raw_sha256
    if source.csv_mapping_status != "MAPPED" or source.csv_mapping_evidence is None:
        raise ValueError("Aily source mapping evidence is not accepted")
    mapped_raw_sha256 = source.csv_mapping_evidence.get("raw_sha256")
    canonical_sha256 = source.csv_mapping_evidence.get("canonical_sha256")
    if mapped_raw_sha256 != raw_sha256 or not isinstance(canonical_sha256, str):
        raise ValueError("Aily source mapping evidence changed")
    return canonical_sha256


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
    csv_mapping_profiles: tuple[BatteryCsvMappingProfile, ...] = (),
    default_scenarios: ReviewedDefaultScenarioRegistry | None = None,
    recommendation_rulesets: (
        ReviewedEngineeringRecommendationRulesetRegistry | None
    ) = None,
    aily_data_identity_resolver: AilyDataIdentityResolver | None = None,
    clock: Clock | None = None,
) -> FeishuAilyComponents:
    """Assemble Feishu callbacks, Aily facade, worker, tools, and delivery ports."""

    now = clock or _utc_now
    checked_base_url = _https_base_url(config.external_https_base_url)
    if config.recheck_table_id is not None and project_model_dependencies is None:
        raise ValueError("recheck actions require project model dependencies")
    if config.aily_mcp is not None and project_model_dependencies is None:
        raise ValueError("Aily MCP requires project model dependencies")
    batch_store = FileSystemVerifiedEarlyCycleBatchStore(config.data_root)
    audit_ledger = SqlAuditLedger(session_factory, clock=now)
    receipt_store = SqlAlchemyFeishuReceiptStore(session_factory)
    job_store = SqlAlchemyFeishuJobStore(session_factory)
    scenario_context_store = SqlAlchemyFeishuScenarioContextStore(session_factory)
    result_resolver: _RegisteredResultResolver = audit_ledger
    project_model_executor = None
    engineering_recommendation_executor = None
    data_identity_resolver: AilyDataIdentityResolver
    if project_model_dependencies is not None:
        if aily_data_identity_resolver is not None:
            raise ValueError(
                "explicit Aily data identity resolver conflicts with project runtime"
            )
        project_batch_resolver = FeishuProjectRecordBatchResolver(session_factory)
        result_resolver = FeishuProjectResultResolver(
            session_factory=session_factory,
            global_resolver=audit_ledger,
            context_service=project_model_dependencies.context_service,
            project_ledger=project_model_dependencies.project_ledger,
        )
        project_model_executor = FeishuProjectModelExecutor(
            context_service=project_model_dependencies.context_service,
            batch_resolver=project_batch_resolver,
            project_tool_service=project_model_dependencies.project_tool_service,
            project_ledger=project_model_dependencies.project_ledger,
            clock=now,
        )
        if recommendation_rulesets is not None:
            engineering_recommendation_executor = (
                FeishuEngineeringRecommendationExecutor(
                    session_factory=session_factory,
                    context_service=project_model_dependencies.context_service,
                    project_ledger=project_model_dependencies.project_ledger,
                    result_resolver=result_resolver,
                    ruleset_resolver=recommendation_rulesets,
                    clock=now,
                )
            )
        data_identity_resolver = _ProjectBoundAilyDataIdentityResolver(
            job_store=job_store,
            batch_store=batch_store,
            context_service=project_model_dependencies.context_service,
            batch_resolver=project_batch_resolver,
        )
    else:
        if recommendation_rulesets is not None:
            raise ValueError(
                "engineering recommendations require project model dependencies"
            )
        data_identity_resolver = (
            aily_data_identity_resolver or _RejectingAilyDataIdentityResolver()
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
    sibling_job_service = SqlAlchemyFeishuSiblingJobService(
        job_store,
        queue=queue,
        clock=now,
    )
    sibling_planner = ProactiveFeishuSiblingPlanner(
        batch_store=batch_store,
        context_store=scenario_context_store,
        sibling_jobs=sibling_job_service,
        default_scenarios=default_scenarios,
        clock=now,
        recommendation_rulesets=recommendation_rulesets,
    )
    scenario_gateway = SqlAlchemyAilyScenarioContextGateway(
        context_store=scenario_context_store,
        batch_store=batch_store,
        data_identity_resolver=data_identity_resolver,
        reference_use_authorizer=scenario_reference_use_authorizer,
        clock=now,
    )
    aily_task_gateway = SqlAlchemyAilyAnalysisTaskGateway(
        job_store=job_store,
        scenario_context_store=scenario_context_store,
        data_identity_resolver=data_identity_resolver,
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
        field_profile=CHINESE_ANALYSIS_BITABLE_PROFILE,
    )
    bitable_media_uploader = BitableMediaUploader(
        client,
        app_token=config.bitable_app_token,
    )
    analysis_plotter = FeishuAnalysisPlotter()
    recheck_action_gateway = None
    if config.recheck_table_id is not None:
        assert config.recheck_permission_reference is not None
        assert project_model_dependencies is not None
        recheck_action_gateway = _ConfiguredRecheckActionGateway(
            FeishuRecheckActionService(
                client=client,
                app_token=config.bitable_app_token,
                table_id=config.recheck_table_id,
                authorization_verifier=(
                    ProjectBoundRecheckActionAuthorizationVerifier(
                        session_factory=session_factory,
                        job_store=job_store,
                        result_resolver=result_resolver,
                        result_authorizer=authorizer,
                        context_service=project_model_dependencies.context_service,
                        expected_permission_reference=(
                            config.recheck_permission_reference
                        ),
                    )
                ),
                receipt_store=receipt_store,
                clock=now,
                field_profile=CHINESE_RECHECK_ACTION_FIELD_PROFILE,
            ),
            permission_reference=config.recheck_permission_reference,
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
        scenario_plotter=analysis_plotter,
        analysis_plotter=analysis_plotter,
        bitable_curve_plotter=analysis_plotter,
        bitable_media_uploader=bitable_media_uploader,
    )
    aily_delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable_writer,
        result_authorizer=authorizer,
        report_link_factory=lambda job, report: (
            f"{checked_base_url}/v1/aily/analysis-tasks/{job.run_id}"
            f"/reports/{report.result_id}"
        ),
        analysis_plotter=analysis_plotter,
        bitable_media_uploader=bitable_media_uploader,
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
    csv_normalizer = (
        ReviewedBatteryCsvNormalizer(csv_mapping_profiles)
        if csv_mapping_profiles
        else None
    )
    worker = FeishuAnalysisJobWorker(
        job_store,
        client=client,
        attachment_policy=FeishuAttachmentPolicy(),
        csv_normalizer=csv_normalizer,
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
        engineering_recommendation_executor=(
            engineering_recommendation_executor
        ),
        delivery=OriginAwareAnalysisJobDelivery(
            feishu_delivery=feishu_delivery,
            aily_delivery=aily_delivery,
        ),
        sibling_planner=sibling_planner,
        sibling_dispatcher=sibling_job_service,
        aily_data_identity_resolver=data_identity_resolver,
        clock=now,
    )
    aily_http_adapter = create_aily_http_adapter(
        AilyConnectorConfig(api_key=config.aily_connector_api_key),
        gateway=aily_task_gateway,
        scenario_context_gateway=scenario_gateway,
        audit_ledger=result_resolver,
        report_exporter=artifact_exporter,
        result_authorizer=authorizer,
        recheck_action_gateway=recheck_action_gateway,
    )
    aily_mcp_adapter = None
    if config.aily_mcp is not None:
        assert project_model_dependencies is not None
        from quanxin_life.api.aily_mcp import (
            AilyMcpConfig,
            create_aily_mcp_adapter,
        )

        aily_mcp_adapter = create_aily_mcp_adapter(
            AilyMcpConfig(
                endpoint_token=config.aily_mcp.endpoint_token,
                allowed_source_ips=config.aily_mcp.allowed_source_ips,
                allowed_hosts=config.aily_mcp.allowed_hosts,
            ),
            gateway=aily_task_gateway,
            scenario_context_gateway=scenario_gateway,
            audit_ledger=result_resolver,
            report_exporter=artifact_exporter,
            result_authorizer=authorizer,
            caller_authorizer=ProjectBoundAilyMcpCallerAuthorizer(
                job_store=job_store,
                context_service=project_model_dependencies.context_service,
                identity_bindings=config.aily_mcp.identity_bindings,
            ),
            recheck_action_gateway=recheck_action_gateway,
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
        aily_mcp_adapter=aily_mcp_adapter,
        aily_task_gateway=aily_task_gateway,
        sibling_job_service=sibling_job_service,
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
    "AilyMcpAssemblyConfig",
    "FeishuAilyAssemblyConfig",
    "FeishuAilyComponents",
    "FeishuCsvRegistrationResolver",
    "FeishuProjectModelDependencies",
    "RegisteredFeishuCsvRegistrationResolver",
    "create_feishu_aily_components",
]
