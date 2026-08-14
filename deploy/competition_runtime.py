"""Explicit production assembly for the single-node competition runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from celery.signals import worker_ready

from deploy.advanced_agent_context import CompetitionAgentPolicyContextResolver
from deploy.competition_inputs import (
    load_advanced_agent_policy,
    load_calibration_source_registrations,
    load_feishu_csv_registrations,
    load_feishu_default_scenario_profiles,
)
from deploy.runtime_settings import CompetitionRuntimeSettings
from quanxin_life.agents.supervisor import SupervisorPlanner
from quanxin_life.api.agent_runs import create_agent_run_http_adapter
from quanxin_life.api.analysis_catalog import create_analysis_catalog_http_adapter
from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.datasets import create_dataset_http_adapter
from quanxin_life.api.model_artifacts import create_model_artifact_http_adapter
from quanxin_life.api.model_routes import create_model_route_http_adapter
from quanxin_life.api.project_reports import create_project_report_http_adapter
from quanxin_life.api.projects import create_project_http_adapter
from quanxin_life.api.record_batches import create_record_batch_http_adapter
from quanxin_life.application.advanced_calibration_evidence import (
    RegisteredAdvancedCalibrationEvidenceResolver,
)
from quanxin_life.application.advanced_calibration_inputs import (
    RegisteredMatrAdvancedCalibrationCellInputResolver,
)
from quanxin_life.application.advanced_deployment_registry import (
    AdvancedDeepModelArtifactCatalogSource,
    AdvancedDeploymentBundleRegistry,
)
from quanxin_life.application.advanced_prediction import (
    AdvancedRULPredictionService,
    AdvancedSOHPredictionService,
)
from quanxin_life.application.advanced_runtime import (
    ActiveAdvancedRuntimeResolver,
    ManagedAdvancedRuntimeProvider,
)
from quanxin_life.application.agent_run_execution import AgentRunExecutionWorker
from quanxin_life.application.agent_run_invocation import (
    PersistentAgentRunInvocationResolver,
)
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.assembly import (
    AdvancedCalibrationAssemblyDependencies,
    ProjectPredictionToolDependencies,
    create_advanced_calibration_components,
    create_project_prediction_tool_invocation_service,
)
from quanxin_life.application.battery_csv_mapping import (
    load_battery_csv_mapping_profile,
)
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.engineering_recommendation_rules import (
    load_engineering_recommendation_ruleset_registry,
)
from quanxin_life.application.feishu_aily_assembly import (
    FeishuAilyAssemblyConfig,
    FeishuProjectModelDependencies,
    RegisteredFeishuCsvRegistrationResolver,
    create_feishu_aily_components,
)
from quanxin_life.application.ingestion import (
    FileSystemVerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
)
from quanxin_life.application.model_artifact_catalog import (
    ModelArtifactCatalogService,
)
from quanxin_life.application.model_route_activation import (
    ModelRouteActivationService,
)
from quanxin_life.application.project_reports import (
    ProjectReportExportService,
    SqlProjectReportResultResolver,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.record_batch_bindings import (
    RecordBatchBindingService,
)
from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.infrastructure.calibration_queue import (
    CeleryAdvancedCalibrationQueue,
)
from quanxin_life.infrastructure.celery_queue import (
    CeleryAgentRunQueue,
    CeleryQueueConfig,
    create_celery_app,
)
from quanxin_life.infrastructure.report_queue import CeleryProjectReportQueue
from quanxin_life.persistence import (
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.reporting.audited_artifacts import reviewed_reportlab_vera_font
from quanxin_life.reporting.project_artifacts import ProjectReportArtifactRenderer
from quanxin_life.tasks.advanced_calibration import (
    register_advanced_calibration_task,
)
from quanxin_life.tasks.agent_runs import register_agent_run_task
from quanxin_life.tasks.feishu import register_feishu_analysis_task
from quanxin_life.tasks.project_reports import register_project_report_task
from quanxin_life.tools import ToolExecutionScope

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompetitionRuntime:
    """Assembled HTTP and Worker surfaces sharing one persistent boundary."""

    http_app: Any
    celery_app: Any
    session_factory: SessionFactory
    agent_worker: AgentRunExecutionWorker
    calibration_worker: Any
    report_worker: ProjectReportExportService
    feishu_worker: Any | None
    feishu_sibling_job_service: Any | None


def recover_feishu_sibling_dispatches(runtime: CompetitionRuntime) -> tuple[str, ...]:
    """Re-enqueue persisted sibling jobs once when a worker process starts."""

    service = runtime.feishu_sibling_job_service
    if service is None:
        return ()
    return tuple(service.dispatch_pending())


def register_feishu_sibling_recovery(
    runtime: CompetitionRuntime,
) -> Callable[..., tuple[str, ...]]:
    """Recover persisted sibling dispatches after the Celery consumer is ready."""

    def recover_on_worker_ready(**_: object) -> tuple[str, ...]:
        try:
            return recover_feishu_sibling_dispatches(runtime)
        except Exception as exc:
            _LOGGER.error(
                "Feishu sibling startup recovery failed: %s",
                type(exc).__name__,
            )
            return ()

    worker_ready.connect(recover_on_worker_ready, weak=False)
    return recover_on_worker_ready


def create_competition_runtime(
    settings: CompetitionRuntimeSettings,
    *,
    auth_environment: Literal["production", "development"] = "production",
) -> CompetitionRuntime:
    """Assemble the Advanced competition vertical without starting services."""

    engine = create_engine_from_config(
        DatabaseConfig(url=settings.database_url.get_secret_value())
    )
    session_factory = create_session_factory(engine)
    context_service = ProjectInvocationContextService(session_factory)
    ledger = SqlProjectAuditLedger(
        session_factory,
        context_validator=context_service,
    )

    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=Argon2idPasswordHasher(),
        password_policy=PasswordPolicy(),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(
            environment=auth_environment,
            allowed_origins=(settings.trusted_origin,),
        ),
    )

    batch_store = FileSystemVerifiedEarlyCycleBatchStore(settings.data_root)
    record_batch_service = RecordBatchBindingService(
        session_factory,
        batch_store,
        context_validator=context_service,
    )
    deployment_registry = AdvancedDeploymentBundleRegistry(
        settings.deployment_registry_root
    )
    artifact_source = AdvancedDeepModelArtifactCatalogSource(
        deployment_registry,
        settings.deployment_registry_id,
    )
    artifact_service = ModelArtifactCatalogService(
        session_factory,
        source=artifact_source,
    )
    route_service = ModelRouteActivationService(
        session_factory,
        source=artifact_source,
    )
    runtime_resolver = ActiveAdvancedRuntimeResolver(
        context_service=context_service,
        route_service=route_service,
        runtime_provider=ManagedAdvancedRuntimeProvider(deployment_registry),
    )
    invocation_resolver = PersistentAgentRunInvocationResolver(
        session_factory,
        context_service=context_service,
    )
    recommendation_rulesets = None
    if (
        settings.feishu_aily is not None
        and settings.feishu_aily.engineering_recommendation_rulesets_file is not None
        and settings.feishu_aily.engineering_recommendation_rulesets_sha256 is not None
    ):
        recommendation_rulesets = load_engineering_recommendation_ruleset_registry(
            settings.feishu_aily.engineering_recommendation_rulesets_file,
            expected_file_sha256=(
                settings.feishu_aily.engineering_recommendation_rulesets_sha256
            ),
        )
    tool_service = create_project_prediction_tool_invocation_service(
        ProjectPredictionToolDependencies(
            project_audit_ledger=ledger,
            project_context_validator=context_service,
            agent_run_invocation_resolver=invocation_resolver,
            advanced_input_batch_resolver=record_batch_service,
            advanced_rul_inference_service=AdvancedRULPredictionService(
                batch_resolver=record_batch_service,
                runtime_resolver=runtime_resolver,
            ),
            advanced_soh_inference_service=AdvancedSOHPredictionService(
                batch_resolver=record_batch_service,
                runtime_resolver=runtime_resolver,
            ),
            engineering_recommendation_ruleset_resolver=recommendation_rulesets,
        )
    )

    celery_app = create_celery_app(
        CeleryQueueConfig(broker_url=settings.redis_url)
    )
    agent_queue = CeleryAgentRunQueue(app=celery_app)
    calibration_queue = CeleryAdvancedCalibrationQueue(app=celery_app)
    policy = load_advanced_agent_policy(settings.agent_policy_file)
    policy_context = CompetitionAgentPolicyContextResolver(
        session_factory,
        context_service=context_service,
        record_batch_resolver=record_batch_service,
        policy=policy.policy,
    )
    evidence_resolver = RegisteredAdvancedCalibrationEvidenceResolver(
        load_calibration_source_registrations(
            settings.calibration_registrations_file,
            evidence_root=settings.calibration_evidence_root,
        )
    )
    calibration = create_advanced_calibration_components(
        AdvancedCalibrationAssemblyDependencies(
            session_factory=session_factory,
            context_service=context_service,
            evidence_resolver=evidence_resolver,
            runtime_resolver=runtime_resolver,
            cell_input_resolver=(
                RegisteredMatrAdvancedCalibrationCellInputResolver(
                    source_resolver=evidence_resolver,
                )
            ),
            project_materializer=ledger,
            queue=calibration_queue,
            auth_adapter=auth_adapter,
            agent_context_delegate=policy_context,
            target_record_batch_resolver=record_batch_service,
        )
    )

    available_tools = tuple(
        schema.tool_name
        for schema in tool_service.registry.list_schemas(
            execution_scope=ToolExecutionScope.PROJECT
        )
    )
    run_service = AgentRunService(
        session_factory,
        planner=SupervisorPlanner(gateway=None),
    )
    dataset_service = DatasetService(session_factory)
    agent_worker = AgentRunExecutionWorker(
        session_factory,
        run_service=run_service,
        tool_service=tool_service,
        context_resolver=calibration.agent_context_resolver,
        agent_run_invocation_resolver=invocation_resolver,
        clock=lambda: datetime.now(UTC),
    )
    celery_task_app = cast(Any, celery_app)
    register_agent_run_task(app=celery_task_app, worker=agent_worker)
    register_advanced_calibration_task(
        app=celery_task_app,
        worker=calibration.worker,
    )
    report_worker = ProjectReportExportService(
        session_factory,
        run_reader=run_service,
        result_resolver=SqlProjectReportResultResolver(session_factory),
        renderer=ProjectReportArtifactRenderer(
            pdf_font=reviewed_reportlab_vera_font()
        ),
        artifact_root=settings.artifact_root,
    )
    report_queue = CeleryProjectReportQueue(app=celery_app)
    register_project_report_task(app=celery_task_app, worker=report_worker)
    feishu_aily = None
    if settings.feishu_aily is not None:
        integration = settings.feishu_aily
        feishu_aily = create_feishu_aily_components(
            session_factory=session_factory,
            celery_app=celery_task_app,
            config=FeishuAilyAssemblyConfig(
                app_id=integration.app_id,
                app_secret=integration.app_secret,
                verification_token=integration.verification_token,
                encrypt_key=integration.encrypt_key,
                aily_connector_api_key=integration.aily_connector_api_key,
                bitable_app_token=integration.bitable_app_token,
                bitable_table_id=integration.bitable_table_id,
                external_https_base_url=integration.external_https_base_url,
                data_root=settings.data_root,
                allow_candidate_scenario_execution=(
                    integration.allow_candidate_scenario_execution
                ),
                allow_candidate_scenario_results=(
                    integration.allow_candidate_scenario_results
                ),
            ),
            registration_resolver=RegisteredFeishuCsvRegistrationResolver(
                load_feishu_csv_registrations(integration.csv_registrations_file)
            ),
            default_scenarios=load_feishu_default_scenario_profiles(
                integration.default_scenario_profiles_file
            ),
            project_model_dependencies=FeishuProjectModelDependencies(
                context_service=context_service,
                project_ledger=ledger,
                project_tool_service=tool_service,
            ),
            csv_mapping_profiles=(
                load_battery_csv_mapping_profile(
                    "configs/data_layouts/feishu_battery_csv_v1.json"
                ),
            ),
        )
        register_feishu_analysis_task(
            app=celery_task_app,
            worker=feishu_aily.worker,
        )

    http_app = create_fastapi_app(
        tool_service,
        auth_adapter=auth_adapter,
        project_adapter=create_project_http_adapter(
            ProjectService(session_factory),
            auth_adapter=auth_adapter,
        ),
        dataset_adapter=create_dataset_http_adapter(
            dataset_service,
            auth_adapter=auth_adapter,
        ),
        record_batch_adapter=create_record_batch_http_adapter(
            record_batch_service,
            auth_adapter=auth_adapter,
        ),
        agent_run_adapter=create_agent_run_http_adapter(
            run_service,
            auth_adapter=auth_adapter,
            available_tools=available_tools,
            queue=agent_queue,
        ),
        analysis_catalog_adapter=create_analysis_catalog_http_adapter(
            auth_adapter=auth_adapter,
            dataset_service=dataset_service,
            record_batch_service=record_batch_service,
            context_service=context_service,
            run_service=run_service,
            available_tools=available_tools,
            queue=agent_queue,
        ),
        model_artifact_adapter=create_model_artifact_http_adapter(
            artifact_service,
            auth_adapter=auth_adapter,
        ),
        model_route_adapter=create_model_route_http_adapter(
            route_service,
            auth_adapter=auth_adapter,
        ),
        advanced_calibration_adapter=calibration.http_adapter,
        project_report_adapter=create_project_report_http_adapter(
            report_worker,
            auth_adapter=auth_adapter,
            queue=report_queue,
        ),
        feishu_adapter=(
            feishu_aily.feishu_http_adapter if feishu_aily is not None else None
        ),
        aily_adapter=(
            feishu_aily.aily_http_adapter if feishu_aily is not None else None
        ),
        project_invocation_context_service=context_service,
    )
    return CompetitionRuntime(
        http_app=http_app,
        celery_app=celery_app,
        session_factory=session_factory,
        agent_worker=agent_worker,
        calibration_worker=calibration.worker,
        report_worker=report_worker,
        feishu_worker=feishu_aily.worker if feishu_aily is not None else None,
        feishu_sibling_job_service=(
            feishu_aily.sibling_job_service if feishu_aily is not None else None
        ),
    )


__all__ = [
    "CompetitionRuntime",
    "create_competition_runtime",
    "recover_feishu_sibling_dispatches",
    "register_feishu_sibling_recovery",
]
