from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine

import quanxin_life.application as application_package
from quanxin_life.api.aily import AilyCreateAnalysisTaskRequest
from quanxin_life.application import (
    AilyMcpAssemblyConfig as ExportedAilyMcpAssemblyConfig,
)
from quanxin_life.application import (
    FeishuAilyAssemblyConfig as ExportedFeishuAilyAssemblyConfig,
)
from quanxin_life.application import (
    create_feishu_aily_components as exported_create_feishu_aily_components,
)
from quanxin_life.application.aily_mcp_authorization import (
    load_aily_mcp_identity_bindings,
)
from quanxin_life.application.feishu_aily_assembly import (
    AilyMcpAssemblyConfig,
    FeishuAilyAssemblyConfig,
    FeishuProjectModelDependencies,
    RegisteredFeishuCsvRegistrationResolver,
    create_feishu_aily_components,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.core import (
    CellMetadata,
    DatasetStatus,
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.infrastructure.feishu_queue import FEISHU_ANALYSIS_TASK
from quanxin_life.integrations.feishu.analysis_plots import FeishuAnalysisPlotter
from quanxin_life.integrations.feishu.bitable import (
    CHINESE_ANALYSIS_BITABLE_PROFILE,
    BitableMediaUploader,
)
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioRegistry,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import (
    Dataset,
    FeishuBindingRow,
    FeishuEventReceipt,
    Project,
    RecordBatchBinding,
    User,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
CSV = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"UPLOAD,registered-cell,1,0,0.0,3.6,1.0,25.0,1.2,1.1,0.02,true,true\n"
    b"UPLOAD,registered-cell,20,0,0.0,3.5,1.0,25.0,1.1,1.0,0.03,true,true\n"
)


class _AsyncResult:
    id = "task-feishu-aily"


class _CeleryApp:
    def __init__(self) -> None:
        self.tasks: dict[str, object] = {}
        self.sent: list[dict[str, object]] = []

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult:
        self.sent.append({"name": name, "kwargs": kwargs, "queue": queue})
        return _AsyncResult()


class _NoNetworkTransport:
    def request(self, _request: object) -> object:
        raise AssertionError("assembly must not access Feishu while being constructed")


def _config(
    data_root: Path,
    *,
    recheck_table_id: str | None = None,
    recheck_permission_reference: str | None = None,
    aily_mcp: AilyMcpAssemblyConfig | None = None,
) -> FeishuAilyAssemblyConfig:
    return FeishuAilyAssemblyConfig(
        app_id="cli_test_app",
        app_secret=SecretStr("app-secret"),
        verification_token=SecretStr("verification-token"),
        encrypt_key=SecretStr("0123456789abcdef"),
        aily_connector_api_key=SecretStr("aily-key"),
        bitable_app_token="app-token",
        bitable_table_id="table-id",
        external_https_base_url="https://integration.example.test",
        data_root=data_root,
        allow_candidate_scenario_execution=True,
        allow_candidate_scenario_results=True,
        recheck_table_id=recheck_table_id,
        recheck_permission_reference=recheck_permission_reference,
        aily_mcp=aily_mcp,
    )


def _registration(
    *,
    metadata_sha256: str,
    observed_sha256: str,
) -> CanonicalCsvBatchRegistration:
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="registered-cell",
            chemistry="LFP/graphite",
            nominal_capacity_ah=250.0,
            source_uri="feishu://canonical/registered-cell.csv",
            source_sha256=metadata_sha256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="feishu-upload-v1",
        split_version="operator-registration-v1",
        provenance=(
            ProvenanceRecord(
                source_id="registered-feishu-upload",
                source_kind=SourceKind.OBSERVED,
                uri="feishu://canonical/registered-cell.csv",
                sha256=observed_sha256,
                description="Operator-approved canonical CSV registration.",
                created_at=NOW,
            ),
        ),
    )


def test_feishu_aily_assembly_shares_one_persistent_boundary(tmp_path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    celery_app = _CeleryApp()

    components = create_feishu_aily_components(
        session_factory=sessions,
        celery_app=celery_app,
        config=_config(tmp_path / "batches"),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        clock=lambda: NOW,
    )

    assert components.event_processor.decryptor_configured is True
    assert components.worker._store is components.job_store
    assert components.aily_task_gateway._job_store is components.job_store
    assert components.worker._result_resolver is components.audit_ledger
    assert components.worker._sibling_planner is not None
    assert components.sibling_job_service._store is components.job_store
    assert (
        components.worker._sibling_planner._sibling_jobs
        is components.sibling_job_service
    )
    analysis_plotter = components.worker._delivery._feishu_delivery._analysis_plotter
    scenario_plotter = components.worker._delivery._feishu_delivery._scenario_plotter
    assert isinstance(analysis_plotter, FeishuAnalysisPlotter)
    assert scenario_plotter is analysis_plotter
    feishu_delivery = components.worker._delivery._feishu_delivery
    aily_delivery = components.worker._delivery._aily_delivery
    media_uploader = feishu_delivery._bitable_media_uploader
    assert isinstance(media_uploader, BitableMediaUploader)
    assert feishu_delivery._bitable_curve_plotter is analysis_plotter
    assert aily_delivery._bitable_media_uploader is media_uploader
    assert aily_delivery._analysis_plotter is analysis_plotter
    assert (
        feishu_delivery._bitable_writer._field_profile
        is CHINESE_ANALYSIS_BITABLE_PROFILE
    )
    assert components.aily_http_adapter is not None
    assert components.aily_mcp_adapter is None
    assert "/v1/aily/recheck-actions" not in {
        getattr(route, "path", None)
        for route in components.aily_http_adapter.router.routes
    }
    assert components.feishu_http_adapter is not None
    assert FEISHU_ANALYSIS_TASK not in celery_app.tasks


def test_feishu_aily_assembly_enables_mcp_only_with_project_identity_binding(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    identity_file = tmp_path / "aily-mcp-identities.json"
    identity_file.write_text(
        '{"schema_version":"quanxin-aily-mcp-identity-bindings-v1",'
        '"bindings":[{"aily_user_id":"aily-user-1",'
        '"local_user_id":"local-user-1"}]}\n',
        encoding="utf-8",
    )
    mcp_config = AilyMcpAssemblyConfig(
        endpoint_token=SecretStr("mcp_endpoint_token_0123456789abcdef"),
        allowed_source_ips=("101.126.59.88",),
        allowed_hosts=("integration.example.test",),
        identity_bindings=load_aily_mcp_identity_bindings(identity_file.resolve()),
    )

    with pytest.raises(ValueError, match="MCP requires project"):
        create_feishu_aily_components(
            session_factory=sessions,
            celery_app=_CeleryApp(),
            config=_config(tmp_path / "rejected-batches", aily_mcp=mcp_config),
            feishu_transport=_NoNetworkTransport(),
            default_scenarios=ReviewedDefaultScenarioRegistry(),
            clock=lambda: NOW,
        )

    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    components = create_feishu_aily_components(
        session_factory=sessions,
        celery_app=_CeleryApp(),
        config=_config(tmp_path / "batches", aily_mcp=mcp_config),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        project_model_dependencies=FeishuProjectModelDependencies(
            context_service=context_service,
            project_ledger=SqlProjectAuditLedger(
                sessions,
                context_validator=context_service,
                clock=lambda: NOW,
            ),
            project_tool_service=SimpleNamespace(
                invoke_in_project=lambda *_args, **_kwargs: None
            ),
        ),
        clock=lambda: NOW,
    )

    assert components.aily_mcp_adapter is not None
    tools = asyncio.run(components.aily_mcp_adapter.server.list_tools())
    assert {tool.name for tool in tools} == {
        "quanxin_create_analysis_task",
        "quanxin_create_scenario_context",
        "quanxin_get_analysis_task",
        "quanxin_get_audited_report",
        "quanxin_get_audited_result",
    }


def test_feishu_aily_assembly_mounts_recheck_only_with_project_authorization(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    config = _config(
        tmp_path / "batches",
        recheck_table_id="tbl-reviewed-rechecks",
        recheck_permission_reference="permission-reviewed-v1",
    )

    with pytest.raises(ValueError, match="recheck actions require project"):
        create_feishu_aily_components(
            session_factory=sessions,
            celery_app=_CeleryApp(),
            config=config,
            feishu_transport=_NoNetworkTransport(),
            default_scenarios=ReviewedDefaultScenarioRegistry(),
            clock=lambda: NOW,
        )

    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    components = create_feishu_aily_components(
        session_factory=sessions,
        celery_app=_CeleryApp(),
        config=config,
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        project_model_dependencies=FeishuProjectModelDependencies(
            context_service=context_service,
            project_ledger=SqlProjectAuditLedger(
                sessions,
                context_validator=context_service,
                clock=lambda: NOW,
            ),
            project_tool_service=SimpleNamespace(
                invoke_in_project=lambda *_args, **_kwargs: None
            ),
        ),
        clock=lambda: NOW,
    )

    assert "/v1/aily/recheck-actions" in {
        getattr(route, "path", None)
        for route in components.aily_http_adapter.router.routes
    }


def test_feishu_aily_config_rejects_partial_recheck_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="recheck table and permission"):
        _config(
            tmp_path / "batches",
            recheck_table_id="tbl-reviewed-rechecks",
        )


def test_application_package_lazily_exports_the_feishu_aily_assembly() -> None:
    assert ExportedAilyMcpAssemblyConfig is AilyMcpAssemblyConfig
    assert ExportedFeishuAilyAssemblyConfig is FeishuAilyAssemblyConfig
    assert exported_create_feishu_aily_components is create_feishu_aily_components
    assert (
        getattr(application_package, "RegisteredFeishuCsvRegistrationResolver", None)
        is RegisteredFeishuCsvRegistrationResolver
    )


@pytest.mark.parametrize(
    ("mapping_sha256", "metadata_sha256", "observed_sha256", "message"),
    (
        ("a" * 64, "b" * 64, "a" * 64, "metadata SHA-256"),
        ("a" * 64, "a" * 64, "b" * 64, "provenance SHA-256"),
    ),
)
def test_registered_feishu_csv_resolver_rejects_unbound_direct_construction(
    mapping_sha256: str,
    metadata_sha256: str,
    observed_sha256: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RegisteredFeishuCsvRegistrationResolver(
            {
                mapping_sha256: _registration(
                    metadata_sha256=metadata_sha256,
                    observed_sha256=observed_sha256,
                )
            }
        )


def test_feishu_aily_assembly_exposes_only_the_minimal_global_tools(tmp_path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    components = create_feishu_aily_components(
        session_factory=create_session_factory(engine),
        celery_app=_CeleryApp(),
        config=_config(tmp_path / "batches"),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        clock=lambda: NOW,
    )

    tool_names = {
        schema.tool_name.value
        for schema in components.tool_service.registry.list_schemas()
    }
    assert tool_names == {
        "validate_battery_data",
        "compare_operation_scenarios",
        "project_storage_lifetime",
        "generate_audited_report",
    }


def test_feishu_aily_assembly_injects_existing_project_model_runtime(tmp_path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    project_ledger = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )
    project_tool_service = SimpleNamespace(invoke_in_project=lambda *_args, **_kwargs: None)

    components = create_feishu_aily_components(
        session_factory=sessions,
        celery_app=_CeleryApp(),
        config=_config(tmp_path / "batches"),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        project_model_dependencies=FeishuProjectModelDependencies(
            context_service=context_service,
            project_ledger=project_ledger,
            project_tool_service=project_tool_service,
        ),
        clock=lambda: NOW,
    )

    executor = components.worker._project_model_executor
    assert executor is not None
    assert executor._context_service is context_service
    assert executor._project_ledger is project_ledger
    assert executor._project_tool_service is project_tool_service
    assert components.worker._result_resolver is not components.audit_ledger


def test_project_runtime_authorizes_only_the_anchored_frozen_aily_batch(
    tmp_path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    project_ledger = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )
    celery_app = _CeleryApp()
    components = create_feishu_aily_components(
        session_factory=sessions,
        celery_app=celery_app,
        config=_config(tmp_path / "batches"),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        project_model_dependencies=FeishuProjectModelDependencies(
            context_service=context_service,
            project_ledger=project_ledger,
            project_tool_service=SimpleNamespace(
                invoke_in_project=lambda *_args, **_kwargs: None
            ),
        ),
        clock=lambda: NOW,
    )
    digest = sha256(CSV).hexdigest()
    raw_upload_digest = "b" * 64
    mapping_evidence = {
        "raw_sha256": raw_upload_digest,
        "canonical_sha256": digest,
        "profile_id": "reviewed-layout",
        "profile_version": "reviewed-layout-v1",
        "profile_sha256": "c" * 64,
        "column_evidence": [],
    }
    registration = _registration(
        metadata_sha256=digest,
        observed_sha256=digest,
    )
    content_batch_id = components.batch_store.register_canonical_csv(
        CSV,
        registration=registration,
    )
    content_batch = components.batch_store.resolve_verified_early_cycle_batch(
        content_batch_id
    )
    user_id = str(uuid4())
    project_id = str(uuid4())
    dataset_id = str(uuid4())
    source_run_id = str(uuid4())
    with sessions.begin() as session:
        session.add_all(
            (
                User(
                    id=user_id,
                    username="aily-project@example.test",
                    credential_hash="not-used-by-feishu",
                    must_change_credential=False,
                    role=UserRole.ADMIN.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Project(
                    id=project_id,
                    owner_user_id=user_id,
                    name="Aily project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Dataset(
                    id=dataset_id,
                    project_id=project_id,
                    name="Aily frozen dataset",
                    data_version=registration.data_version,
                    schema_version=registration.metadata.schema_version,
                    status=DatasetStatus.FROZEN.value,
                    manifest_uri=None,
                    manifest_sha256=None,
                    created_at=NOW,
                    frozen_at=NOW,
                ),
                RecordBatchBinding(
                    id=str(uuid4()),
                    binding_schema_version="record-batch-binding-v1",
                    content_batch_id=content_batch_id,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    source_manifest_sha256=content_batch.source_manifest_hash,
                    registration_sha256=sha256_canonical(
                        registration.model_dump(mode="json")
                    ),
                    content_dataset_id=registration.metadata.dataset_id,
                    dataset_schema_version=registration.metadata.schema_version,
                    cell_id=registration.metadata.cell_id,
                    cutoff_cycle=registration.feature_config.cutoff_cycle,
                    data_version=registration.data_version,
                    split_version=registration.split_version,
                    feature_version=registration.feature_config.feature_version,
                    created_by_user_id=user_id,
                    created_at=NOW,
                ),
                FeishuBindingRow(
                    id=str(uuid4()),
                    project_id=project_id,
                    chat_id="oc-aily-project",
                    bitable_app_token=None,
                    bitable_table_id=None,
                    user_open_id_map_json={user_id: "ou-aily-project"},
                    binding_version="feishu-binding-v1",
                    status="ACTIVE",
                    created_at=NOW,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="evt-aily-project-source",
                    event_type="im.message.receive_v1",
                    payload_sha256="a" * 64,
                    status="PROCESSED",
                    attempt_count=1,
                    received_at=NOW,
                    processed_at=NOW,
                    job_id=source_run_id,
                    job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                    job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                    job_stage="SUCCEEDED",
                    task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                    run_id=source_run_id,
                    chat_id="oc-aily-project",
                    sender_id="ou-aily-project",
                    receive_id_type="chat_id",
                    event_time=NOW,
                    record_batch_id=content_batch_id,
                    cell_reference=registration.metadata.cell_id,
                    input_file_sha256=raw_upload_digest,
                    csv_mapping_status="MAPPED",
                    csv_mapping_evidence_json=mapping_evidence,
                    csv_mapping_evidence_sha256=sha256_canonical(mapping_evidence),
                    validation_result_id=str(uuid4()),
                    job_attempt_count=1,
                    job_created_at=NOW,
                    job_updated_at=NOW,
                    job_completed_at=NOW,
                ),
            )
        )

    created = components.aily_task_gateway.create_analysis_task(
        AilyCreateAnalysisTaskRequest(
            task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            source_run_id=source_run_id,
            data_batch_id=content_batch_id,
        )
    )

    persisted = components.job_store.get(created.run_id)
    assert persisted.job_origin is FeishuAnalysisJobOrigin.AILY
    assert persisted.source_job_id == source_run_id
    assert persisted.record_batch_id == content_batch_id
    assert len(celery_app.sent) == 1

    with sessions.begin() as session:
        dataset = session.get(Dataset, dataset_id)
        assert dataset is not None
        dataset.status = DatasetStatus.DRAFT.value
        dataset.frozen_at = None

    with pytest.raises(ValueError, match="not authorized"):
        components.aily_task_gateway.create_analysis_task(
            AilyCreateAnalysisTaskRequest(
                task_type=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
                source_run_id=source_run_id,
                data_batch_id=content_batch_id,
            )
        )
    assert len(celery_app.sent) == 1


def test_feishu_file_registration_is_fail_closed_without_trusted_metadata(
    tmp_path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    components = create_feishu_aily_components(
        session_factory=create_session_factory(engine),
        celery_app=_CeleryApp(),
        config=_config(tmp_path / "batches"),
        feishu_transport=_NoNetworkTransport(),
        default_scenarios=ReviewedDefaultScenarioRegistry(),
        clock=lambda: NOW,
    )

    try:
        components.registration_resolver(
            SimpleNamespace(),
            SimpleNamespace(),
            NOW,
        )
    except ValueError as exc:
        assert str(exc) == "trusted Feishu CSV registration metadata is not configured"
    else:  # pragma: no cover - fail-closed invariant
        raise AssertionError("unregistered Feishu CSV metadata must be rejected")
