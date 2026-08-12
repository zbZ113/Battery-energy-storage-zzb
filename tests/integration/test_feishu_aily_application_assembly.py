from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine

import quanxin_life.application as application_package
from quanxin_life.application import (
    FeishuAilyAssemblyConfig as ExportedFeishuAilyAssemblyConfig,
)
from quanxin_life.application import (
    create_feishu_aily_components as exported_create_feishu_aily_components,
)
from quanxin_life.application.feishu_aily_assembly import (
    FeishuAilyAssemblyConfig,
    RegisteredFeishuCsvRegistrationResolver,
    create_feishu_aily_components,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.infrastructure.feishu_queue import FEISHU_ANALYSIS_TASK
from quanxin_life.persistence import Base, create_session_factory

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


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


def _config(data_root: Path) -> FeishuAilyAssemblyConfig:
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
        clock=lambda: NOW,
    )

    assert components.event_processor.decryptor_configured is True
    assert components.worker._store is components.job_store
    assert components.aily_task_gateway._job_store is components.job_store
    assert components.worker._result_resolver is components.audit_ledger
    assert components.aily_http_adapter is not None
    assert components.feishu_http_adapter is not None
    assert FEISHU_ANALYSIS_TASK not in celery_app.tasks


def test_application_package_lazily_exports_the_feishu_aily_assembly() -> None:
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
