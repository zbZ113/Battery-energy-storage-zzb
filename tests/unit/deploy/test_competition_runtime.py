from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import deploy.competition_runtime as competition_runtime_module
from deploy.competition_runtime import create_competition_runtime
from deploy.runtime_settings import (
    CompetitionRuntimeSettings,
    FeishuAilyRuntimeSettings,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.infrastructure.calibration_queue import (
    ADVANCED_CALIBRATION_TASK,
)
from quanxin_life.infrastructure.celery_queue import AGENT_RUN_TASK
from quanxin_life.infrastructure.feishu_queue import FEISHU_ANALYSIS_TASK
from quanxin_life.infrastructure.report_queue import PROJECT_REPORT_TASK


def _settings(tmp_path, *, trusted_origin: str) -> CompetitionRuntimeSettings:
    roots = {
        name: tmp_path / name
        for name in (
            "data",
            "artifacts",
            "policies",
            "deployment-registry",
            "calibration-evidence",
            "config",
        )
    }
    for path in roots.values():
        path.mkdir(parents=True)
    source_root = roots["calibration-evidence"] / "matr-three-batch"
    source_root.mkdir()
    registrations = roots["config"] / "calibration-sources.json"
    registrations.write_text(
        json.dumps(
            {
                "schema_version": "advanced-calibration-source-registry-v1",
                "sources": [
                    {
                        "registration_id": "matr-three-batch-final-v1",
                        "evidence_relative_root": "matr-three-batch",
                        "three_batch_manifest_sha256": "1" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_file = roots["policies"] / "advanced-agent.json"
    policy_file.write_text(
        json.dumps(
            {
                "schema_version": "advanced-agent-policy-v1",
                "conformal_alpha": 0.1,
            }
        ),
        encoding="utf-8",
    )
    return CompetitionRuntimeSettings(
        database_url=SecretStr(
            f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}"
        ),
        redis_url=SecretStr("redis://runtime.test:6379/0"),
        trusted_origin=trusted_origin,
        data_root=roots["data"],
        artifact_root=roots["artifacts"],
        policy_root=roots["policies"],
        deployment_registry_root=roots["deployment-registry"],
        deployment_registry_id="2" * 64,
        calibration_registrations_file=registrations,
        calibration_evidence_root=roots["calibration-evidence"],
        agent_policy_file=policy_file,
    )


def _enabled_settings(tmp_path) -> CompetitionRuntimeSettings:
    settings = _settings(
        tmp_path,
        trusted_origin="https://runtime.example.test",
    )
    payload = b"operator-registered-canonical-csv"
    payload_sha256 = sha256(payload).hexdigest()
    registration = CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="registered-cell",
            chemistry="LFP/graphite",
            nominal_capacity_ah=250.0,
            source_uri="feishu://canonical/registered-cell.csv",
            source_sha256=payload_sha256,
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
                sha256=payload_sha256,
                description="Operator-approved canonical CSV registration.",
                created_at=datetime(2026, 8, 12, tzinfo=UTC),
            ),
        ),
    )
    csv_registrations = tmp_path / "config" / "feishu-csv-registrations.json"
    csv_registrations.write_text(
        json.dumps(
            {
                "schema_version": "feishu-canonical-csv-registration-registry-v1",
                "registrations": [
                    {
                        "payload_sha256": payload_sha256,
                        "registration": registration.model_dump(mode="json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return replace(
        settings,
        feishu_aily=FeishuAilyRuntimeSettings(
            app_id="cli_test_app",
            app_secret=SecretStr("app-secret"),
            verification_token=SecretStr("verification-token"),
            encrypt_key=SecretStr("0123456789abcdef"),
            aily_connector_api_key=SecretStr("aily-key"),
            bitable_app_token="app-token",
            bitable_table_id="table-id",
            external_https_base_url="https://integration.example.test",
            csv_registrations_file=csv_registrations,
            allow_candidate_scenario_execution=True,
            allow_candidate_scenario_results=True,
        ),
    )


def test_competition_runtime_assembles_http_and_both_identity_only_workers(
    tmp_path,
) -> None:
    runtime = create_competition_runtime(
        _settings(tmp_path, trusted_origin="https://runtime.example.test")
    )

    response = TestClient(runtime.http_app).get("/health")

    assert response.status_code == 200
    route_paths = set(runtime.http_app.openapi()["paths"])
    assert {
        "/v1/auth/login",
        "/v1/projects",
        "/v1/agent/runs",
        "/v1/projects/{project_id}/analysis-inputs",
        "/v1/projects/{project_id}/datasets",
        "/v1/datasets/{dataset_id}/batches",
        "/v1/projects/{project_id}/agent/runs",
        "/v1/agent/runs/{run_id}/results",
        "/v1/projects/{project_id}/advanced-analyses",
        "/v1/admin/model-artifacts/advanced-candidates",
        "/v1/admin/model-routes/activations",
        "/v1/projects/{project_id}/advanced-calibration/materializations",
    }.issubset(route_paths)
    assert AGENT_RUN_TASK in runtime.celery_app.tasks
    assert ADVANCED_CALIBRATION_TASK in runtime.celery_app.tasks
    assert PROJECT_REPORT_TASK in runtime.celery_app.tasks
    assert FEISHU_ANALYSIS_TASK not in runtime.celery_app.tasks
    assert "/v1/integrations/feishu/events" not in route_paths
    assert "/v1/aily/scenario-contexts" not in route_paths
    assert runtime.feishu_worker is None
    assert (
        "/v1/projects/{project_id}/agent/runs/{run_id}/reports/"
        "{report_result_id}/exports"
    ) in route_paths
    assert runtime.report_worker is not None


def test_competition_runtime_opt_in_mounts_feishu_aily_and_shared_worker(
    tmp_path,
) -> None:
    runtime = create_competition_runtime(_enabled_settings(tmp_path))

    route_paths = set(runtime.http_app.openapi()["paths"])

    assert "/v1/integrations/feishu/events" in route_paths
    assert "/v1/aily/scenario-contexts" in route_paths
    assert "/v1/aily/analysis-tasks" in route_paths
    assert FEISHU_ANALYSIS_TASK in runtime.celery_app.tasks
    assert runtime.feishu_worker is not None
    project_executor = runtime.feishu_worker._project_model_executor
    assert project_executor is not None
    assert project_executor._project_tool_service is runtime.agent_worker._tool_service
    assert (
        project_executor._project_ledger
        is runtime.agent_worker._tool_service.project_audit_ledger
    )

    resolver = runtime.feishu_worker._registration_resolver
    registered = resolver(
        object(),
        type(
            "Attachment",
            (),
            {"sha256": sha256(b"operator-registered-canonical-csv").hexdigest()},
        )(),
        datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert registered.metadata.cell_id == "registered-cell"
    with pytest.raises(ValueError, match="not registered"):
        resolver(
            object(),
            type("Attachment", (), {"sha256": "f" * 64})(),
            datetime(2026, 8, 12, tzinfo=UTC),
        )


def test_competition_runtime_defaults_to_production_cookie_and_requires_local_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    captured = []
    original = competition_runtime_module.create_auth_http_adapter

    def capture(service, config):
        captured.append(config)
        return original(service, config)

    monkeypatch.setattr(
        competition_runtime_module,
        "create_auth_http_adapter",
        capture,
    )

    create_competition_runtime(
        _settings(
            tmp_path / "production",
            trusted_origin="https://runtime.example.test",
        )
    )
    create_competition_runtime(
        _settings(
            tmp_path / "local",
            trusted_origin="http://127.0.0.1:8080",
        ),
        auth_environment="development",
    )

    assert [config.environment for config in captured] == [
        "production",
        "development",
    ]
    assert [config.cookie_name for config in captured] == [
        "__Host-quanxin_session",
        "quanxin_dev_session",
    ]
