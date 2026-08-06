from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic import SecretStr

import deploy.competition_runtime as competition_runtime_module
from deploy.competition_runtime import create_competition_runtime
from deploy.runtime_settings import CompetitionRuntimeSettings
from quanxin_life.infrastructure.calibration_queue import (
    ADVANCED_CALIBRATION_TASK,
)
from quanxin_life.infrastructure.celery_queue import AGENT_RUN_TASK
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
    assert (
        "/v1/projects/{project_id}/agent/runs/{run_id}/reports/"
        "{report_result_id}/exports"
    ) in route_paths
    assert runtime.report_worker is not None


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
