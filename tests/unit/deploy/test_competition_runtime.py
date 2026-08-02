from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic import SecretStr

from deploy.competition_runtime import create_competition_runtime
from deploy.runtime_settings import CompetitionRuntimeSettings
from quanxin_life.infrastructure.calibration_queue import (
    ADVANCED_CALIBRATION_TASK,
)
from quanxin_life.infrastructure.celery_queue import AGENT_RUN_TASK


def test_competition_runtime_assembles_http_and_both_identity_only_workers(
    tmp_path,
) -> None:
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
        path.mkdir()
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
    runtime = create_competition_runtime(
        CompetitionRuntimeSettings(
            database_url=SecretStr(
                f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}"
            ),
            redis_url=SecretStr("redis://runtime.test:6379/0"),
            trusted_origin="https://47.98.37.232",
            data_root=roots["data"],
            artifact_root=roots["artifacts"],
            policy_root=roots["policies"],
            deployment_registry_root=roots["deployment-registry"],
            deployment_registry_id="2" * 64,
            calibration_registrations_file=registrations,
            calibration_evidence_root=roots["calibration-evidence"],
            agent_policy_file=policy_file,
        )
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
