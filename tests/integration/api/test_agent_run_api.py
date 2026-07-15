from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from quanxin_life.agents.supervisor import (
    SupervisorPlanningRequest,
    SupervisorPlanningResult,
)
from quanxin_life.api.agent_runs import create_agent_run_http_adapter
from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.datasets import create_dataset_http_adapter
from quanxin_life.api.projects import create_project_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.task_queue import AgentRunDispatchReceipt
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import (
    AgentFailurePolicy,
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
    UserRole,
    UserStatus,
)
from quanxin_life.core.product import AgentIntent
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import User
from quanxin_life.tools import StandardToolName

ORIGIN = "https://app.example.test"
USERNAME = "agent-user@example.test"
PASSWORD = "agent API passphrase 2026"
NOW = datetime(2026, 7, 16, 11, 0, tzinfo=UTC)


class FixedPlanner:
    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: object,
    ) -> SupervisorPlanningResult:
        del available_tools
        intent = AgentIntent(
            intent_id=str(uuid4()),
            project_id=request.project_id,
            goal=request.user_goal,
            dataset_ids=request.dataset_ids,
            requested_outputs=request.requested_outputs,
            created_at=NOW,
        )
        return SupervisorPlanningResult(
            intent=intent,
            plan=AgentPlan.build(
                plan_version="api-test-v1",
                intent_id=intent.intent_id,
                steps=(
                    AgentPlanStep(
                        step_id="features",
                        role=AgentRole.DATA_QUALITY,
                        tool_name=(
                            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value
                        ),
                        input_references={
                            "record_batch_id": "intent.dataset_ids[0]"
                        },
                        failure_policy=AgentFailurePolicy.STOP,
                    ),
                ),
                planning_mode=AgentPlanningMode.FIXED_FALLBACK,
                created_at=NOW,
            ),
        )


class RecordingQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def enqueue(self, *, run_id: str, plan_hash: str) -> AgentRunDispatchReceipt:
        self.calls.append((run_id, plan_hash))
        return AgentRunDispatchReceipt(run_id=run_id, task_id=f"task-{run_id}")


@pytest.fixture
def agent_client(tmp_path: Path) -> tuple[TestClient, RecordingQueue]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'agent-api.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    with session_factory.begin() as session:
        session.add(
            User(
                id=str(uuid4()),
                username=USERNAME,
                credential_hash=hasher.hash_password(PASSWORD),
                must_change_credential=False,
                role=UserRole.MEMBER.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,)),
    )
    project_adapter = create_project_http_adapter(
        ProjectService(session_factory), auth_adapter=auth_adapter
    )
    dataset_adapter = create_dataset_http_adapter(
        DatasetService(session_factory), auth_adapter=auth_adapter
    )
    queue = RecordingQueue()
    agent_adapter = create_agent_run_http_adapter(
        AgentRunService(session_factory, planner=FixedPlanner()),
        auth_adapter=auth_adapter,
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        queue=queue,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        project_adapter=project_adapter,
        dataset_adapter=dataset_adapter,
        agent_run_adapter=agent_adapter,
    )
    client = TestClient(app, base_url="https://api.example.test")
    login = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": PASSWORD},
    )
    assert login.status_code == 200
    return client, queue


def _create_scope(client: TestClient) -> tuple[str, str]:
    project = client.post(
        "/v1/projects",
        headers={"Origin": ORIGIN},
        json={"name": "Agent API project"},
    )
    assert project.status_code == 201
    dataset = client.post(
        "/v1/datasets",
        headers={"Origin": ORIGIN},
        json={
            "project_id": project.json()["project_id"],
            "name": "safe dataset",
            "data_version": "safe-v1",
            "schema_version": "canonical-v1",
        },
    )
    assert dataset.status_code == 201
    frozen = client.post(
        f"/v1/datasets/{dataset.json()['dataset_id']}/freeze",
        headers={"Origin": ORIGIN},
    )
    assert frozen.status_code == 200
    return project.json()["project_id"], frozen.json()["dataset_id"]


def test_agent_run_create_requires_idempotency_and_dispatches_only_once(
    agent_client: tuple[TestClient, RecordingQueue],
) -> None:
    client, queue = agent_client
    project_id, dataset_id = _create_scope(client)
    payload = {
        "project_id": project_id,
        "user_goal": "analyze this cell lifetime",
        "dataset_ids": [dataset_id],
        "requested_outputs": ["cycle_life"],
    }
    missing_key = client.post(
        "/v1/agent/runs", headers={"Origin": ORIGIN}, json=payload
    )
    assert missing_key.status_code == 422

    headers = {
        "Origin": ORIGIN,
        "Idempotency-Key": "browser-run-create-0001",
    }
    created = client.post("/v1/agent/runs", headers=headers, json=payload)
    repeated = client.post("/v1/agent/runs", headers=headers, json=payload)
    assert created.status_code == 202, created.text
    assert repeated.status_code == 202
    assert repeated.json() == created.json()
    assert created.json()["dispatch_status"] == "DISPATCHED"
    assert len(queue.calls) == 1
    assert client.get(f"/v1/agent/runs/{created.json()['run_id']}").json() == created.json()


def test_agent_run_write_requires_origin_and_idempotency_conflicts_return_409(
    agent_client: tuple[TestClient, RecordingQueue],
) -> None:
    client, _ = agent_client
    project_id, dataset_id = _create_scope(client)
    headers = {"Origin": ORIGIN, "Idempotency-Key": "conflict-key-0001"}
    payload = {
        "project_id": project_id,
        "user_goal": "first request",
        "dataset_ids": [dataset_id],
        "requested_outputs": ["cycle_life"],
    }
    first = client.post("/v1/agent/runs", headers=headers, json=payload)
    assert first.status_code == 202, first.text
    payload["user_goal"] = "second request"
    conflict = client.post("/v1/agent/runs", headers=headers, json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency_conflict"
    no_origin = client.post(
        "/v1/agent/runs",
        headers={"Idempotency-Key": "missing-origin-0001"},
        json=payload,
    )
    assert no_origin.status_code == 403


def test_cancelled_run_exposes_a_replayable_terminal_sse_timeline(
    agent_client: tuple[TestClient, RecordingQueue],
) -> None:
    client, _ = agent_client
    project_id, dataset_id = _create_scope(client)
    created = client.post(
        "/v1/agent/runs",
        headers={"Origin": ORIGIN, "Idempotency-Key": "sse-run-0001"},
        json={
            "project_id": project_id,
            "user_goal": "build a safe timeline",
            "dataset_ids": [dataset_id],
            "requested_outputs": ["cycle_life"],
        },
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    cancelled = client.post(
        f"/v1/agent/runs/{run_id}/cancel", headers={"Origin": ORIGIN}
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"

    with client.stream(
        "GET",
        f"/v1/agent/runs/{run_id}/events",
        headers={"Last-Event-ID": "1"},
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 2" in body
    assert "id: 3" in body
    assert "RUN_DISPATCHED" in body
    assert "RUN_CANCELLED" in body
    assert "build a safe timeline" not in body
