from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from quanxin_life.agents.supervisor import SupervisorPlanningRequest
from quanxin_life.api.agent_runs import create_agent_run_http_adapter
from quanxin_life.api.analysis_catalog import create_analysis_catalog_http_adapter
from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.datasets import create_dataset_http_adapter
from quanxin_life.api.projects import create_project_http_adapter
from quanxin_life.api.record_batches import create_record_batch_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.ingestion import (
    CANONICAL_CYCLE_CSV_FIELDS,
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.record_batch_bindings import RecordBatchBindingService
from quanxin_life.application.task_queue import AgentRunDispatchReceipt
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, UserRole, UserStatus
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import User, UserProjectRole
from quanxin_life.tools import StandardToolName

NOW = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
USERNAME = "analysis-catalog@example.test"
OUTSIDER_USERNAME = "analysis-outsider@example.test"
JUDGE_USERNAME = "analysis-judge@example.test"
PASSWORD = "analysis catalog passphrase 2026"
ADVANCED_TOOLS = (
    StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
    StandardToolName.PREDICT_CYCLE_LIFE,
    StandardToolName.PREDICT_SOH_TRAJECTORY,
    StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
    StandardToolName.GENERATE_AUDITED_REPORT,
)
FIXED_STEP_IDS = (
    "advanced-input",
    "rul-point",
    "rul-coverage",
    "soh",
    "rul-calibration",
    "rul-interval",
    "soh-calibration",
    "soh-band",
    "report",
)


class RecordingQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def enqueue(self, *, run_id: str, plan_hash: str) -> AgentRunDispatchReceipt:
        self.calls.append((run_id, plan_hash))
        return AgentRunDispatchReceipt(run_id=run_id, task_id=f"task-{run_id}")


class ExplodingPlanner:
    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: object,
    ) -> object:
        del request, available_tools
        raise AssertionError("advanced analyses must not use the generic planner")


@dataclass(frozen=True, slots=True)
class _Context:
    client: TestClient
    outsider_client: TestClient
    judge_client: TestClient
    queue: RecordingQueue
    project_id: str
    dataset_id: str
    record_batch_ids: tuple[str, str]


def _csv_payload(*, cell_id: str, cutoff_cycle: int) -> bytes:
    header = ",".join(CANONICAL_CYCLE_CSV_FIELDS)
    rows = (
        f"UPLOAD,{cell_id},1,0,0,3.1,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},1,1,1,3.2,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},{cutoff_cycle},0,0,3.1,-1,25,1.0,0.9,0.02,true,true",
        f"UPLOAD,{cell_id},{cutoff_cycle},1,1,3.2,-1,25,1.0,0.9,0.02,true,true",
    )
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _upload_body(*, cell_id: str, cutoff_cycle: int) -> dict[str, object]:
    payload = _csv_payload(cell_id=cell_id, cutoff_cycle=cutoff_cycle)
    digest = hashlib.sha256(payload).hexdigest()
    registration = CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id=cell_id,
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="upload://canonical/cell-catalog.csv",
            source_sha256=digest,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=cutoff_cycle),
        data_version="catalog-data-v1",
        split_version="catalog-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id=f"catalog-source-{cell_id}",
                source_kind=SourceKind.OBSERVED,
                uri="upload://canonical/cell-catalog.csv",
                sha256=digest,
                description="Observed canonical catalog fixture",
                created_at=NOW,
            ),
        ),
    )
    return {
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "registration": registration.model_dump(mode="json"),
    }


@pytest.fixture
def catalog_context(tmp_path: Path) -> _Context:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    user_ids = {
        USERNAME: str(uuid4()),
        OUTSIDER_USERNAME: str(uuid4()),
        JUDGE_USERNAME: str(uuid4()),
    }
    with session_factory.begin() as session:
        for username, role in (
            (USERNAME, UserRole.MEMBER),
            (OUTSIDER_USERNAME, UserRole.MEMBER),
            (JUDGE_USERNAME, UserRole.JUDGE),
        ):
            session.add(
                User(
                    id=user_ids[username],
                    username=username,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=role.value,
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
    context_service = ProjectInvocationContextService(session_factory, clock=lambda: NOW)
    batch_service = RecordBatchBindingService(
        session_factory,
        InMemoryVerifiedEarlyCycleBatchStore(),
        context_validator=context_service,
    )
    run_service = AgentRunService(
        session_factory,
        planner=ExplodingPlanner(),
    )
    queue = RecordingQueue()
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        project_adapter=create_project_http_adapter(
            ProjectService(session_factory), auth_adapter=auth_adapter
        ),
        dataset_adapter=create_dataset_http_adapter(
            DatasetService(session_factory), auth_adapter=auth_adapter
        ),
        record_batch_adapter=create_record_batch_http_adapter(
            batch_service, auth_adapter=auth_adapter
        ),
        agent_run_adapter=create_agent_run_http_adapter(
            run_service,
            auth_adapter=auth_adapter,
            available_tools=ADVANCED_TOOLS,
            queue=queue,
        ),
        analysis_catalog_adapter=create_analysis_catalog_http_adapter(
            auth_adapter=auth_adapter,
            dataset_service=DatasetService(session_factory),
            record_batch_service=batch_service,
            context_service=context_service,
            run_service=run_service,
            available_tools=ADVANCED_TOOLS,
            queue=queue,
        ),
    )
    client = TestClient(app, base_url="https://api.example.test")
    outsider_client = TestClient(app, base_url="https://api.example.test")
    judge_client = TestClient(app, base_url="https://api.example.test")
    for login_client, username in (
        (client, USERNAME),
        (outsider_client, OUTSIDER_USERNAME),
        (judge_client, JUDGE_USERNAME),
    ):
        login = login_client.post(
            "/v1/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": username, "password": PASSWORD},
        )
        assert login.status_code == 200
    project = client.post(
        "/v1/projects",
        headers={"Origin": ORIGIN},
        json={"name": "Analysis catalog project"},
    )
    assert project.status_code == 201
    project_id = project.json()["project_id"]
    with session_factory.begin() as session:
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=user_ids[JUDGE_USERNAME],
                project_id=project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    dataset = client.post(
        "/v1/datasets",
        headers={"Origin": ORIGIN},
        json={
            "project_id": project_id,
            "name": "Catalog dataset",
            "data_version": "catalog-data-v1",
            "schema_version": "cycle-record-v1",
        },
    )
    assert dataset.status_code == 201
    dataset_id = dataset.json()["dataset_id"]
    batches = []
    for cell_id, cutoff_cycle in (("cell-catalog", 20), ("cell-catalog-2", 50)):
        batch = client.post(
            f"/v1/datasets/{dataset_id}/batches/canonical-csv",
            headers={"Origin": ORIGIN},
            json=_upload_body(cell_id=cell_id, cutoff_cycle=cutoff_cycle),
        )
        assert batch.status_code == 201, batch.text
        batches.append(batch.json()["record_batch_id"])
    frozen = client.post(
        f"/v1/datasets/{dataset_id}/freeze",
        headers={"Origin": ORIGIN},
    )
    assert frozen.status_code == 200
    return _Context(
        client=client,
        outsider_client=outsider_client,
        judge_client=judge_client,
        queue=queue,
        project_id=project_id,
        dataset_id=dataset_id,
        record_batch_ids=(batches[0], batches[1]),
    )


def test_catalog_endpoints_return_project_scoped_inputs_and_empty_run_results(
    catalog_context: _Context,
) -> None:
    context = catalog_context

    datasets = context.client.get(f"/v1/projects/{context.project_id}/datasets")
    assert datasets.status_code == 200
    assert [item["dataset_id"] for item in datasets.json()] == [context.dataset_id]

    batches = context.client.get(f"/v1/datasets/{context.dataset_id}/batches")
    assert batches.status_code == 200
    assert [item["record_batch_id"] for item in batches.json()] == [
        context.record_batch_ids[0],
        context.record_batch_ids[1],
    ]

    inputs = context.client.get(
        f"/v1/projects/{context.project_id}/analysis-inputs"
    )
    assert inputs.status_code == 200
    assert inputs.json()["project_id"] == context.project_id
    assert [item["dataset_id"] for item in inputs.json()["datasets"]] == [
        context.dataset_id
    ]
    assert [item["record_batch_id"] for item in inputs.json()["batches"]] == [
        context.record_batch_ids[0],
        context.record_batch_ids[1],
    ]

    runs = context.client.get(f"/v1/projects/{context.project_id}/agent/runs")
    assert runs.status_code == 200
    assert runs.json() == []


def test_advanced_analysis_creates_one_fixed_nine_step_run_idempotently(
    catalog_context: _Context,
) -> None:
    context = catalog_context
    headers = {
        "Origin": ORIGIN,
        "Idempotency-Key": "advanced-analysis-catalog-0001",
    }
    payload = {
        "record_batch_id": context.record_batch_ids[0],
        "cell_id": "cell-catalog",
        "cutoff_cycle": 20,
    }

    created = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers=headers,
        json=payload,
    )
    repeated = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers=headers,
        json=payload,
    )

    assert created.status_code == 202, created.text
    assert repeated.status_code == 202
    assert repeated.json() == created.json()
    assert created.json()["intent"]["dataset_ids"] == [context.record_batch_ids[0]]
    assert tuple(step["step_id"] for step in created.json()["plan"]["steps"]) == (
        FIXED_STEP_IDS
    )
    assert len(context.queue.calls) == 1

    runs = context.client.get(f"/v1/projects/{context.project_id}/agent/runs")
    assert runs.status_code == 200
    assert [item["run_id"] for item in runs.json()] == [created.json()["run_id"]]

    results = context.client.get(
        f"/v1/agent/runs/{created.json()['run_id']}/results"
    )
    assert results.status_code == 200
    assert results.json() == []


def test_catalog_access_and_advanced_analysis_conflicts_fail_closed(
    catalog_context: _Context,
) -> None:
    context = catalog_context
    dataset_path = f"/v1/projects/{context.project_id}/datasets"
    assert context.outsider_client.get(dataset_path).status_code == 404
    assert context.judge_client.get(dataset_path).status_code == 200

    forbidden = context.judge_client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers={"Origin": ORIGIN, "Idempotency-Key": "judge-advanced-0001"},
        json={
            "record_batch_id": context.record_batch_ids[0],
            "cell_id": "cell-catalog",
            "cutoff_cycle": 20,
        },
    )
    assert forbidden.status_code == 403

    missing_origin = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers={"Idempotency-Key": "missing-origin-advanced-0001"},
        json={
            "record_batch_id": context.record_batch_ids[0],
            "cell_id": "cell-catalog",
            "cutoff_cycle": 20,
        },
    )
    assert missing_origin.status_code == 403

    mismatch = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers={"Origin": ORIGIN, "Idempotency-Key": "mismatch-advanced-0001"},
        json={
            "record_batch_id": context.record_batch_ids[0],
            "cell_id": "another-cell",
            "cutoff_cycle": 20,
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"] == "analysis_input_conflict"
    assert context.queue.calls == []

    headers = {
        "Origin": ORIGIN,
        "Idempotency-Key": "different-batch-conflict-0001",
    }
    first = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers=headers,
        json={
            "record_batch_id": context.record_batch_ids[0],
            "cell_id": "cell-catalog",
            "cutoff_cycle": 20,
        },
    )
    conflict = context.client.post(
        f"/v1/projects/{context.project_id}/advanced-analyses",
        headers=headers,
        json={
            "record_batch_id": context.record_batch_ids[1],
            "cell_id": "cell-catalog-2",
            "cutoff_cycle": 50,
        },
    )
    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency_conflict"
    assert len(context.queue.calls) == 1


def test_analysis_catalog_requires_the_complete_fixed_advanced_tool_set() -> None:
    placeholder = cast(Any, object())
    with pytest.raises(ValueError, match="complete advanced tool set"):
        create_analysis_catalog_http_adapter(
            auth_adapter=placeholder,
            dataset_service=placeholder,
            record_batch_service=placeholder,
            context_service=placeholder,
            run_service=placeholder,
            available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
            queue=placeholder,
        )
