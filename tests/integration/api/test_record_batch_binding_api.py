from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.record_batches import create_record_batch_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.ingestion import (
    CANONICAL_CYCLE_CSV_FIELDS,
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.record_batch_bindings import RecordBatchBindingService
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthPrincipal,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import (
    CellMetadata,
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    UserRole,
    UserStatus,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import Project, User

NOW = datetime(2026, 7, 25, 14, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
PASSWORD = "RecordBatchBinding-2026!"


@dataclass(frozen=True, slots=True)
class _ApiContext:
    owner_client: TestClient
    outsider_client: TestClient
    active_project_id: str
    archived_project_id: str
    draft_dataset_id: str
    archived_project_dataset_id: str
    frozen_dataset_id: str


def _principal(user_id: str, username: str) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=username,
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200


def _csv_payload(*, cell_id: str = "cell-http-1", last_cycle: int = 20) -> bytes:
    header = ",".join(CANONICAL_CYCLE_CSV_FIELDS)
    rows = [
        f"UPLOAD,{cell_id},1,0,0,3.1,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},1,1,1,3.2,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},{last_cycle},0,0,3.1,-1,25,1.0,0.9,0.02,true,true",
        f"UPLOAD,{cell_id},{last_cycle},1,1,3.2,-1,25,1.0,0.9,0.02,true,true",
    ]
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _registration(payload: bytes) -> CanonicalCsvBatchRegistration:
    digest = hashlib.sha256(payload).hexdigest()
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="cell-http-1",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="upload://canonical/cell-http-1.csv",
            source_sha256=digest,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="upload-data-v1",
        split_version="upload-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="canonical-upload-cell-http-1",
                source_kind=SourceKind.OBSERVED,
                uri="upload://canonical/cell-http-1.csv",
                sha256=digest,
                description="Observed canonical CSV HTTP integration fixture",
                created_at=NOW,
            ),
        ),
    )


def _upload_body() -> dict[str, object]:
    payload = _csv_payload()
    return {
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "registration": _registration(payload).model_dump(mode="json"),
    }


def _api_context(tmp_path: Path) -> _ApiContext:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'record-batches.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    owner_user_id = str(uuid4())
    outsider_user_id = str(uuid4())
    owner_username = f"record-owner-{owner_user_id[:8]}@example.test"
    outsider_username = f"record-outsider-{outsider_user_id[:8]}@example.test"
    with session_factory.begin() as session:
        for user_id, username in (
            (owner_user_id, owner_username),
            (outsider_user_id, outsider_username),
        ):
            session.add(
                User(
                    id=user_id,
                    username=username,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=UserRole.MEMBER.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )

    owner_principal = _principal(owner_user_id, owner_username)
    project_service = ProjectService(session_factory)
    dataset_service = DatasetService(session_factory)
    active_project = project_service.create_project(
        owner_principal,
        name="Record batch active project",
        now=NOW,
    )
    archived_project = project_service.create_project(
        owner_principal,
        name="Record batch archived project",
        now=NOW,
    )
    draft_dataset = dataset_service.create_dataset(
        owner_principal,
        project_id=active_project.project_id,
        name="Record batch draft dataset",
        data_version="upload-data-v1",
        schema_version="cycle-record-v1",
        now=NOW,
    )
    frozen_dataset = dataset_service.create_dataset(
        owner_principal,
        project_id=active_project.project_id,
        name="Record batch frozen dataset",
        data_version="upload-data-v1",
        schema_version="cycle-record-v1",
        now=NOW,
    )
    frozen_dataset = dataset_service.freeze_dataset(
        owner_principal,
        frozen_dataset.dataset_id,
        now=NOW,
    )
    archived_project_dataset = dataset_service.create_dataset(
        owner_principal,
        project_id=archived_project.project_id,
        name="Record batch archived-project dataset",
        data_version="upload-data-v1",
        schema_version="cycle-record-v1",
        now=NOW,
    )
    with session_factory.begin() as session:
        persisted = session.get(Project, archived_project.project_id)
        assert persisted is not None
        persisted.status = ProjectStatus.ARCHIVED.value
        persisted.updated_at = NOW

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
    record_batch_service = RecordBatchBindingService(
        session_factory,
        batch_store=InMemoryVerifiedEarlyCycleBatchStore(),
        context_validator=ProjectInvocationContextService(session_factory),
    )
    record_batch_adapter = create_record_batch_http_adapter(
        record_batch_service,
        auth_adapter=auth_adapter,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        record_batch_adapter=record_batch_adapter,
    )
    owner_client = TestClient(app, base_url="https://api.example.test")
    outsider_client = TestClient(app, base_url="https://api.example.test")
    _login(owner_client, owner_username)
    _login(outsider_client, outsider_username)
    return _ApiContext(
        owner_client=owner_client,
        outsider_client=outsider_client,
        active_project_id=active_project.project_id,
        archived_project_id=archived_project.project_id,
        draft_dataset_id=draft_dataset.dataset_id,
        archived_project_dataset_id=archived_project_dataset.dataset_id,
        frozen_dataset_id=frozen_dataset.dataset_id,
    )


def test_owner_uploads_to_draft_dataset_with_project_derived_server_side(
    tmp_path: Path,
) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/datasets/{context.draft_dataset_id}/batches/canonical-csv",
        headers={"Origin": ORIGIN},
        json=_upload_body(),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["record_batch_id"]
    assert body["dataset_id"] == context.draft_dataset_id
    assert body["project_id"] == context.active_project_id


def test_outsider_and_archived_project_scopes_are_hidden(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    outsider = context.outsider_client.post(
        f"/v1/datasets/{context.draft_dataset_id}/batches/canonical-csv",
        headers={"Origin": ORIGIN},
        json=_upload_body(),
    )
    archived = context.owner_client.post(
        f"/v1/datasets/{context.archived_project_dataset_id}/batches/canonical-csv",
        headers={"Origin": ORIGIN},
        json=_upload_body(),
    )

    assert outsider.status_code == 404
    assert outsider.json()["detail"] == "record_batch_scope_not_found"
    assert archived.status_code == 404
    assert archived.json()["detail"] == "record_batch_scope_not_found"


def test_frozen_dataset_rejects_new_canonical_batch_upload(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/datasets/{context.frozen_dataset_id}/batches/canonical-csv",
        headers={"Origin": ORIGIN},
        json=_upload_body(),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "record_batch_state_conflict"


def test_canonical_batch_upload_requires_trusted_origin(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/datasets/{context.draft_dataset_id}/batches/canonical-csv",
        json=_upload_body(),
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("field_name", "injected_value"),
    [
        ("project_id", str(uuid4())),
        ("dataset_id", str(uuid4())),
        ("content_batch_id", "caller-selected-content-batch"),
        ("content_sha256", "0" * 64),
        ("registration_sha256", "1" * 64),
    ],
)
def test_body_cannot_inject_scope_or_server_derived_hashes(
    tmp_path: Path,
    field_name: str,
    injected_value: str,
) -> None:
    context = _api_context(tmp_path)
    body = _upload_body()
    body[field_name] = injected_value

    response = context.owner_client.post(
        f"/v1/datasets/{context.draft_dataset_id}/batches/canonical-csv",
        headers={"Origin": ORIGIN},
        json=body,
    )

    assert response.status_code == 422
