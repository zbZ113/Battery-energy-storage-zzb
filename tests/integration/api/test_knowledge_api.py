from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.knowledge import create_knowledge_http_adapter
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import ProjectStatus, UserRole, UserStatus
from quanxin_life.infrastructure.object_store import StoredObjectRef
from quanxin_life.knowledge.documents import KnowledgeDocumentService
from quanxin_life.knowledge.embedding_index import KnowledgeEmbeddingIndexService
from quanxin_life.knowledge.object_loader import MinioKnowledgeTextLoader
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import Project, User

ORIGIN = "https://app.example.test"
MEMBER_USERNAME = "knowledge-member@example.test"
ADMIN_USERNAME = "knowledge-admin@example.test"
PASSWORD = "knowledge API passphrase 2026"
NOW = datetime(2026, 7, 16, 18, 0, tzinfo=UTC)


class _MemoryObjectStore:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def put_bytes(
        self,
        *,
        namespace: str,
        payload: bytes,
        expected_sha256: str,
        content_type: str,
    ) -> StoredObjectRef:
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        uri = f"memory://{namespace}/{expected_sha256}"
        self.payloads[uri] = payload
        return StoredObjectRef(
            uri=uri,
            sha256=expected_sha256,
            size_bytes=len(payload),
            content_type=content_type,
        )

    def get_bytes(self, stored: StoredObjectRef) -> bytes:
        return self.payloads[stored.uri]


class _DocumentEmbeddingProvider:
    model_version = "knowledge-api-embedding-v1"
    dimensions = 1_536

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, *(0.0 for _ in range(1_535))) for _ in texts)


@pytest.fixture
def knowledge_client(tmp_path: Path) -> tuple[TestClient, str]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'knowledge-api.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    member_id = str(uuid4())
    admin_id = str(uuid4())
    project_id = str(uuid4())
    with sessions.begin() as session:
        session.add_all(
            (
                User(
                    id=member_id,
                    username=MEMBER_USERNAME,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=UserRole.MEMBER.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                User(
                    id=admin_id,
                    username=ADMIN_USERNAME,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=UserRole.ADMIN.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Project(
                    id=project_id,
                    owner_user_id=member_id,
                    name="knowledge API project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
            )
        )
    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(sessions),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,)),
    )
    object_store = _MemoryObjectStore()
    knowledge_adapter = create_knowledge_http_adapter(
        KnowledgeDocumentService(sessions, object_store=object_store),
        auth_adapter=auth_adapter,
        embedding_service=KnowledgeEmbeddingIndexService(
            sessions,
            text_loader=MinioKnowledgeTextLoader(object_store),
            embedding_provider=_DocumentEmbeddingProvider(),
        ),
    )
    app = FastAPI()
    app.include_router(auth_adapter.router)
    app.include_router(knowledge_adapter.router)
    client = TestClient(app, base_url="https://api.example.test")
    _login(client, MEMBER_USERNAME)
    return client, project_id


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200, response.text


def _upload(client: TestClient, project_id: str, *, suffix: str) -> dict[str, object]:
    payload = f"# Reviewed evidence {suffix}\n\nThis text explains a limitation.".encode()
    response = client.post(
        "/v1/knowledge/documents",
        headers={"Origin": ORIGIN, "Idempotency-Key": f"knowledge-{suffix}-0001"},
        json={
            "project_id": project_id,
            "title": f"Reviewed evidence {suffix}",
            "source_uri": f"https://example.test/{suffix}.md",
            "license_name": "CC-BY-4.0",
            "document_version": "v1",
            "content_type": "text/markdown; charset=utf-8",
            "content_base64": base64.b64encode(payload).decode("ascii"),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_member_uploads_and_lists_then_independent_admin_approves_and_indexes(
    knowledge_client: tuple[TestClient, str],
) -> None:
    client, project_id = knowledge_client
    document = _upload(client, project_id, suffix="approved")

    listed = client.get(
        "/v1/knowledge/documents", params={"project_id": project_id}
    )
    assert listed.status_code == 200
    assert listed.json() == [document]
    forbidden = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/approve",
        headers={"Origin": ORIGIN},
    )
    assert forbidden.status_code == 403

    _login(client, ADMIN_USERNAME)
    approved = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/approve",
        headers={"Origin": ORIGIN},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["review_status"] == "APPROVED"
    indexed = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/index",
        headers={"Origin": ORIGIN},
    )
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()[0]["document_id"] == document["document_id"]
    assert indexed.json()[0]["section_label"] == "Reviewed evidence approved"
    embedded = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/embeddings",
        headers={"Origin": ORIGIN},
    )
    assert embedded.status_code == 200, embedded.text
    assert embedded.json() == {
        "document_id": document["document_id"],
        "embedding_model_version": "knowledge-api-embedding-v1",
        "chunk_count": 1,
        "reused": False,
    }


def test_admin_can_reject_pending_document_and_invalid_base64_is_rejected(
    knowledge_client: tuple[TestClient, str],
) -> None:
    client, project_id = knowledge_client
    document = _upload(client, project_id, suffix="rejected")
    invalid = client.post(
        "/v1/knowledge/documents",
        headers={"Origin": ORIGIN, "Idempotency-Key": "knowledge-invalid-0001"},
        json={
            "project_id": project_id,
            "title": "Invalid content",
            "source_uri": "https://example.test/invalid.txt",
            "license_name": "CC-BY-4.0",
            "document_version": "v1",
            "content_type": "text/plain",
            "content_base64": "not-valid-base64!",
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "invalid_knowledge_document"

    _login(client, ADMIN_USERNAME)
    rejected = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/reject",
        headers={"Origin": ORIGIN},
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["review_status"] == "REJECTED"
    cannot_index = client.post(
        f"/v1/knowledge/documents/{document['document_id']}/index",
        headers={"Origin": ORIGIN},
    )
    assert cannot_index.status_code == 409
    assert cannot_index.json()["detail"] == "knowledge_document_state_conflict"


def test_knowledge_write_requires_trusted_origin(
    knowledge_client: tuple[TestClient, str],
) -> None:
    client, project_id = knowledge_client
    response = client.post(
        "/v1/knowledge/documents",
        headers={"Idempotency-Key": "knowledge-no-origin-0001"},
        json={
            "project_id": project_id,
            "title": "No origin",
            "source_uri": "https://example.test/no-origin.txt",
            "license_name": "CC-BY-4.0",
            "document_version": "v1",
            "content_type": "text/plain",
            "content_base64": base64.b64encode(b"safe text").decode("ascii"),
        },
    )
    assert response.status_code == 403


def test_upload_requires_idempotency_and_replay_does_not_duplicate(
    knowledge_client: tuple[TestClient, str],
) -> None:
    client, project_id = knowledge_client
    request = {
        "project_id": project_id,
        "title": "Idempotent evidence",
        "source_uri": "https://example.test/idempotent.md",
        "license_name": "CC-BY-4.0",
        "document_version": "v1",
        "content_type": "text/markdown",
        "content_base64": base64.b64encode(b"# Stable evidence").decode("ascii"),
    }

    missing = client.post(
        "/v1/knowledge/documents",
        headers={"Origin": ORIGIN},
        json=request,
    )
    assert missing.status_code == 422

    headers = {"Origin": ORIGIN, "Idempotency-Key": "knowledge-stable-0001"}
    created = client.post("/v1/knowledge/documents", headers=headers, json=request)
    repeated = client.post("/v1/knowledge/documents", headers=headers, json=request)
    conflict = client.post(
        "/v1/knowledge/documents",
        headers=headers,
        json={**request, "title": "Different request"},
    )

    assert created.status_code == 201
    assert repeated.status_code == 201
    assert repeated.json() == created.json()
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency_conflict"


def test_blank_project_filter_is_validation_error(
    knowledge_client: tuple[TestClient, str],
) -> None:
    client, _ = knowledge_client

    response = client.get(
        "/v1/knowledge/documents",
        params={"project_id": "   "},
    )

    assert response.status_code == 422
