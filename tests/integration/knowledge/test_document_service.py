from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import KnowledgeReviewStatus, ProjectStatus, UserRole, UserStatus
from quanxin_life.infrastructure.object_store import StoredObjectRef
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import KnowledgeChunk, Project, User

NOW = datetime(2026, 7, 16, 17, 0, tzinfo=UTC)


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
        payload = self.payloads[stored.uri]
        assert len(payload) == stored.size_bytes
        assert hashlib.sha256(payload).hexdigest() == stored.sha256
        return payload


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.lower()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


def _service_context(tmp_path):  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.documents import KnowledgeDocumentService

    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'documents.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    member = _principal(str(uuid4()), UserRole.MEMBER)
    admin = _principal(str(uuid4()), UserRole.ADMIN)
    project_id = str(uuid4())
    with sessions.begin() as session:
        for principal in (member, admin):
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$knowledge-service-test",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.add(
            Project(
                id=project_id,
                owner_user_id=member.user_id,
                name="knowledge document project",
                status=ProjectStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    store = _MemoryObjectStore()
    return (
        KnowledgeDocumentService(sessions, object_store=store),
        sessions,
        store,
        member,
        admin,
        project_id,
    )


def test_member_upload_admin_review_and_markdown_page_aware_index(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service, sessions, store, member, admin, project_id = _service_context(tmp_path)
    payload = (
        "# LFP老化机理\n\n温度会影响电芯容量衰减。\n\n"
        "## 使用限制\n\n该材料只用于知识解释; 不产生寿命数值。"
    ).encode()

    document = service.upload_document(
        member,
        project_id=project_id,
        title="LFP老化说明",
        source_uri="https://example.test/lfp-aging.md",
        license_name="CC-BY-4.0",
        document_version="v1",
        payload=payload,
        content_type="text/markdown; charset=utf-8",
        now=NOW,
    )

    assert document.review_status is KnowledgeReviewStatus.PENDING
    assert document.source_sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(PermissionError, match="administrator"):
        service.approve_document(member, document.document_id, now=NOW)

    approved = service.approve_document(admin, document.document_id, now=NOW)
    chunks = service.index_document(admin, document.document_id, now=NOW)

    assert approved.review_status is KnowledgeReviewStatus.APPROVED
    assert chunks
    assert {chunk.section_label for chunk in chunks} == {"LFP老化机理", "使用限制"}
    assert all(chunk.page_start is None for chunk in chunks)
    with sessions() as session:
        persisted = tuple(
            session.scalars(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.document_id == document.document_id)
                .order_by(KnowledgeChunk.chunk_index)
            )
        )
    assert len(persisted) == len(chunks)
    assert all(item.text_object_uri in store.payloads for item in persisted)


def test_uploader_cannot_self_approve_even_when_the_uploader_is_admin(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service, _, _, _, admin, project_id = _service_context(tmp_path)
    document = service.upload_document(
        admin,
        project_id=project_id,
        title="Admin supplied source",
        source_uri="https://example.test/admin-source.txt",
        license_name="reviewed-internal",
        document_version="v1",
        payload=b"reviewed text",
        content_type="text/plain; charset=utf-8",
        now=NOW,
    )

    with pytest.raises(PermissionError, match="own document"):
        service.approve_document(admin, document.document_id, now=NOW)


def test_visible_documents_can_be_listed_and_pending_document_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service, _, _, member, admin, project_id = _service_context(tmp_path)
    document = service.upload_document(
        member,
        project_id=project_id,
        title="Rejected source",
        source_uri="https://example.test/rejected.txt",
        license_name="reviewed-internal",
        document_version="v1",
        payload=b"unsupported claim",
        content_type="text/plain",
        now=NOW,
    )

    assert service.list_documents(member, project_id=project_id) == (document,)
    rejected = service.reject_document(admin, document.document_id, now=NOW)

    assert rejected.review_status is KnowledgeReviewStatus.REJECTED
    assert rejected.reviewer_user_id == admin.user_id
    assert rejected.reviewed_at == NOW
    with pytest.raises(ValueError, match="pending"):
        service.approve_document(admin, document.document_id, now=NOW)


def test_member_cannot_reject_a_document(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service, _, _, member, _, project_id = _service_context(tmp_path)
    document = service.upload_document(
        member,
        project_id=project_id,
        title="Pending source",
        source_uri="https://example.test/pending.txt",
        license_name="reviewed-internal",
        document_version="v1",
        payload=b"pending source",
        content_type="text/plain",
        now=NOW,
    )

    with pytest.raises(PermissionError, match="administrator"):
        service.reject_document(member, document.document_id, now=NOW)
