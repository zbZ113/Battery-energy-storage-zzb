from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import KnowledgeReviewStatus, ProjectStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    Project,
    User,
)

NOW = datetime(2026, 7, 16, 16, 0, tzinfo=UTC)


class _TextLoader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def load_verified_text(
        self,
        *,
        object_uri: str,
        expected_sha256: str,
        size_bytes: int,
        content_type: str,
    ) -> str:
        del expected_sha256, size_bytes, content_type
        payload = self.payloads[object_uri]
        return payload.decode("utf-8")


def _context(tmp_path):  # type: ignore[no-untyped-def]
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'knowledge.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    approved_id = str(uuid4())
    pending_id = str(uuid4())
    approved_text = "磷酸铁锂电芯的温度与循环工况会影响容量衰减。"
    pending_text = "这份未审核材料不允许进入正式检索结果。"
    payloads = {
        "memory://knowledge/approved": approved_text.encode(),
        "memory://knowledge/pending": pending_text.encode(),
    }
    with sessions.begin() as session:
        session.add(
            User(
                id=user_id,
                username="knowledge@example.test",
                credential_hash="$argon2id$knowledge-test",
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Project(
                id=project_id,
                owner_user_id=user_id,
                name="knowledge project",
                status=ProjectStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for document_id, status, suffix in (
            (approved_id, KnowledgeReviewStatus.APPROVED, "approved"),
            (pending_id, KnowledgeReviewStatus.PENDING, "pending"),
        ):
            text = approved_text if suffix == "approved" else pending_text
            digest = hashlib.sha256(text.encode()).hexdigest()
            session.add(
                KnowledgeDocument(
                    id=document_id,
                    project_id=project_id,
                    created_by_user_id=user_id,
                    title=f"{suffix} LFP source",
                    source_uri=f"https://example.test/{suffix}.pdf",
                    source_sha256=hashlib.sha256(f"source-{suffix}".encode()).hexdigest(),
                    license_name="reviewed-test-license",
                    document_version="v1",
                    review_status=status.value,
                    reviewer_user_id=user_id if status is KnowledgeReviewStatus.APPROVED else None,
                    reviewed_at=NOW if status is KnowledgeReviewStatus.APPROVED else None,
                    object_uri=f"memory://document/{suffix}",
                    object_size_bytes=len(text.encode()),
                    object_content_type="application/pdf",
                    created_at=NOW,
                )
            )
            session.add(
                KnowledgeChunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    chunk_index=0,
                    page_start=3,
                    page_end=3,
                    text_object_uri=f"memory://knowledge/{suffix}",
                    text_sha256=digest,
                    text_size_bytes=len(text.encode()),
                    text_content_type="text/plain; charset=utf-8",
                    section_label="老化机理",
                    embedding_model_version=None,
                    embedding=None,
                    created_at=NOW,
                )
            )
    return sessions, project_id, approved_id, pending_id, payloads


def test_database_scope_and_bm25_backend_only_return_reviewed_page_bound_evidence(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.database_backend import (
        DatabaseBm25EvidenceBackend,
        DatabaseKnowledgeConfig,
        DatabaseVerifiedKnowledgeScopeResolver,
    )

    sessions, project_id, approved_id, pending_id, payloads = _context(tmp_path)
    config = DatabaseKnowledgeConfig(
        knowledge_scope_id="lfp-reviewed-v1",
        project_id=project_id,
        corpus_version="corpus-v1",
        index_version="bm25-v1",
        retrieval_policy_version="reviewed-bm25-v1",
    )
    resolver = DatabaseVerifiedKnowledgeScopeResolver(sessions, config)
    scope = resolver.resolve_verified_knowledge_scope(config.knowledge_scope_id)
    response = DatabaseBm25EvidenceBackend(
        sessions,
        text_loader=_TextLoader(payloads),
    ).search(scope=scope, query="温度如何影响电芯容量衰减", top_k=5)

    assert [item.document_id for item in scope.approved_documents] == [approved_id]
    assert pending_id not in {item.document_id for item in scope.approved_documents}
    assert response.retrieval_mode == "bm25_fallback"
    assert len(response.hits) == 1
    assert response.hits[0].document_id == approved_id
    assert response.hits[0].page_number == 3
    assert "温度" in response.hits[0].excerpt


def test_database_bm25_backend_rejects_modified_chunk_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.database_backend import (
        DatabaseBm25EvidenceBackend,
        DatabaseKnowledgeConfig,
        DatabaseVerifiedKnowledgeScopeResolver,
    )

    sessions, project_id, _, _, payloads = _context(tmp_path)
    config = DatabaseKnowledgeConfig(
        knowledge_scope_id="lfp-reviewed-v1",
        project_id=project_id,
        corpus_version="corpus-v1",
        index_version="bm25-v1",
        retrieval_policy_version="reviewed-bm25-v1",
    )
    resolver = DatabaseVerifiedKnowledgeScopeResolver(sessions, config)
    scope = resolver.resolve_verified_knowledge_scope(config.knowledge_scope_id)
    payloads["memory://knowledge/approved"] = b"tampered"

    with pytest.raises(ValueError, match="SHA-256"):
        DatabaseBm25EvidenceBackend(
            sessions,
            text_loader=_TextLoader(payloads),
        ).search(scope=scope, query="温度", top_k=5)
