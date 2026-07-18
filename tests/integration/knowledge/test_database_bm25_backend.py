from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import KnowledgeReviewStatus, ProjectStatus, UserRole, UserStatus
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
    session_scope,
)
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
                    idempotency_key_hash=hashlib.sha256(
                        f"knowledge-key-{suffix}".encode()
                    ).hexdigest(),
                    request_hash=hashlib.sha256(
                        f"knowledge-request-{suffix}".encode()
                    ).hexdigest(),
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
class _QueryEmbeddingProvider:
    model_version = "embedding-reviewed-v1"

    def embed_query(self, query: str) -> tuple[float, ...]:
        assert query
        return (1.0, *(0.0 for _ in range(1_535)))


class _EvidenceReranker:
    model_version = "reranker-reviewed-v1"

    def score(
        self,
        *,
        query: str,
        passages: tuple[str, ...],
    ) -> tuple[float, ...]:
        assert query
        return tuple(0.75 for _ in passages)


def test_database_hybrid_backend_combines_reviewed_vector_bm25_and_reranker(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.database_backend import (
        DatabaseHybridEvidenceBackend,
        DatabaseKnowledgeConfig,
        DatabaseVerifiedKnowledgeScopeResolver,
        HybridRetrievalPolicy,
    )

    sessions, project_id, approved_id, _, payloads = _context(tmp_path)
    with session_scope(sessions) as session:
        row = session.scalar(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == approved_id)
        )
        assert row is not None
        row.embedding_model_version = "embedding-reviewed-v1"
        row.embedding = [1.0, *(0.0 for _ in range(1_535))]

    config = DatabaseKnowledgeConfig(
        knowledge_scope_id="lfp-reviewed-v1",
        project_id=project_id,
        corpus_version="corpus-v1",
        index_version="hybrid-v1",
        retrieval_policy_version="reviewed-hybrid-v1",
    )
    scope = DatabaseVerifiedKnowledgeScopeResolver(
        sessions, config
    ).resolve_verified_knowledge_scope(config.knowledge_scope_id)
    response = DatabaseHybridEvidenceBackend(
        sessions,
        text_loader=_TextLoader(payloads),
        embedding_provider=_QueryEmbeddingProvider(),
        reranker=_EvidenceReranker(),
        policy=HybridRetrievalPolicy(
            policy_version="reviewed-hybrid-v1",
            embedding_model_version="embedding-reviewed-v1",
            reranker_model_version="reranker-reviewed-v1",
        ),
    ).search(scope=scope, query="temperature capacity degradation", top_k=5)

    assert response.retrieval_mode == "pgvector_bm25_reranker"
    assert response.warnings == ()
    assert len(response.hits) == 1
    assert response.hits[0].document_id == approved_id
    assert response.hits[0].page_number == 3
    assert response.hits[0].vector_score == pytest.approx(1.0)
    assert response.hits[0].reranker_score == pytest.approx(0.75)
    assert response.hits[0].final_score > 0.0


def test_database_hybrid_backend_degrades_whole_scope_when_embeddings_are_incomplete(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.database_backend import (
        DatabaseHybridEvidenceBackend,
        DatabaseKnowledgeConfig,
        DatabaseVerifiedKnowledgeScopeResolver,
        HybridRetrievalPolicy,
    )

    sessions, project_id, _, _, payloads = _context(tmp_path)
    config = DatabaseKnowledgeConfig(
        knowledge_scope_id="lfp-reviewed-v1",
        project_id=project_id,
        corpus_version="corpus-v1",
        index_version="hybrid-v1",
        retrieval_policy_version="reviewed-hybrid-v1",
    )
    scope = DatabaseVerifiedKnowledgeScopeResolver(
        sessions, config
    ).resolve_verified_knowledge_scope(config.knowledge_scope_id)
    response = DatabaseHybridEvidenceBackend(
        sessions,
        text_loader=_TextLoader(payloads),
        embedding_provider=_QueryEmbeddingProvider(),
        reranker=_EvidenceReranker(),
        policy=HybridRetrievalPolicy(
            policy_version="reviewed-hybrid-v1",
            embedding_model_version="embedding-reviewed-v1",
            reranker_model_version="reranker-reviewed-v1",
        ),
    ).search(scope=scope, query="娓╁害", top_k=5)

    assert response.retrieval_mode == "bm25_fallback"
    assert "VECTOR_RETRIEVAL_UNAVAILABLE" in response.warnings
    assert "HYBRID_RETRIEVAL_UNAVAILABLE_INCOMPLETE_EMBEDDINGS" in response.warnings


class _DocumentEmbeddingProvider:
    model_version = "embedding-reviewed-v1"
    dimensions = 1_536

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return tuple((1.0, *(0.0 for _ in range(1_535))) for _ in texts)


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        user_id="knowledge-index-admin",
        session_id="knowledge-index-session",
        username="knowledge-index-admin@example.test",
        role=UserRole.ADMIN,
        must_change_password=False,
    )


def test_embedding_index_service_indexes_approved_chunks_and_reuses_same_version(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.embedding_index import KnowledgeEmbeddingIndexService

    sessions, _, approved_id, _, payloads = _context(tmp_path)
    provider = _DocumentEmbeddingProvider()
    service = KnowledgeEmbeddingIndexService(
        sessions,
        text_loader=_TextLoader(payloads),
        embedding_provider=provider,
    )

    indexed = service.index_document(_admin_principal(), approved_id)
    reused = service.index_document(_admin_principal(), approved_id)

    assert indexed.document_id == approved_id
    assert indexed.embedding_model_version == provider.model_version
    assert indexed.chunk_count == 1
    assert indexed.reused is False
    assert reused.reused is True
    assert provider.calls == 1
    with session_scope(sessions) as session:
        row = session.scalar(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == approved_id)
        )
        assert row is not None
        assert row.embedding_model_version == provider.model_version
        assert row.embedding == pytest.approx([1.0, *(0.0 for _ in range(1_535))])


def test_embedding_index_service_rejects_unapproved_document(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.embedding_index import KnowledgeEmbeddingIndexService

    sessions, _, _, pending_id, payloads = _context(tmp_path)
    service = KnowledgeEmbeddingIndexService(
        sessions,
        text_loader=_TextLoader(payloads),
        embedding_provider=_DocumentEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="approved"):
        service.index_document(_admin_principal(), pending_id)


def test_embedding_index_service_does_not_publish_invalid_vectors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from quanxin_life.knowledge.embedding_index import KnowledgeEmbeddingIndexService

    class _InvalidEmbeddingProvider:
        model_version = "embedding-invalid-v1"
        dimensions = 1_536

        def embed_documents(
            self,
            texts: tuple[str, ...],
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0,) for _ in texts)

    sessions, _, approved_id, _, payloads = _context(tmp_path)
    service = KnowledgeEmbeddingIndexService(
        sessions,
        text_loader=_TextLoader(payloads),
        embedding_provider=_InvalidEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="1536"):
        service.index_document(_admin_principal(), approved_id)
    with session_scope(sessions) as session:
        row = session.scalar(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == approved_id)
        )
        assert row is not None
        assert row.embedding_model_version is None
        assert row.embedding is None
