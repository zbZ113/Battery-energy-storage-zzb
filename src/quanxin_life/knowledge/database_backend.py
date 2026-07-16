"""Reviewed PostgreSQL corpus resolver with a deterministic BM25 fallback."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from pydantic import Field, field_validator
from sqlalchemy import select

from quanxin_life.core import (
    EvidenceLevel,
    KnowledgeReviewStatus,
    ProvenanceRecord,
    SourceKind,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import KnowledgeChunk, KnowledgeDocument
from quanxin_life.tools.battery_evidence import (
    ApprovedKnowledgeDocument,
    HybridEvidenceHit,
    HybridEvidenceSearchResponse,
    VerifiedKnowledgeScope,
)

_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)


class DatabaseKnowledgeConfig(ContractModel):
    """Versioned identity of one project-scoped, review-approved corpus."""

    knowledge_scope_id: str = Field(min_length=1, max_length=100)
    project_id: str = Field(min_length=1, max_length=64)
    corpus_version: str = Field(min_length=1, max_length=100)
    index_version: str = Field(min_length=1, max_length=100)
    retrieval_policy_version: str = Field(min_length=1, max_length=100)

    @field_validator(
        "knowledge_scope_id",
        "project_id",
        "corpus_version",
        "index_version",
        "retrieval_policy_version",
    )
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge configuration identifiers must not be blank")
        return normalized


class KnowledgeChunkTextLoader(Protocol):
    """Read one immutable text object; the backend rechecks its digest."""

    def load_verified_text(
        self,
        *,
        object_uri: str,
        expected_sha256: str,
        size_bytes: int,
        content_type: str,
    ) -> str: ...


class DatabaseVerifiedKnowledgeScopeResolver:
    """Resolve only administrator-reviewed sources from one project corpus."""

    def __init__(
        self,
        session_factory: SessionFactory,
        config: DatabaseKnowledgeConfig,
    ) -> None:
        self._session_factory = session_factory
        self._config = DatabaseKnowledgeConfig.model_validate(config.model_dump(mode="json"))

    def resolve_verified_knowledge_scope(
        self,
        knowledge_scope_id: str,
    ) -> VerifiedKnowledgeScope:
        if knowledge_scope_id != self._config.knowledge_scope_id:
            raise ValueError("knowledge scope was not found")
        with session_scope(self._session_factory) as session:
            rows = tuple(
                session.scalars(
                    select(KnowledgeDocument)
                    .where(
                        KnowledgeDocument.project_id == self._config.project_id,
                        KnowledgeDocument.review_status
                        == KnowledgeReviewStatus.APPROVED.value,
                    )
                    .order_by(KnowledgeDocument.title, KnowledgeDocument.id)
                ).all()
            )
        if not rows:
            raise ValueError("knowledge scope has no approved documents")
        documents: list[ApprovedKnowledgeDocument] = []
        provenance: list[ProvenanceRecord] = []
        for row in rows:
            if row.reviewer_user_id is None or row.reviewed_at is None:
                raise ValueError("approved knowledge document lacks review evidence")
            source_id = f"knowledge-document:{row.id}"
            documents.append(
                ApprovedKnowledgeDocument(
                    document_id=row.id,
                    source_id=source_id,
                    title=row.title,
                    source_uri=row.source_uri,
                    source_sha256=row.source_sha256,
                    license_name=row.license_name,
                    evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
                )
            )
            provenance.append(
                ProvenanceRecord(
                    source_id=source_id,
                    source_kind=SourceKind.OBSERVED,
                    uri=row.source_uri,
                    sha256=row.source_sha256,
                    description=f"Reviewed knowledge source: {row.title}",
                    created_at=row.reviewed_at,
                )
            )
        return VerifiedKnowledgeScope(
            knowledge_scope_id=self._config.knowledge_scope_id,
            corpus_version=self._config.corpus_version,
            index_version=self._config.index_version,
            retrieval_policy_version=self._config.retrieval_policy_version,
            approved_documents=tuple(documents),
            provenance=tuple(provenance),
        )


class DatabaseBm25EvidenceBackend:
    """Search reviewed chunks when embedding or pgvector is unavailable."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        text_loader: KnowledgeChunkTextLoader,
        backend_version: str = "database-bm25-v1",
    ) -> None:
        normalized = backend_version.strip()
        if not normalized:
            raise ValueError("backend_version must not be blank")
        self._session_factory = session_factory
        self._text_loader = text_loader
        self._backend_version = normalized

    def search(
        self,
        *,
        scope: VerifiedKnowledgeScope,
        query: str,
        top_k: int,
    ) -> HybridEvidenceSearchResponse:
        validated_scope = VerifiedKnowledgeScope.model_validate(scope.model_dump(mode="json"))
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("knowledge query must not be blank")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        document_ids = tuple(item.document_id for item in validated_scope.approved_documents)
        with session_scope(self._session_factory) as session:
            rows = tuple(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id.in_(document_ids))
                    .order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)
                ).all()
            )
        corpus: list[tuple[KnowledgeChunk, str, list[str]]] = []
        for row in rows:
            text = self._text_loader.load_verified_text(
                object_uri=row.text_object_uri,
                expected_sha256=row.text_sha256,
                size_bytes=row.text_size_bytes,
                content_type=row.text_content_type,
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError("knowledge chunk text must be nonblank UTF-8 text")
            actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if actual_sha256 != row.text_sha256:
                raise ValueError("knowledge chunk text SHA-256 does not match its manifest")
            corpus.append((row, text.strip(), _tokens(text)))
        query_tokens = _tokens(normalized_query)
        if not corpus or not query_tokens:
            return self._response(())
        scores = _bm25_scores([tokens for _, _, tokens in corpus], query_tokens)
        ranked = sorted(
            zip(corpus, scores, strict=True),
            key=lambda item: (-item[1], item[0][0].document_id, item[0][0].chunk_index),
        )
        hits: list[HybridEvidenceHit] = []
        for (row, text, _), score in ranked:
            if score <= 0:
                continue
            hits.append(
                HybridEvidenceHit(
                    document_id=row.document_id,
                    chunk_id=row.id,
                    page_number=row.page_start,
                    section_label=row.section_label,
                    excerpt=text[:1_200],
                    evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
                    bm25_score=score,
                    vector_score=0.0,
                    reranker_score=0.0,
                    final_score=score,
                )
            )
            if len(hits) == top_k:
                break
        return self._response(tuple(hits))

    def _response(
        self,
        hits: tuple[HybridEvidenceHit, ...],
    ) -> HybridEvidenceSearchResponse:
        return HybridEvidenceSearchResponse(
            backend_version=self._backend_version,
            retrieval_mode="bm25_fallback",
            hits=hits,
            warnings=("VECTOR_RETRIEVAL_UNAVAILABLE",),
        )


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(value)]


def _bm25_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    document_count = len(corpus)
    average_length = sum(len(tokens) for tokens in corpus) / document_count
    document_frequency = Counter(
        token for tokens in corpus for token in set(tokens)
    )
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    for tokens in corpus:
        frequencies = Counter(tokens)
        score = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if frequency == 0:
                continue
            frequency_in_documents = document_frequency[token]
            inverse_frequency = math.log(
                1
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            length_normalizer = 1 - b + b * len(tokens) / max(average_length, 1.0)
            score += inverse_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_normalizer)
            )
        scores.append(score)
    return scores


__all__ = [
    "DatabaseBm25EvidenceBackend",
    "DatabaseKnowledgeConfig",
    "DatabaseVerifiedKnowledgeScopeResolver",
    "KnowledgeChunkTextLoader",
]
