"""Reviewed PostgreSQL corpus resolver with a deterministic BM25 fallback."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, cast

from pydantic import Field, field_validator, model_validator
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


class QueryEmbeddingProvider(Protocol):
    """Versioned query embedding boundary; credentials remain provider-owned."""

    @property
    def model_version(self) -> str: ...

    def embed_query(self, query: str) -> Sequence[float]: ...


class EvidenceReranker(Protocol):
    """Versioned passage reranker returning bounded relevance scores."""

    @property
    def model_version(self) -> str: ...

    def score(
        self,
        *,
        query: str,
        passages: tuple[str, ...],
    ) -> Sequence[float]: ...


class HybridRetrievalPolicy(ContractModel):
    """Versioned score policy for reviewed pgvector/BM25/reranker retrieval."""

    policy_version: str = Field(min_length=1, max_length=100)
    embedding_model_version: str = Field(min_length=1, max_length=100)
    reranker_model_version: str = Field(min_length=1, max_length=100)
    embedding_dimensions: int = Field(default=1_536, ge=1, le=16_384)
    candidate_pool_size: int = Field(default=50, ge=20, le=500)
    bm25_weight: float = Field(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False)
    vector_weight: float = Field(default=0.35, ge=0.0, le=1.0, allow_inf_nan=False)
    reranker_weight: float = Field(default=0.40, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator(
        "policy_version",
        "embedding_model_version",
        "reranker_model_version",
    )
    @classmethod
    def versions_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("hybrid retrieval versions must not be blank")
        return normalized

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> HybridRetrievalPolicy:
        total = self.bm25_weight + self.vector_weight + self.reranker_weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("hybrid retrieval weights must sum to 1.0")
        return self


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


class DatabaseHybridEvidenceBackend:
    """Combine reviewed pgvector storage, Chinese BM25 and a versioned reranker."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        text_loader: KnowledgeChunkTextLoader,
        embedding_provider: QueryEmbeddingProvider,
        reranker: EvidenceReranker,
        policy: HybridRetrievalPolicy,
        backend_version: str = "database-pgvector-bm25-reranker-v1",
    ) -> None:
        normalized_backend = backend_version.strip()
        if not normalized_backend:
            raise ValueError("backend_version must not be blank")
        self._session_factory = session_factory
        self._text_loader = text_loader
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._policy = HybridRetrievalPolicy.model_validate(
            policy.model_dump(mode="json")
        )
        self._backend_version = normalized_backend
        if embedding_provider.model_version != self._policy.embedding_model_version:
            raise ValueError("embedding provider version does not match retrieval policy")
        if reranker.model_version != self._policy.reranker_model_version:
            raise ValueError("reranker version does not match retrieval policy")
        self._fallback = DatabaseBm25EvidenceBackend(
            session_factory,
            text_loader=text_loader,
        )

    def search(
        self,
        *,
        scope: VerifiedKnowledgeScope,
        query: str,
        top_k: int,
    ) -> HybridEvidenceSearchResponse:
        validated_scope = VerifiedKnowledgeScope.model_validate(
            scope.model_dump(mode="json")
        )
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("knowledge query must not be blank")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        if validated_scope.retrieval_policy_version != self._policy.policy_version:
            raise ValueError("knowledge scope retrieval policy does not match hybrid backend")

        document_ids = tuple(
            item.document_id for item in validated_scope.approved_documents
        )
        with session_scope(self._session_factory) as session:
            rows = tuple(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id.in_(document_ids))
                    .order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)
                ).all()
            )
        if not rows or any(
            row.embedding is None
            or row.embedding_model_version != self._policy.embedding_model_version
            for row in rows
        ):
            return self._fallback_response(
                scope=validated_scope,
                query=normalized_query,
                top_k=top_k,
                reason="HYBRID_RETRIEVAL_UNAVAILABLE_INCOMPLETE_EMBEDDINGS",
            )

        corpus = tuple(
            (
                row,
                self._load_verified_text(row),
                _validated_vector(
                    cast(list[float], row.embedding),
                    dimensions=self._policy.embedding_dimensions,
                    label=f"knowledge chunk {row.id}",
                ),
            )
            for row in rows
        )
        query_vector = _validated_vector(
            self._embedding_provider.embed_query(normalized_query),
            dimensions=self._policy.embedding_dimensions,
            label="knowledge query",
        )
        tokenized_corpus = [_tokens(text) for _, text, _ in corpus]
        bm25_raw = _bm25_scores(tokenized_corpus, _tokens(normalized_query))
        bm25_scores = _normalize_nonnegative_scores(bm25_raw)
        vector_scores = [
            _normalized_cosine_similarity(query_vector, vector)
            for _, _, vector in corpus
        ]
        candidate_indices = _candidate_indices(
            bm25_scores,
            vector_scores,
            limit=self._policy.candidate_pool_size,
        )
        passages = tuple(corpus[index][1] for index in candidate_indices)
        reranker_scores = _validated_reranker_scores(
            self._reranker.score(query=normalized_query, passages=passages),
            expected_count=len(passages),
        )

        scored: list[tuple[KnowledgeChunk, str, float, float, float, float]] = []
        for index, reranker_score in zip(
            candidate_indices,
            reranker_scores,
            strict=True,
        ):
            row, text, _ = corpus[index]
            bm25_score = bm25_scores[index]
            vector_score = vector_scores[index]
            final_score = (
                self._policy.bm25_weight * bm25_score
                + self._policy.vector_weight * vector_score
                + self._policy.reranker_weight * reranker_score
            )
            scored.append(
                (
                    row,
                    text,
                    bm25_score,
                    vector_score,
                    reranker_score,
                    final_score,
                )
            )
        ranked = sorted(
            scored,
            key=lambda item: (-item[5], item[0].document_id, item[0].chunk_index),
        )
        hits = tuple(
            HybridEvidenceHit(
                document_id=row.document_id,
                chunk_id=row.id,
                page_number=row.page_start,
                section_label=row.section_label,
                excerpt=text[:1_200],
                evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
                bm25_score=bm25_score,
                vector_score=vector_score,
                reranker_score=reranker_score,
                final_score=final_score,
            )
            for row, text, bm25_score, vector_score, reranker_score, final_score in ranked[
                :top_k
            ]
        )
        return HybridEvidenceSearchResponse(
            backend_version=self._backend_version,
            retrieval_mode="pgvector_bm25_reranker",
            hits=hits,
            warnings=(),
        )

    def _load_verified_text(self, row: KnowledgeChunk) -> str:
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
        return text.strip()

    def _fallback_response(
        self,
        *,
        scope: VerifiedKnowledgeScope,
        query: str,
        top_k: int,
        reason: str,
    ) -> HybridEvidenceSearchResponse:
        response = self._fallback.search(scope=scope, query=query, top_k=top_k)
        return response.model_copy(
            update={"warnings": tuple(dict.fromkeys((*response.warnings, reason)))}
        )


def _validated_vector(
    value: Sequence[float],
    *,
    dimensions: int,
    label: str,
) -> tuple[float, ...]:
    vector = tuple(value)
    if len(vector) != dimensions:
        raise ValueError(f"{label} embedding dimension does not match retrieval policy")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in vector
    ):
        raise ValueError(f"{label} embedding must contain only finite numbers")
    normalized = tuple(float(item) for item in vector)
    if math.isclose(sum(item * item for item in normalized), 0.0, abs_tol=1e-24):
        raise ValueError(f"{label} embedding must not be a zero vector")
    return normalized


def _normalize_nonnegative_scores(values: Sequence[float]) -> list[float]:
    maximum = max(values, default=0.0)
    if maximum <= 0.0:
        return [0.0 for _ in values]
    return [max(0.0, float(value)) / maximum for value in values]


def _normalized_cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    cosine = max(-1.0, min(1.0, dot_product / (left_norm * right_norm)))
    return (cosine + 1.0) / 2.0


def _candidate_indices(
    bm25_scores: Sequence[float],
    vector_scores: Sequence[float],
    *,
    limit: int,
) -> tuple[int, ...]:
    candidate_count = min(limit, len(bm25_scores))
    bm25_ranked = sorted(
        range(len(bm25_scores)),
        key=lambda index: (-bm25_scores[index], index),
    )[:candidate_count]
    vector_ranked = sorted(
        range(len(vector_scores)),
        key=lambda index: (-vector_scores[index], index),
    )[:candidate_count]
    return tuple(dict.fromkeys((*bm25_ranked, *vector_ranked)))


def _validated_reranker_scores(
    values: Sequence[float],
    *,
    expected_count: int,
) -> tuple[float, ...]:
    scores = tuple(values)
    if len(scores) != expected_count:
        raise ValueError("reranker score count does not match candidate count")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) < 0.0
        or float(item) > 1.0
        for item in scores
    ):
        raise ValueError("reranker scores must be finite values between 0 and 1")
    return tuple(float(item) for item in scores)


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
    "DatabaseHybridEvidenceBackend",
    "DatabaseKnowledgeConfig",
    "DatabaseVerifiedKnowledgeScopeResolver",
    "EvidenceReranker",
    "HybridRetrievalPolicy",
    "KnowledgeChunkTextLoader",
    "QueryEmbeddingProvider",
]
