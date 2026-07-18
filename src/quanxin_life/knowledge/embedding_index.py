"""Review-gated, idempotent embedding indexing for immutable knowledge chunks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field
from sqlalchemy import select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import KnowledgeReviewStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.knowledge.database_backend import KnowledgeChunkTextLoader
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import KnowledgeChunk, KnowledgeDocument

KNOWLEDGE_EMBEDDING_DIMENSIONS = 1_536


class DocumentEmbeddingProvider(Protocol):
    """Versioned document embedding boundary without credential persistence."""

    @property
    def model_version(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> Sequence[Sequence[float]]: ...


class KnowledgeEmbeddingIndexResult(ContractModel):
    """Secret-free result of one review-gated embedding index operation."""

    document_id: str = Field(min_length=1)
    embedding_model_version: str = Field(min_length=1)
    chunk_count: int = Field(ge=1)
    reused: bool


@dataclass(frozen=True, slots=True)
class _ChunkSnapshot:
    chunk_id: str
    text_object_uri: str
    text_sha256: str
    text_size_bytes: int
    text_content_type: str


class KnowledgeEmbeddingIndexService:
    """Populate the pgvector-backed column only for approved immutable chunks."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        text_loader: KnowledgeChunkTextLoader,
        embedding_provider: DocumentEmbeddingProvider,
    ) -> None:
        model_version = embedding_provider.model_version.strip()
        if not model_version:
            raise ValueError("embedding provider model_version must not be blank")
        if embedding_provider.dimensions != KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding provider must output {KNOWLEDGE_EMBEDDING_DIMENSIONS} dimensions"
            )
        self._session_factory = session_factory
        self._text_loader = text_loader
        self._embedding_provider = embedding_provider
        self._model_version = model_version

    def index_document(
        self,
        principal: AuthPrincipal,
        document_id: str,
    ) -> KnowledgeEmbeddingIndexResult:
        if principal.role is not UserRole.ADMIN:
            raise PermissionError("only an administrator may index knowledge embeddings")
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be blank")

        snapshots, reused = self._load_snapshots(normalized_document_id)
        if reused:
            return self._result(normalized_document_id, len(snapshots), reused=True)
        texts = tuple(self._load_verified_text(snapshot) for snapshot in snapshots)
        vectors = _validated_document_vectors(
            self._embedding_provider.embed_documents(texts),
            expected_count=len(snapshots),
        )
        concurrent_reuse = self._publish_vectors(
            normalized_document_id,
            snapshots=snapshots,
            vectors=vectors,
        )
        return self._result(
            normalized_document_id,
            len(snapshots),
            reused=concurrent_reuse,
        )

    def _load_snapshots(
        self,
        document_id: str,
    ) -> tuple[tuple[_ChunkSnapshot, ...], bool]:
        with session_scope(self._session_factory) as session:
            document = session.scalar(
                select(KnowledgeDocument).where(KnowledgeDocument.id == document_id)
            )
            if document is None:
                raise LookupError("knowledge document was not found")
            if document.review_status != KnowledgeReviewStatus.APPROVED.value:
                raise ValueError("only approved knowledge documents may be embedded")
            rows = tuple(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id == document_id)
                    .order_by(KnowledgeChunk.chunk_index)
                ).all()
            )
            if not rows:
                raise ValueError("approved knowledge document has no indexed chunks")
            reused = _embedding_state(rows, self._model_version)
            snapshots = tuple(
                _ChunkSnapshot(
                    chunk_id=row.id,
                    text_object_uri=row.text_object_uri,
                    text_sha256=row.text_sha256,
                    text_size_bytes=row.text_size_bytes,
                    text_content_type=row.text_content_type,
                )
                for row in rows
            )
        return snapshots, reused

    def _load_verified_text(self, snapshot: _ChunkSnapshot) -> str:
        text = self._text_loader.load_verified_text(
            object_uri=snapshot.text_object_uri,
            expected_sha256=snapshot.text_sha256,
            size_bytes=snapshot.text_size_bytes,
            content_type=snapshot.text_content_type,
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("knowledge chunk text must be nonblank UTF-8 text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != snapshot.text_sha256:
            raise ValueError("knowledge chunk text SHA-256 does not match its manifest")
        return text.strip()

    def _publish_vectors(
        self,
        document_id: str,
        *,
        snapshots: tuple[_ChunkSnapshot, ...],
        vectors: tuple[tuple[float, ...], ...],
    ) -> bool:
        with session_scope(self._session_factory) as session:
            document = session.scalar(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.id == document_id)
                .with_for_update()
            )
            if (
                document is None
                or document.review_status != KnowledgeReviewStatus.APPROVED.value
            ):
                raise ValueError("knowledge document approval changed during embedding")
            rows = tuple(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id == document_id)
                    .order_by(KnowledgeChunk.chunk_index)
                    .with_for_update()
                ).all()
            )
            expected_identity = tuple(
                (item.chunk_id, item.text_sha256) for item in snapshots
            )
            actual_identity = tuple((row.id, row.text_sha256) for row in rows)
            if actual_identity != expected_identity:
                raise ValueError("knowledge chunks changed during embedding")
            if _embedding_state(rows, self._model_version):
                return True
            for row, vector in zip(rows, vectors, strict=True):
                row.embedding_model_version = self._model_version
                row.embedding = list(vector)
        return False

    def _result(
        self,
        document_id: str,
        chunk_count: int,
        *,
        reused: bool,
    ) -> KnowledgeEmbeddingIndexResult:
        return KnowledgeEmbeddingIndexResult(
            document_id=document_id,
            embedding_model_version=self._model_version,
            chunk_count=chunk_count,
            reused=reused,
        )


def _embedding_state(rows: Sequence[KnowledgeChunk], model_version: str) -> bool:
    empty = [
        row.embedding is None and row.embedding_model_version is None for row in rows
    ]
    if all(empty):
        return False
    if any(empty):
        raise ValueError("knowledge embedding index is partially populated")
    if any(row.embedding_model_version != model_version for row in rows):
        raise ValueError("knowledge embeddings already use a different model version")
    for row in rows:
        _validated_vector(row.embedding, label=f"knowledge chunk {row.id}")
    return True


def _validated_document_vectors(
    values: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    vectors = tuple(values)
    if len(vectors) != expected_count:
        raise ValueError("embedding count does not match knowledge chunk count")
    return tuple(
        _validated_vector(vector, label=f"embedding output {index}")
        for index, vector in enumerate(vectors)
    )


def _validated_vector(
    value: Sequence[float] | None,
    *,
    label: str,
) -> tuple[float, ...]:
    if value is None or len(value) != KNOWLEDGE_EMBEDDING_DIMENSIONS:
        raise ValueError(f"{label} must contain {KNOWLEDGE_EMBEDDING_DIMENSIONS} values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{label} must contain only finite numbers")
    vector = tuple(float(item) for item in value)
    if math.isclose(sum(item * item for item in vector), 0.0, abs_tol=1e-24):
        raise ValueError(f"{label} must not be a zero vector")
    return vector


__all__ = [
    "KNOWLEDGE_EMBEDDING_DIMENSIONS",
    "DocumentEmbeddingProvider",
    "KnowledgeEmbeddingIndexResult",
    "KnowledgeEmbeddingIndexService",
]
