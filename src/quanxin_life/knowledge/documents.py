"""Project-scoped knowledge upload, independent review and deterministic indexing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import Field, field_validator
from sqlalchemy import select

from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import KnowledgeReviewStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.infrastructure.object_store import StoredObjectRef
from quanxin_life.knowledge.parsers import parse_knowledge_document
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import KnowledgeChunk, KnowledgeDocument, Project

MAX_KNOWLEDGE_DOCUMENT_BYTES = 25 * 1024 * 1024
_ALLOWED_MEDIA_TYPES = frozenset({"application/pdf", "text/markdown", "text/plain"})


class KnowledgeObjectStore(Protocol):
    def put_bytes(
        self,
        *,
        namespace: str,
        payload: bytes,
        expected_sha256: str,
        content_type: str,
    ) -> StoredObjectRef: ...

    def get_bytes(self, stored: StoredObjectRef) -> bytes: ...


class KnowledgeDocumentRecord(ContractModel):
    document_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    created_by_user_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_name: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    review_status: KnowledgeReviewStatus
    reviewer_user_id: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime

    @field_validator("reviewed_at", "created_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


class KnowledgeChunkRecord(ContractModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_label: str | None = None
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeDocumentService:
    """Persist immutable sources and index only independently approved content."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        object_store: KnowledgeObjectStore,
    ) -> None:
        self._session_factory = session_factory
        self._object_store = object_store

    def upload_document(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        title: str,
        source_uri: str,
        license_name: str,
        document_version: str,
        payload: bytes,
        content_type: str,
        now: datetime,
    ) -> KnowledgeDocumentRecord:
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise PermissionError("role is not allowed to upload knowledge documents")
        timestamp = _utc(now)
        normalized_title = _bounded_text(title, "title", 500)
        normalized_source = _source_uri(source_uri)
        normalized_license = _bounded_text(license_name, "license_name", 200)
        if normalized_license.lower() in {"unknown", "unknown-license", "待确认"}:
            raise ValueError("knowledge document license must be reviewed before upload")
        normalized_version = _bounded_text(document_version, "document_version", 100)
        normalized_content_type = _content_type(content_type)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("knowledge document payload must be nonempty bytes")
        if len(payload) > MAX_KNOWLEDGE_DOCUMENT_BYTES:
            raise ValueError("knowledge document exceeds the 25 MB limit")
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == project_id
                )
            )
            if project is None:
                raise LookupError("project was not found")
            digest = hashlib.sha256(payload).hexdigest()
            existing = session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.source_sha256 == digest
                )
            )
            if existing is not None:
                raise ValueError("knowledge document content is already registered")
            stored = self._object_store.put_bytes(
                namespace="knowledge/documents",
                payload=payload,
                expected_sha256=digest,
                content_type=normalized_content_type,
            )
            row = KnowledgeDocument(
                id=str(uuid4()),
                project_id=project.id,
                created_by_user_id=principal.user_id,
                title=normalized_title,
                source_uri=normalized_source,
                source_sha256=digest,
                license_name=normalized_license,
                document_version=normalized_version,
                review_status=KnowledgeReviewStatus.PENDING.value,
                reviewer_user_id=None,
                reviewed_at=None,
                object_uri=stored.uri,
                object_size_bytes=stored.size_bytes,
                object_content_type=stored.content_type,
                created_at=timestamp,
            )
            session.add(row)
            session.flush()
            return _document_record(row)

    def list_documents(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
    ) -> tuple[KnowledgeDocumentRecord, ...]:
        """List documents only when the caller can see their parent project."""

        normalized_project_id = _bounded_text(project_id, "project_id", 64)
        visible_projects = ProjectService.visible_projects_statement(principal)
        visible_project_ids = visible_projects.with_only_columns(Project.id)
        with session_scope(self._session_factory) as session:
            rows = tuple(
                session.scalars(
                    select(KnowledgeDocument)
                    .where(
                        KnowledgeDocument.project_id == normalized_project_id,
                        KnowledgeDocument.project_id.in_(visible_project_ids),
                    )
                    .order_by(KnowledgeDocument.created_at, KnowledgeDocument.id)
                ).all()
            )
        return tuple(_document_record(row) for row in rows)

    def approve_document(
        self,
        principal: AuthPrincipal,
        document_id: str,
        *,
        now: datetime,
    ) -> KnowledgeDocumentRecord:
        if principal.role is not UserRole.ADMIN:
            raise PermissionError("only an administrator may approve knowledge documents")
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.id == document_id)
                .with_for_update()
            )
            if row is None:
                raise LookupError("knowledge document was not found")
            if row.created_by_user_id == principal.user_id:
                raise PermissionError("an uploader cannot approve their own document")
            if row.review_status == KnowledgeReviewStatus.APPROVED.value:
                return _document_record(row)
            if row.review_status != KnowledgeReviewStatus.PENDING.value:
                raise ValueError("knowledge document is not pending review")
            row.review_status = KnowledgeReviewStatus.APPROVED.value
            row.reviewer_user_id = principal.user_id
            row.reviewed_at = timestamp
            session.flush()
            return _document_record(row)

    def reject_document(
        self,
        principal: AuthPrincipal,
        document_id: str,
        *,
        now: datetime,
    ) -> KnowledgeDocumentRecord:
        """Reject a pending source without allowing the uploader to review it."""

        if principal.role is not UserRole.ADMIN:
            raise PermissionError("only an administrator may reject knowledge documents")
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.id == document_id)
                .with_for_update()
            )
            if row is None:
                raise LookupError("knowledge document was not found")
            if row.created_by_user_id == principal.user_id:
                raise PermissionError("an uploader cannot review their own document")
            if row.review_status == KnowledgeReviewStatus.REJECTED.value:
                return _document_record(row)
            if row.review_status != KnowledgeReviewStatus.PENDING.value:
                raise ValueError("knowledge document is not pending review")
            row.review_status = KnowledgeReviewStatus.REJECTED.value
            row.reviewer_user_id = principal.user_id
            row.reviewed_at = timestamp
            session.flush()
            return _document_record(row)

    def index_document(
        self,
        principal: AuthPrincipal,
        document_id: str,
        *,
        now: datetime,
    ) -> tuple[KnowledgeChunkRecord, ...]:
        if principal.role is not UserRole.ADMIN:
            raise PermissionError("only an administrator may index knowledge documents")
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.id == document_id)
                .with_for_update()
            )
            if row is None:
                raise LookupError("knowledge document was not found")
            if row.review_status != KnowledgeReviewStatus.APPROVED.value:
                raise ValueError("only approved knowledge documents may be indexed")
            existing = tuple(
                session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.document_id == row.id)
                    .order_by(KnowledgeChunk.chunk_index)
                ).all()
            )
            if existing:
                return tuple(_chunk_record(item) for item in existing)
            source_ref = StoredObjectRef(
                uri=row.object_uri,
                sha256=row.source_sha256,
                size_bytes=row.object_size_bytes,
                content_type=row.object_content_type,
            )
            payload = self._object_store.get_bytes(source_ref)
            parsed = parse_knowledge_document(
                payload,
                content_type=row.object_content_type,
            )
            persisted: list[KnowledgeChunk] = []
            for index, chunk in enumerate(parsed):
                text_payload = chunk.text.encode("utf-8")
                digest = hashlib.sha256(text_payload).hexdigest()
                stored = self._object_store.put_bytes(
                    namespace="knowledge/chunks",
                    payload=text_payload,
                    expected_sha256=digest,
                    content_type="text/plain; charset=utf-8",
                )
                chunk_row = KnowledgeChunk(
                    id=str(uuid4()),
                    document_id=row.id,
                    chunk_index=index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text_object_uri=stored.uri,
                    text_sha256=stored.sha256,
                    text_size_bytes=stored.size_bytes,
                    text_content_type=stored.content_type,
                    section_label=chunk.section_label,
                    embedding_model_version=None,
                    embedding=None,
                    created_at=timestamp,
                )
                session.add(chunk_row)
                persisted.append(chunk_row)
            session.flush()
            return tuple(_chunk_record(item) for item in persisted)


def _document_record(row: KnowledgeDocument) -> KnowledgeDocumentRecord:
    return KnowledgeDocumentRecord(
        document_id=row.id,
        project_id=row.project_id,
        created_by_user_id=row.created_by_user_id,
        title=row.title,
        source_uri=row.source_uri,
        source_sha256=row.source_sha256,
        license_name=row.license_name,
        document_version=row.document_version,
        review_status=KnowledgeReviewStatus(row.review_status),
        reviewer_user_id=row.reviewer_user_id,
        reviewed_at=row.reviewed_at,
        created_at=row.created_at,
    )


def _chunk_record(row: KnowledgeChunk) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        chunk_id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        page_start=row.page_start,
        page_end=row.page_end,
        section_label=row.section_label,
        text_sha256=row.text_sha256,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("knowledge timestamp must include a timezone")
    return value.astimezone(UTC)


def _bounded_text(value: str, field_name: str, maximum: int) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{field_name} must contain between 1 and {maximum} characters")
    return normalized


def _source_uri(value: str) -> str:
    normalized = _bounded_text(value, "source_uri", 2_000)
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("knowledge source_uri must be an HTTP(S) URL")
    return normalized


def _content_type(value: str) -> str:
    normalized = _bounded_text(value, "content_type", 200).lower()
    media_type = normalized.split(";", maxsplit=1)[0].strip()
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise ValueError("only PDF, Markdown and plain text knowledge files are allowed")
    return normalized


__all__ = [
    "MAX_KNOWLEDGE_DOCUMENT_BYTES",
    "KnowledgeChunkRecord",
    "KnowledgeDocumentRecord",
    "KnowledgeDocumentService",
    "KnowledgeObjectStore",
]
