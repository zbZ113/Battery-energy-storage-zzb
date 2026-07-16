"""Authenticated FastAPI adapter for reviewed knowledge document workflows."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.knowledge.documents import (
    MAX_KNOWLEDGE_DOCUMENT_BYTES,
    KnowledgeChunkRecord,
    KnowledgeDocumentConflictError,
    KnowledgeDocumentIntegrityError,
    KnowledgeDocumentRecord,
    KnowledgeDocumentService,
    KnowledgeSourceConflictError,
)
from quanxin_life.knowledge.parsers import KnowledgeParsingError

_MAX_BASE64_CHARACTERS = ((MAX_KNOWLEDGE_DOCUMENT_BYTES + 2) // 3) * 4


class UploadKnowledgeDocumentRequest(ContractModel):
    project_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    source_uri: str = Field(min_length=1, max_length=2_000)
    license_name: str = Field(min_length=1, max_length=200)
    document_version: str = Field(min_length=1, max_length=100)
    content_type: str = Field(min_length=1, max_length=200)
    content_base64: str = Field(min_length=1, max_length=_MAX_BASE64_CHARACTERS)


@dataclass(frozen=True, slots=True)
class KnowledgeHttpAdapter:
    router: APIRouter


def create_knowledge_http_adapter(
    service: KnowledgeDocumentService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> KnowledgeHttpAdapter:
    """Build upload, review, indexing and project-scoped listing routes."""

    router = APIRouter(prefix="/v1/knowledge/documents", tags=["knowledge"])
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )
    administrator_dependency = auth_adapter.require_roles({UserRole.ADMIN})
    trusted_origin = Depends(auth_adapter.require_trusted_origin)
    operator = Depends(operator_dependency)
    administrator = Depends(administrator_dependency)
    ready_user = Depends(auth_adapter.require_ready_user)
    idempotency_header = Header(alias="Idempotency-Key")

    @router.post(
        "",
        response_model=KnowledgeDocumentRecord,
        status_code=201,
        dependencies=[trusted_origin],
    )
    def upload_document(
        payload: UploadKnowledgeDocumentRequest,
        principal: AuthPrincipal = operator,
        idempotency_key: str = idempotency_header,
    ) -> Any:
        try:
            raw_content = base64.b64decode(payload.content_base64, validate=True)
            return service.upload_document(
                principal,
                project_id=payload.project_id,
                title=payload.title,
                source_uri=payload.source_uri,
                license_name=payload.license_name,
                document_version=payload.document_version,
                payload=raw_content,
                content_type=payload.content_type,
                idempotency_key=idempotency_key,
                now=datetime.now(UTC),
            )
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="invalid_knowledge_document"
            ) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except KnowledgeSourceConflictError as exc:
            raise HTTPException(status_code=409, detail="duplicate_source") from exc
        except KnowledgeDocumentConflictError as exc:
            raise HTTPException(status_code=409, detail="idempotency_conflict") from exc

    @router.get("", response_model=list[KnowledgeDocumentRecord])
    def list_documents(
        project_id: Annotated[str, Query(min_length=1, max_length=64)],
        principal: AuthPrincipal = ready_user,
    ) -> Any:
        try:
            return list(service.list_documents(principal, project_id=project_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_project_filter") from exc

    @router.post(
        "/{document_id}/approve",
        response_model=KnowledgeDocumentRecord,
        dependencies=[trusted_origin],
    )
    def approve_document(
        document_id: str,
        principal: AuthPrincipal = administrator,
    ) -> Any:
        return _review_document(
            lambda: service.approve_document(
                principal, document_id, now=datetime.now(UTC)
            )
        )

    @router.post(
        "/{document_id}/reject",
        response_model=KnowledgeDocumentRecord,
        dependencies=[trusted_origin],
    )
    def reject_document(
        document_id: str,
        principal: AuthPrincipal = administrator,
    ) -> Any:
        return _review_document(
            lambda: service.reject_document(
                principal, document_id, now=datetime.now(UTC)
            )
        )

    @router.post(
        "/{document_id}/index",
        response_model=list[KnowledgeChunkRecord],
        dependencies=[trusted_origin],
    )
    def index_document(
        document_id: str,
        principal: AuthPrincipal = administrator,
    ) -> Any:
        try:
            return list(
                service.index_document(
                    principal,
                    document_id,
                    now=datetime.now(UTC),
                )
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="review_not_allowed") from exc
        except LookupError as exc:
            raise HTTPException(
                status_code=404, detail="knowledge_document_not_found"
            ) from exc
        except KnowledgeParsingError as exc:
            raise HTTPException(
                status_code=422, detail="knowledge_document_unprocessable"
            ) from exc
        except KnowledgeDocumentIntegrityError as exc:
            raise HTTPException(
                status_code=500, detail="knowledge_document_integrity_failure"
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409, detail="knowledge_document_state_conflict"
            ) from exc

    return KnowledgeHttpAdapter(router=router)


def _review_document(
    operation: Callable[[], KnowledgeDocumentRecord],
) -> KnowledgeDocumentRecord:
    try:
        return operation()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="review_not_allowed") from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail="knowledge_document_not_found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail="knowledge_document_state_conflict"
        ) from exc


__all__ = ["KnowledgeHttpAdapter", "create_knowledge_http_adapter"]
