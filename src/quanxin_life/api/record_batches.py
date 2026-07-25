"""Authenticated HTTP adapter for project-scoped canonical batch bindings."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64DecodeError
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.ingestion import (
    MAX_CANONICAL_CSV_BYTES,
    CanonicalCsvBatchRegistration,
)
from quanxin_life.application.record_batch_bindings import (
    RecordBatchBindingAccessError,
    RecordBatchBindingNotFoundError,
    RecordBatchBindingRecord,
    RecordBatchBindingService,
    RecordBatchBindingStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel


class CanonicalCsvUploadRequest(ContractModel):
    """JSON-safe transport envelope for canonical CSV bytes and metadata."""

    payload_base64: str = Field(min_length=1)
    registration: CanonicalCsvBatchRegistration

    def decoded_payload(self) -> bytes:
        try:
            payload = b64decode(self.payload_base64, validate=True)
        except (Base64DecodeError, ValueError) as exc:
            raise ValueError("payload_base64 must be valid base64") from exc
        if not payload:
            raise ValueError("decoded canonical CSV payload must not be empty")
        if len(payload) > MAX_CANONICAL_CSV_BYTES:
            raise ValueError(
                "decoded canonical CSV payload exceeds the configured size limit"
            )
        return payload


@dataclass(frozen=True, slots=True)
class RecordBatchHttpAdapter:
    router: APIRouter


def create_record_batch_http_adapter(
    service: RecordBatchBindingService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> RecordBatchHttpAdapter:
    """Build the only product upload route that can create batch bindings."""

    router = APIRouter(prefix="/v1/datasets", tags=["record-batches"])
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )
    operator_principal = Depends(operator_dependency)

    @router.post(
        "/{dataset_id}/batches/canonical-csv",
        response_model=RecordBatchBindingRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def register_canonical_csv(
        dataset_id: str,
        payload: CanonicalCsvUploadRequest,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            return service.register_canonical_csv(
                principal,
                dataset_id,
                payload.decoded_payload(),
                payload.registration,
                now=datetime.now(UTC),
            )
        except RecordBatchBindingAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except RecordBatchBindingNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="record_batch_scope_not_found",
            ) from exc
        except RecordBatchBindingStateError as exc:
            raise HTTPException(
                status_code=409,
                detail="record_batch_state_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_record_batch") from exc

    return RecordBatchHttpAdapter(router=router)


__all__ = [
    "CanonicalCsvUploadRequest",
    "RecordBatchHttpAdapter",
    "create_record_batch_http_adapter",
]
