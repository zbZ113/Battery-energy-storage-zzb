"""Secret-free FastAPI transport for authenticated Feishu callbacks."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from quanxin_life.integrations.feishu import (
    FeishuEventError,
    FeishuEventProcessor,
    FeishuPayloadDecryptionUnavailable,
    FeishuReceiptClaimStatus,
    FeishuSignatureError,
)

MAX_FEISHU_CALLBACK_BYTES = 1024 * 1024


class FeishuEventRouteStatus(StrEnum):
    """Durable idempotent enqueue acknowledgement from the business router."""

    ENQUEUED = "ENQUEUED"
    ALREADY_ENQUEUED = "ALREADY_ENQUEUED"


class FeishuEventRouter(Protocol):
    """Idempotently enqueue by event ID without receiving the raw payload."""

    def route(
        self, *, event_id: str, event_type: str, claim_token: str
    ) -> FeishuEventRouteStatus:
        """Persist one task or confirm that this event ID was already enqueued."""
        ...


@dataclass(frozen=True, slots=True)
class FeishuHttpAdapter:
    """Optional unauthenticated-by-session router secured by Feishu signatures."""

    router: APIRouter


def create_feishu_http_adapter(
    processor: FeishuEventProcessor,
    *,
    router: FeishuEventRouter | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> FeishuHttpAdapter:
    """Create the callback endpoint around the reviewed event processor."""

    clock = now_factory or (lambda: datetime.now(UTC))
    api_router = APIRouter(prefix="/v1/integrations/feishu", tags=["feishu"])

    @api_router.post("/events")
    async def receive_event(request: Request) -> dict[str, str]:
        body = await _read_limited_body(request)
        try:
            outcome = processor.handle(
                headers=request.headers,
                body=body,
                now=clock(),
            )
        except FeishuSignatureError as exc:
            raise HTTPException(
                status_code=401,
                detail="feishu_callback_authentication_failed",
            ) from exc
        except FeishuPayloadDecryptionUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="feishu_callback_decryption_unavailable",
            ) from exc
        except FeishuEventError as exc:
            raise HTTPException(status_code=422, detail="invalid_feishu_callback") from exc

        if outcome.kind == "url_verification":
            return outcome.response
        if outcome.duplicate:
            if outcome.receipt_status is FeishuReceiptClaimStatus.PROCESSED:
                return {}
            if outcome.receipt_status is FeishuReceiptClaimStatus.IN_PROGRESS:
                raise HTTPException(status_code=503, detail="feishu_event_in_progress")
            raise HTTPException(status_code=500, detail="invalid_feishu_event_state")

        event_id = outcome.event_id
        event_type = outcome.event_type
        claim_token = outcome.claim_token
        if event_id is None or event_type is None or claim_token is None:
            raise HTTPException(status_code=500, detail="invalid_feishu_event_state")

        if router is None:
            _mark_failed_safely(
                processor,
                event_id=event_id,
                claim_token=claim_token,
                failed_at=clock(),
            )
            raise HTTPException(status_code=503, detail="feishu_event_router_unavailable")

        try:
            route_status = await run_in_threadpool(
                router.route,
                event_id=event_id,
                event_type=event_type,
                claim_token=claim_token,
            )
            if not isinstance(route_status, FeishuEventRouteStatus):
                raise RuntimeError("Feishu event router did not confirm durable enqueue")
            processor.mark_processed(
                event_id=event_id,
                claim_token=claim_token,
                processed_at=clock(),
            )
        except Exception as exc:
            _mark_failed_safely(
                processor,
                event_id=event_id,
                claim_token=claim_token,
                failed_at=clock(),
            )
            raise HTTPException(status_code=503, detail="feishu_event_routing_failed") from exc
        return {}

    return FeishuHttpAdapter(router=api_router)


async def _read_limited_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = 0
        if declared_length > MAX_FEISHU_CALLBACK_BYTES:
            raise HTTPException(status_code=413, detail="feishu_callback_too_large")
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > MAX_FEISHU_CALLBACK_BYTES:
            raise HTTPException(status_code=413, detail="feishu_callback_too_large")
        payload.extend(chunk)
    return bytes(payload)


def _mark_failed_safely(
    processor: FeishuEventProcessor,
    *,
    event_id: str,
    claim_token: str,
    failed_at: datetime,
) -> None:
    with suppress(Exception):
        processor.mark_failed(
            event_id=event_id,
            claim_token=claim_token,
            failed_at=failed_at,
        )


__all__ = [
    "MAX_FEISHU_CALLBACK_BYTES",
    "FeishuEventRouteStatus",
    "FeishuEventRouter",
    "FeishuHttpAdapter",
    "create_feishu_http_adapter",
]
