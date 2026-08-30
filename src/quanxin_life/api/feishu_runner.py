"""Standalone Feishu callback runner assembly.

The runner owns only transport composition and health reporting. Event routing,
receipt persistence, and downstream task execution remain injected through the
existing Feishu processor/router ports.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from fastapi import FastAPI

from quanxin_life.api.feishu import (
    FeishuEventRouter,
    FeishuHttpAdapter,
    FeishuSanitizedEventRouter,
    create_feishu_http_adapter,
)
from quanxin_life.integrations.feishu import (
    FeishuEventProcessor,
    FeishuPayloadDecryptor,
)


class FeishuCallbackSecurityMode(StrEnum):
    """Explicit callback encryption posture exposed by the health endpoint."""

    LOCAL_PLAINTEXT = "LOCAL_PLAINTEXT"
    LOCAL_ENCRYPTED = "LOCAL_ENCRYPTED"
    PRODUCTION_REVIEWED = "PRODUCTION_REVIEWED"


def create_feishu_callback_app(
    processor: FeishuEventProcessor,
    *,
    router: FeishuEventRouter | FeishuSanitizedEventRouter,
    security_mode: FeishuCallbackSecurityMode,
    reviewed_decryptor: FeishuPayloadDecryptor | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Create only the callback surface, never the broad foundation API.

    ``LOCAL_PLAINTEXT`` is an explicit loopback-only posture for local tests.
    Production assembly must inject a reviewed decryptor before it can start.
    The processor must be constructed with that same decryptor by the caller.
    """

    if security_mode in {
        FeishuCallbackSecurityMode.LOCAL_ENCRYPTED,
        FeishuCallbackSecurityMode.PRODUCTION_REVIEWED,
    }:
        if reviewed_decryptor is None:
            raise ValueError(
                "encrypted Feishu callback runner requires a reviewed decryptor"
            )
        if not processor.decryptor_configured:
            raise ValueError(
                "encrypted Feishu callback runner requires a processor decryptor"
            )
    elif reviewed_decryptor is not None:
        raise ValueError("a reviewed decryptor is not allowed in local plaintext mode")

    adapter: FeishuHttpAdapter = create_feishu_http_adapter(
        processor,
        router=router,
        now_factory=now_factory,
    )
    app = FastAPI(
        title="Hiro Feishu Callback Runner",
        version="phase1-v1",
        docs_url=None,
        redoc_url=None,
    )
    app.include_router(adapter.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "feishu-callback-runner",
            "security_mode": security_mode.value,
        }

    return app


__all__ = [
    "FeishuCallbackSecurityMode",
    "create_feishu_callback_app",
]
