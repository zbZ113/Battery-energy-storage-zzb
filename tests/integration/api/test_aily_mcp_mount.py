from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.service import create_available_tool_invocation_service


class _MountedApp:
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        del receive
        assert scope["root_path"] == "/v1/aily/mcp/secret"
        body = b"mounted"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(body)).encode("ascii"))],
            }
        )
        await send({"type": "http.response.body", "body": body})


@dataclass
class _Adapter:
    mount_path: str = "/v1/aily/mcp/secret"
    asgi_app: Any = field(default_factory=_MountedApp)
    entered: bool = False
    exited: bool = False

    @asynccontextmanager
    async def lifespan(self, _: Any) -> AsyncIterator[None]:
        self.entered = True
        try:
            yield
        finally:
            self.exited = True


def test_fastapi_mounts_the_aily_mcp_app_and_runs_its_parent_lifespan() -> None:
    adapter = _Adapter()
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        aily_mcp_adapter=adapter,
    )

    with TestClient(app) as client:
        response = client.get(f"{adapter.mount_path}/")
        assert adapter.entered is True
        assert adapter.exited is False

    assert response.status_code == 200
    assert response.text == "mounted"
    assert adapter.exited is True
