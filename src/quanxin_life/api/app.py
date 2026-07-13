"""Optional FastAPI transport adapter for the shared tool invocation service.

Importing this module never imports FastAPI.  The factory fails explicitly when
the optional API dependency group is absent, while the domain service remains
usable by MCP, command-line and test callers.
"""

from __future__ import annotations

import importlib
from typing import Any

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.tools import StandardToolName, ToolRegistryError


class FastApiDependencyUnavailable(RuntimeError):
    """Raised when an API host is requested without the optional dependency group."""


def create_fastapi_app(service: ToolInvocationService) -> Any:
    """Create the HTTP adapter without duplicating domain-tool execution logic."""
    try:
        fastapi_module = importlib.import_module("fastapi")
    except ModuleNotFoundError as exc:  # pragma: no cover
        message = "FastAPI support requires installing the 'quanxin-life[api]' extra"
        raise FastApiDependencyUnavailable(message) from exc

    app: Any = fastapi_module.FastAPI(title="泉芯智寿 Tool API", version="v1")

    @app.get("/health")  # type: ignore[untyped-decorator]
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "quanxin-life-tool-api"}

    @app.get("/v1/tools")  # type: ignore[untyped-decorator]
    async def list_tools() -> list[dict[str, Any]]:
        return [schema.model_dump(mode="json") for schema in service.registry.list_schemas()]

    @app.post("/v1/tools/{tool_name}")  # type: ignore[untyped-decorator]
    async def invoke_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            invocation = ToolInvocation(
                tool_name=StandardToolName(tool_name), input_value=payload
            )
            result = service.invoke(invocation)
        except (TypeError, ValueError, ToolRegistryError) as exc:
            raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    return app
