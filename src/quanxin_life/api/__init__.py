"""Transport-neutral service contracts and optional FastAPI adapters."""

from quanxin_life.api.app import FastApiDependencyUnavailable, create_fastapi_app
from quanxin_life.api.service import (
    ToolInvocation,
    ToolInvocationService,
    create_available_tool_invocation_service,
)

__all__ = [
    "FastApiDependencyUnavailable",
    "ToolInvocation",
    "ToolInvocationService",
    "create_available_tool_invocation_service",
    "create_fastapi_app",
]
