"""Transport-neutral service contracts and optional FastAPI adapters."""

from quanxin_life.api.app import FastApiDependencyUnavailable, create_fastapi_app
from quanxin_life.api.service import ToolInvocation, ToolInvocationService

__all__ = [
    "FastApiDependencyUnavailable",
    "ToolInvocation",
    "ToolInvocationService",
    "create_fastapi_app",
]
