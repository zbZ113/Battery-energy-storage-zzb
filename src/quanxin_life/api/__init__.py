"""Transport-neutral service contracts and lazy optional FastAPI adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "FastApiDependencyUnavailable": "app",
    "ToolInvocation": "service",
    "ToolInvocationService": "service",
    "create_available_tool_invocation_service": "service",
    "create_fastapi_app": "app",
    "FeishuCallbackSecurityMode": "feishu_runner",
    "create_feishu_callback_app": "feishu_runner",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
