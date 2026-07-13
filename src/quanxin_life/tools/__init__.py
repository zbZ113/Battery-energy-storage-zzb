"""Shared typed tool registry for all service and agent entry points."""

from quanxin_life.tools.mcp_adapter import (
    McpAdapterError,
    McpRequestValidationError,
    McpSdkUnavailableError,
    McpToolAdapter,
    McpToolCallRequest,
    load_optional_mcp_sdk,
)
from quanxin_life.tools.registry import (
    DuplicateToolError,
    RegisteredTool,
    StandardToolName,
    ToolAuthorizationError,
    ToolContractError,
    ToolDefinition,
    ToolExecutionError,
    ToolInputValidationError,
    ToolRegistry,
    ToolRegistryError,
    ToolSchema,
    UnknownToolError,
)

__all__ = [
    "DuplicateToolError",
    "McpAdapterError",
    "McpRequestValidationError",
    "McpSdkUnavailableError",
    "McpToolAdapter",
    "McpToolCallRequest",
    "RegisteredTool",
    "StandardToolName",
    "ToolAuthorizationError",
    "ToolContractError",
    "ToolDefinition",
    "ToolExecutionError",
    "ToolInputValidationError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSchema",
    "UnknownToolError",
    "load_optional_mcp_sdk",
]
