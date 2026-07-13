"""Shared typed tool registry for all service and agent entry points."""

from quanxin_life.tools.batch_decision import (
    BATCH_DECISION_TOOL_VERSION,
    BatchDecisionToolInput,
    execute_batch_decision_tool,
    register_batch_decision_tool,
)
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
    ValidateBatteryDataToolInput,
    execute_validate_battery_data_tool,
    register_validate_battery_data_tool,
)
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
from quanxin_life.tools.split_audit import (
    SPLIT_AUDIT_MODEL_VERSION,
    SPLIT_AUDIT_TOOL_VERSION,
    AuditDatasetSplitToolInput,
    execute_audit_dataset_split_tool,
    register_audit_dataset_split_tool,
)

__all__ = [
    "BATCH_DECISION_TOOL_VERSION",
    "DATA_QUALITY_MODEL_VERSION",
    "DATA_QUALITY_TOOL_VERSION",
    "SPLIT_AUDIT_MODEL_VERSION",
    "SPLIT_AUDIT_TOOL_VERSION",
    "AuditDatasetSplitToolInput",
    "BatchDecisionToolInput",
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
    "ValidateBatteryDataToolInput",
    "execute_audit_dataset_split_tool",
    "execute_batch_decision_tool",
    "execute_validate_battery_data_tool",
    "load_optional_mcp_sdk",
    "register_audit_dataset_split_tool",
    "register_batch_decision_tool",
    "register_validate_battery_data_tool",
]
