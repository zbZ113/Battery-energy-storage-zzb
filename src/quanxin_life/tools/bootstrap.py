"""Single assembly point for every currently implemented shared domain tool."""

from __future__ import annotations

from quanxin_life.tools.data_quality import register_validate_battery_data_tool
from quanxin_life.tools.physics_check import register_check_operating_condition_tool
from quanxin_life.tools.registry import ToolRegistry
from quanxin_life.tools.split_audit import register_audit_dataset_split_tool


def create_available_tool_registry() -> ToolRegistry:
    """Create one registry containing each implemented standard tool exactly once.

    The function deliberately registers only working implementations.  Standard
    names that are still under construction remain absent so callers receive an
    explicit ``UnknownToolError`` rather than a placeholder response.
    """
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_audit_dataset_split_tool(registry)
    register_check_operating_condition_tool(registry)
    return registry
