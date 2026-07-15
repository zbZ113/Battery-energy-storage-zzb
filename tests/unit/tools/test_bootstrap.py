from __future__ import annotations

from quanxin_life.tools.registry import StandardToolName


def test_available_tool_registry_registers_each_implemented_standard_tool_once() -> None:
    from quanxin_life.tools.bootstrap import create_available_tool_registry

    registry = create_available_tool_registry()

    assert [schema.tool_name for schema in registry.list_schemas()] == [
        StandardToolName.AUDIT_DATASET_SPLIT,
        StandardToolName.CHECK_OPERATING_CONDITION,
        StandardToolName.VALIDATE_BATTERY_DATA,
    ]
