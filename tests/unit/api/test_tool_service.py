from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry


class _Input(ContractModel):
    batch_id: str = Field(min_length=1)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    def executor(value: _Input) -> ToolResult:
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
            tool_version="api-service-test-v1",
            model_version="api-service-model-v1",
            data_version="api-service-data-v1",
            feature_version="api-service-feature-v1",
            input_hash=sha256_canonical(value.model_dump(mode="json")),
            values={"validated_batch": value.batch_id},
            provenance=[
                ProvenanceRecord(
                    source_id="api-service-fixture",
                    source_kind=SourceKind.OBSERVED,
                    uri="test://api-service/fixture",
                    sha256=sha256_canonical({"fixture": "api-service"}),
                    description="Tool service test fixture",
                    created_at=datetime(2026, 7, 13, tzinfo=UTC),
                )
            ],
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="api-service-test-v1",
            input_model=_Input,
            executor=executor,
        )
    )
    return registry


def test_service_delegates_external_invocation_to_shared_registry() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService

    service = ToolInvocationService(registry=_registry())
    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": "batch-A"},
        )
    )

    assert result.tool_name == StandardToolName.VALIDATE_BATTERY_DATA.value
    assert result.values == {"validated_batch": "batch-A"}
    assert result.input_hash == sha256_canonical({"batch_id": "batch-A"})


def test_available_service_uses_the_shared_available_tool_assembly() -> None:
    from quanxin_life.api.service import create_available_tool_invocation_service

    service = create_available_tool_invocation_service()

    assert [schema.tool_name.value for schema in service.registry.list_schemas()] == [
        "audit_dataset_split",
        "check_operating_condition",
        "make_batch_decision",
        "validate_battery_data",
    ]


def test_service_requires_explicit_nonempty_allowlist_for_agent_invocation() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService
    from quanxin_life.tools import ToolAuthorizationError

    service = ToolInvocationService(registry=_registry())
    invocation = ToolInvocation(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        input_value={"batch_id": "batch-A"},
    )

    with pytest.raises(ToolAuthorizationError, match="non-empty allowlist"):
        service.invoke_for_agent(invocation, allowed_tool_names=())


def test_invocation_rejects_non_json_input_value() -> None:
    from quanxin_life.api.service import ToolInvocation

    with pytest.raises(ValueError, match="JSON-compatible"):
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": object()},
        )


def test_fastapi_factory_fails_explicitly_when_optional_dependency_is_unavailable() -> None:
    if importlib.util.find_spec("fastapi") is not None:
        pytest.skip("FastAPI is installed; transport behavior belongs to the API dependency suite")

    from quanxin_life.api.app import FastApiDependencyUnavailable, create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService

    with pytest.raises(FastApiDependencyUnavailable, match=r"quanxin-life\[api\]"):
        create_fastapi_app(ToolInvocationService(registry=_registry()))
