from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel


class RegistryInput(ContractModel):
    request_label: str = Field(min_length=1)


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="registry-test-input",
        source_kind=SourceKind.OBSERVED,
        uri="test://tool-registry/input",
        sha256=sha256_canonical({"fixture": "registry"}),
        description="Registry test input provenance",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _result(
    validated_input: RegistryInput,
    *,
    tool_name: str = "validate_battery_data",
    tool_version: str = "1.0.0",
    input_hash: str | None = None,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version=tool_version,
        model_version="registry-test-model-v1",
        data_version="registry-test-data-v1",
        feature_version="registry-test-feature-v1",
        input_hash=input_hash
        or sha256_canonical(validated_input.model_dump(mode="json")),
        values={"execution_status": "validated"},
        provenance=[_provenance()],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _definition(
    executor: Callable[[RegistryInput], ToolResult] = _result,
):
    from quanxin_life.tools import StandardToolName, ToolDefinition

    return ToolDefinition(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="1.0.0",
        input_model=RegistryInput,
        executor=executor,
    )


def test_executes_validated_input_and_preserves_tool_produced_audit_fields() -> None:
    from quanxin_life.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())

    result = registry.execute("validate_battery_data", {"request_label": "batch-A"})

    assert result.tool_name == "validate_battery_data"
    assert result.tool_version == "1.0.0"
    assert result.input_hash == sha256_canonical({"request_label": "batch-A"})
    assert result.values == {"execution_status": "validated"}
    assert result.provenance == [_provenance()]


def test_rejects_invalid_input_before_tool_execution() -> None:
    from quanxin_life.tools import ToolInputValidationError, ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())

    with pytest.raises(ToolInputValidationError):
        registry.execute("validate_battery_data", {"unknown_field": "not-permitted"})


def test_revalidates_constructed_input_instead_of_trusting_its_runtime_type() -> None:
    from quanxin_life.tools import ToolInputValidationError, ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())
    forged_input = RegistryInput.model_construct(request_label="")

    with pytest.raises(ToolInputValidationError):
        registry.execute("validate_battery_data", forged_input)


def test_rejects_tool_outside_agent_allowlist() -> None:
    from quanxin_life.tools import ToolAuthorizationError, ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())

    with pytest.raises(ToolAuthorizationError):
        registry.execute(
            "validate_battery_data",
            {"request_label": "batch-A"},
            allowed_tool_names=frozenset(),
        )


def test_agent_execution_requires_a_nonempty_allowlist() -> None:
    from quanxin_life.tools import ToolAuthorizationError, ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())

    with pytest.raises(ToolAuthorizationError, match="non-empty allowlist"):
        registry.execute_for_agent("validate_battery_data", {"request_label": "batch-A"})
    with pytest.raises(ToolAuthorizationError, match="non-empty allowlist"):
        registry.execute_for_agent(
            "validate_battery_data",
            {"request_label": "batch-A"},
            allowed_tool_names=frozenset(),
        )

    result = registry.execute_for_agent(
        "validate_battery_data",
        {"request_label": "batch-A"},
        allowed_tool_names={"validate_battery_data"},
    )

    assert result.tool_name == "validate_battery_data"


@pytest.mark.parametrize(
    ("tool_name", "tool_version", "input_hash"),
    [
        ("predict_cycle_life", "1.0.0", None),
        ("validate_battery_data", "2.0.0", None),
        ("validate_battery_data", "1.0.0", "0" * 64),
    ],
)
def test_rejects_result_audit_contract_mismatch(
    tool_name: str,
    tool_version: str,
    input_hash: str | None,
) -> None:
    from quanxin_life.tools import ToolContractError, ToolRegistry

    def invalid_executor(validated_input: RegistryInput) -> ToolResult:
        return _result(
            validated_input,
            tool_name=tool_name,
            tool_version=tool_version,
            input_hash=input_hash,
        )

    registry = ToolRegistry()
    registry.register(_definition(invalid_executor))

    with pytest.raises(ToolContractError):
        registry.execute("validate_battery_data", {"request_label": "batch-A"})


def test_rejects_non_tool_result_return_type() -> None:
    from quanxin_life.tools import ToolContractError, ToolRegistry

    def invalid_executor(validated_input: RegistryInput) -> ToolResult:
        del validated_input
        return {"execution_status": "invalid"}  # type: ignore[return-value]

    registry = ToolRegistry()
    registry.register(_definition(invalid_executor))

    with pytest.raises(ToolContractError, match="ToolResult"):
        registry.execute("validate_battery_data", {"request_label": "batch-A"})


def test_rejects_result_missing_required_version_metadata() -> None:
    from quanxin_life.tools import ToolContractError, ToolRegistry

    def invalid_executor(validated_input: RegistryInput) -> ToolResult:
        result = _result(validated_input)
        return result.model_copy(update={"feature_version": None})

    registry = ToolRegistry()
    registry.register(_definition(invalid_executor))

    with pytest.raises(ToolContractError, match="feature_version"):
        registry.execute("validate_battery_data", {"request_label": "batch-A"})


def test_revalidates_constructed_tool_result_against_public_contract() -> None:
    from quanxin_life.tools import ToolContractError, ToolRegistry

    def invalid_executor(validated_input: RegistryInput) -> ToolResult:
        result = _result(validated_input)
        forged_payload = dict(result.__dict__)
        forged_payload["result_id"] = "not-a-uuid"
        return ToolResult.model_construct(**forged_payload)

    registry = ToolRegistry()
    registry.register(_definition(invalid_executor))

    with pytest.raises(ToolContractError, match="public Pydantic contract"):
        registry.execute("validate_battery_data", {"request_label": "batch-A"})


def test_rejects_whitespace_only_tool_result_version_metadata() -> None:
    from quanxin_life.tools import ToolContractError, ToolRegistry

    def invalid_executor(validated_input: RegistryInput) -> ToolResult:
        result = _result(validated_input)
        return result.model_copy(update={"data_version": "   "})

    registry = ToolRegistry()
    registry.register(_definition(invalid_executor))

    with pytest.raises(ToolContractError, match="data_version"):
        registry.execute("validate_battery_data", {"request_label": "batch-A"})


def test_definition_rejects_invalid_name_input_type_and_executor() -> None:
    from quanxin_life.tools import StandardToolName, ToolDefinition

    with pytest.raises(TypeError, match="StandardToolName"):
        ToolDefinition(  # type: ignore[arg-type]
            tool_name="validate_battery_data",
            tool_version="1.0.0",
            input_model=RegistryInput,
            executor=_result,
        )
    with pytest.raises(TypeError, match="input_model"):
        ToolDefinition(  # type: ignore[type-var]
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="1.0.0",
            input_model="not-a-contract-model",
            executor=_result,
        )
    with pytest.raises(TypeError, match="executor"):
        ToolDefinition(  # type: ignore[arg-type]
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="1.0.0",
            input_model=RegistryInput,
            executor="not-callable",
        )


def test_rejects_duplicate_registration_and_unknown_tool() -> None:
    from quanxin_life.tools import DuplicateToolError, ToolRegistry, UnknownToolError

    registry = ToolRegistry()
    registry.register(_definition())

    with pytest.raises(DuplicateToolError):
        registry.register(_definition())
    with pytest.raises(UnknownToolError):
        registry.execute("unknown_tool", {"request_label": "batch-A"})


def test_lists_complete_input_schema_for_registered_standard_tool() -> None:
    from quanxin_life.tools import StandardToolName, ToolRegistry

    registry = ToolRegistry()
    registry.register(_definition())

    schemas = registry.list_schemas()

    assert len(schemas) == 1
    assert schemas[0].tool_name is StandardToolName.VALIDATE_BATTERY_DATA
    assert schemas[0].tool_version == "1.0.0"
    assert schemas[0].input_schema["properties"]["request_label"]["type"] == "string"
    assert schemas[0].input_schema["additionalProperties"] is False
