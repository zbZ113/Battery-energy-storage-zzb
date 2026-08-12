"""Shared, auditable execution registry for domain tools.

The registry deliberately performs no battery-domain calculation.  It validates
typed inputs, invokes a registered numerical tool, and rejects any returned
result whose audit metadata does not exactly describe that invocation.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from pydantic import ValidationError

from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )


class StandardToolName(StrEnum):
    """The only planned domain tool names exposed to agents and service layers."""

    VALIDATE_BATTERY_DATA = "validate_battery_data"
    AUDIT_DATASET_SPLIT = "audit_dataset_split"
    EXTRACT_EARLY_CYCLE_FEATURES = "extract_early_cycle_features"
    INGEST_NEWLY_OBSERVED_SOH = "ingest_newly_observed_soh"
    PREDICT_CYCLE_LIFE = "predict_cycle_life"
    CONVERT_SCENARIO_LIFETIME = "convert_scenario_lifetime"
    COMPARE_OPERATION_SCENARIOS = "compare_operation_scenarios"
    PROJECT_STORAGE_LIFETIME = "project_storage_lifetime"
    PREDICT_SOH_TRAJECTORY = "predict_soh_trajectory"
    CALIBRATE_PREDICTION_INTERVAL = "calibrate_prediction_interval"
    ADAPT_TO_TARGET_DOMAIN = "adapt_to_target_domain"
    UPDATE_CELL_PARAMETERS = "update_cell_parameters"
    CHECK_OPERATING_CONDITION = "check_operating_condition"
    RECOMMEND_NEXT_EXPERIMENT = "recommend_next_experiment"
    MAKE_BATCH_DECISION = "make_batch_decision"
    RETRIEVE_BATTERY_EVIDENCE = "retrieve_battery_evidence"
    GENERATE_AUDITED_REPORT = "generate_audited_report"


class ToolExecutionScope(StrEnum):
    """Trusted execution boundary required by one registered tool."""

    GLOBAL = "GLOBAL"
    PROJECT = "PROJECT"


class ToolRegistryError(RuntimeError):
    """Base class for explicit tool registry failures."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a name already has a registered implementation."""


class UnknownToolError(ToolRegistryError):
    """Raised when a caller requests a standard tool without an implementation."""


class ToolAuthorizationError(ToolRegistryError):
    """Raised when an agent attempts to execute a tool outside its allowlist."""


class ToolInputValidationError(ToolRegistryError):
    """Raised when untrusted input cannot satisfy a tool's Pydantic contract."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a registered tool raises before it can return a result."""


class ToolContractError(ToolRegistryError):
    """Raised when a tool return value fails the shared audit contract."""


InputContractT = TypeVar("InputContractT", bound=ContractModel)


class ProjectInvocationContextValidator(Protocol):
    """Live validator for issuer-bound project invocation contexts."""

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext: ...


@dataclass(frozen=True, slots=True)
class ToolDefinition(Generic[InputContractT]):
    """A versioned, strongly typed domain tool implementation."""

    tool_name: StandardToolName
    tool_version: str
    input_model: type[InputContractT]
    executor: Callable[[InputContractT], ToolResult] | None
    execution_scope: ToolExecutionScope = ToolExecutionScope.GLOBAL
    project_executor: (
        Callable[[InputContractT, VerifiedProjectInvocationContext], ToolResult] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, StandardToolName):
            raise TypeError("tool_name must be a StandardToolName")
        if not isinstance(self.tool_version, str) or not self.tool_version.strip():
            raise ValueError("tool_version must not be blank")
        if not isinstance(self.input_model, type) or not issubclass(
            self.input_model, ContractModel
        ):
            raise TypeError("input_model must be a ContractModel type")
        if not isinstance(self.execution_scope, ToolExecutionScope):
            raise TypeError("execution_scope must be a ToolExecutionScope")
        if self.execution_scope is ToolExecutionScope.GLOBAL:
            if not callable(self.executor):
                raise TypeError("executor must be callable for a global-scoped tool")
            if self.project_executor is not None:
                raise TypeError("global-scoped tool cannot define project_executor")
        else:
            if self.executor is not None:
                raise TypeError("project-scoped tool cannot define a global executor")
            if not callable(self.project_executor):
                raise TypeError("project_executor must be callable for a project-scoped tool")


@dataclass(frozen=True, slots=True)
class RegisteredTool(Generic[InputContractT]):
    """A registry-owned tool definition with no hidden service initialization."""

    definition: ToolDefinition[InputContractT]

    @property
    def tool_name(self) -> StandardToolName:
        return self.definition.tool_name

    @property
    def tool_version(self) -> str:
        return self.definition.tool_version


class ToolSchema(ContractModel):
    """Serializable metadata for a tool discovery endpoint or MCP adapter."""

    tool_name: StandardToolName
    tool_version: str
    input_schema: dict[str, Any]


class ToolRegistry:
    """In-process registry shared by agents, APIs, MCP and user interfaces."""

    def __init__(
        self,
        *,
        project_context_validator: ProjectInvocationContextValidator | None = None,
    ) -> None:
        self._tools: dict[StandardToolName, RegisteredTool[ContractModel]] = {}
        self._project_context_validator = project_context_validator

    @property
    def project_context_validator(self) -> ProjectInvocationContextValidator | None:
        return self._project_context_validator

    def register(
        self, definition: ToolDefinition[InputContractT]
    ) -> RegisteredTool[InputContractT]:
        """Register exactly one implementation for each standard tool name."""

        if definition.tool_name in self._tools:
            raise DuplicateToolError(
                f"Tool '{definition.tool_name.value}' is already registered and cannot be replaced"
            )

        registered = RegisteredTool(definition=definition)
        self._tools[definition.tool_name] = cast(RegisteredTool[ContractModel], registered)
        return registered

    def list_schemas(
        self,
        *,
        execution_scope: ToolExecutionScope = ToolExecutionScope.GLOBAL,
    ) -> tuple[ToolSchema, ...]:
        """Return deterministic discovery metadata for one trusted execution scope."""

        if not isinstance(execution_scope, ToolExecutionScope):
            raise TypeError("execution_scope must be a ToolExecutionScope")

        return tuple(
            ToolSchema(
                tool_name=registered.tool_name,
                tool_version=registered.tool_version,
                input_schema=registered.definition.input_model.model_json_schema(),
            )
            for _, registered in sorted(self._tools.items(), key=lambda item: item[0].value)
            if registered.definition.execution_scope is execution_scope
        )

    def execute(
        self,
        tool_name: StandardToolName | str,
        input_value: Mapping[str, Any] | ContractModel,
        *,
        allowed_tool_names: Collection[StandardToolName | str] | None = None,
    ) -> ToolResult:
        """Validate, authorize and execute a tool without modifying its result."""

        normalized_name = self._coerce_tool_name(tool_name)
        registered = self._tools.get(normalized_name)
        if registered is None:
            raise UnknownToolError(
                f"No implementation is registered for tool '{normalized_name.value}'"
            )

        if registered.definition.execution_scope is ToolExecutionScope.PROJECT:
            raise ToolAuthorizationError(
                f"Tool '{normalized_name.value}' is project-scoped and requires "
                "a verified project invocation context"
            )

        if allowed_tool_names is not None and normalized_name not in self._normalize_allowlist(
            allowed_tool_names
        ):
            raise ToolAuthorizationError(
                f"Tool '{normalized_name.value}' is not permitted for this execution"
            )

        executor = registered.definition.executor
        if executor is None:  # pragma: no cover - guarded by ToolDefinition
            raise ToolAuthorizationError("global-scoped tool executor is unavailable")
        return self._execute_registered(registered, input_value, executor)

    def execute_in_project(
        self,
        tool_name: StandardToolName | str,
        input_value: Mapping[str, Any] | ContractModel,
        *,
        context: VerifiedProjectInvocationContext,
        allowed_tool_names: Collection[StandardToolName | str] | None = None,
    ) -> ToolResult:
        """Execute only a PROJECT tool with one verified immutable context."""

        from quanxin_life.application.invocation_context import (
            VerifiedProjectInvocationContext,
        )

        if not isinstance(context, VerifiedProjectInvocationContext):
            raise ToolAuthorizationError("verified project invocation context is required")
        if self._project_context_validator is None:
            raise ToolAuthorizationError("project context validator is not configured")
        verified_context = self._project_context_validator.revalidate(context)
        normalized_name = self._coerce_tool_name(tool_name)
        registered = self._tools.get(normalized_name)
        if registered is None:
            raise UnknownToolError(
                f"No implementation is registered for tool '{normalized_name.value}'"
            )
        if registered.definition.execution_scope is ToolExecutionScope.GLOBAL:
            raise ToolAuthorizationError(
                f"Tool '{normalized_name.value}' is global-scoped and cannot execute "
                "through a project invocation"
            )
        if allowed_tool_names is not None and normalized_name not in self._normalize_allowlist(
            allowed_tool_names
        ):
            raise ToolAuthorizationError(
                f"Tool '{normalized_name.value}' is not permitted for this execution"
            )
        executor = registered.definition.project_executor
        if executor is None:  # pragma: no cover - guarded by ToolDefinition
            raise ToolAuthorizationError("project-scoped tool executor is unavailable")
        return self._execute_registered(
            registered,
            input_value,
            lambda value: executor(value, verified_context),
        )

    def execute_for_agent(
        self,
        tool_name: StandardToolName | str,
        input_value: Mapping[str, Any] | ContractModel,
        *,
        allowed_tool_names: Collection[StandardToolName | str] | None = None,
    ) -> ToolResult:
        """Execute through the strict Agent boundary with a nonempty allowlist."""

        if allowed_tool_names is None:
            raise ToolAuthorizationError("Agent execution requires a non-empty allowlist")
        normalized_allowlist = self._normalize_allowlist(allowed_tool_names)
        if not normalized_allowlist:
            raise ToolAuthorizationError("Agent execution requires a non-empty allowlist")
        return self.execute(
            tool_name,
            input_value,
            allowed_tool_names=normalized_allowlist,
        )

    def canonical_input_hash(
        self,
        tool_name: StandardToolName | str,
        input_value: Mapping[str, Any] | ContractModel,
    ) -> str:
        """Hash the complete Pydantic-normalized input for one registered tool."""

        return sha256_canonical(
            self.canonical_input_value(tool_name, input_value)
        )

    def canonical_input_value(
        self,
        tool_name: StandardToolName | str,
        input_value: Mapping[str, Any] | ContractModel,
    ) -> dict[str, Any]:
        """Return the complete JSON input after registered Pydantic validation."""

        normalized_name = self._coerce_tool_name(tool_name)
        registered = self._tools.get(normalized_name)
        if registered is None:
            raise UnknownToolError(
                f"No implementation is registered for tool '{normalized_name.value}'"
            )
        validated = self._validate_input(registered, input_value)
        return validated.model_dump(mode="json")

    def execution_scope(
        self,
        tool_name: StandardToolName | str,
    ) -> ToolExecutionScope:
        """Return the immutable execution boundary for one registered tool."""

        normalized_name = self._coerce_tool_name(tool_name)
        registered = self._tools.get(normalized_name)
        if registered is None:
            raise UnknownToolError(
                f"No implementation is registered for tool '{normalized_name.value}'"
            )
        return registered.definition.execution_scope

    def _execute_registered(
        self,
        registered: RegisteredTool[ContractModel],
        input_value: Mapping[str, Any] | ContractModel,
        executor: Callable[[ContractModel], ToolResult],
    ) -> ToolResult:
        validated_input = self._validate_input(registered, input_value)
        expected_input_hash = sha256_canonical(validated_input.model_dump(mode="json"))

        try:
            result = executor(validated_input)
        except ToolRegistryError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{registered.tool_name.value}' failed before returning a ToolResult"
            ) from exc

        return self._validate_result(registered, result, expected_input_hash)

    @staticmethod
    def _coerce_tool_name(tool_name: StandardToolName | str) -> StandardToolName:
        try:
            return StandardToolName(tool_name)
        except ValueError as exc:
            raise UnknownToolError(f"Unknown standard tool '{tool_name}'") from exc

    @classmethod
    def _normalize_allowlist(
        cls, allowed_tool_names: Collection[StandardToolName | str]
    ) -> frozenset[StandardToolName]:
        try:
            return frozenset(cls._coerce_tool_name(tool_name) for tool_name in allowed_tool_names)
        except UnknownToolError as exc:
            message = "Tool allowlist contains an unknown standard tool"
            raise ToolAuthorizationError(message) from exc

    @staticmethod
    def _validate_input(
        registered: RegisteredTool[ContractModel],
        input_value: Mapping[str, Any] | ContractModel,
    ) -> ContractModel:
        if isinstance(input_value, ContractModel):
            raw_input: Mapping[str, Any] = input_value.model_dump(mode="json")
        elif isinstance(input_value, Mapping):
            raw_input = input_value
        else:
            raise ToolInputValidationError("Tool input must be a mapping or ContractModel")

        try:
            return registered.definition.input_model.model_validate(raw_input)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ToolInputValidationError(
                f"Input does not satisfy '{registered.tool_name.value}' contract"
            ) from exc

    @staticmethod
    def _validate_result(
        registered: RegisteredTool[ContractModel],
        result: object,
        expected_input_hash: str,
    ) -> ToolResult:
        if not isinstance(result, ToolResult):
            raise ToolContractError("Registered tool executor must return a ToolResult")

        try:
            validated_result = ToolResult.model_validate(result.model_dump(mode="json"))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ToolContractError(
                "ToolResult does not satisfy its public Pydantic contract"
            ) from exc

        if validated_result.tool_name != registered.tool_name.value:
            raise ToolContractError("ToolResult tool_name does not match the registered tool")
        if validated_result.tool_version != registered.tool_version:
            raise ToolContractError("ToolResult tool_version does not match the registered tool")
        if validated_result.input_hash != expected_input_hash:
            raise ToolContractError("ToolResult input_hash does not match validated input")
        version_fields = ("model_version", "data_version", "feature_version")
        for field_name in version_fields:
            version = getattr(validated_result, field_name)
            if version is None or not version.strip():
                raise ToolContractError(f"ToolResult must declare nonblank {field_name}")
        if not validated_result.provenance:
            raise ToolContractError("ToolResult must include provenance")
        return validated_result
