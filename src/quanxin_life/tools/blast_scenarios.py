"""Audited BLAST-Lite operating-scenario and storage-lifetime tools."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, Self
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import (
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.scenarios import (
    BlastRouteManifest,
    BlastRouteRejected,
    BlastScenarioRejected,
    BlastScenarioRunner,
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioProjection,
    ScenarioStateReference,
    VerifiedScenarioContext,
    load_packaged_blast_route_catalog,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

COMPARE_OPERATION_SCENARIOS_TOOL_VERSION = "compare-operation-scenarios-v1"
PROJECT_STORAGE_LIFETIME_TOOL_VERSION = "project-storage-lifetime-v1"
COMPARE_OPERATION_SCENARIOS_ARTIFACT_TYPE = (
    "quanxin_life.compare_operation_scenarios.v1"
)
PROJECT_STORAGE_LIFETIME_ARTIFACT_TYPE = (
    "quanxin_life.project_storage_lifetime.v1"
)
SCENARIO_FEATURE_VERSION = "operation-scenario-contract-v1"
_REFERENCE_MODEL_LABEL = "BLAST-Lite LFP reference scenario"
_UNCERTAINTY = {
    "kind": "DETERMINISTIC_SCENARIO",
    "prediction_interval_included": False,
    "sensitivity_envelope_included": False,
}
Clock = Callable[[], datetime]


class ScenarioContextResolver(Protocol):
    def resolve_scenario_context(
        self,
        scenario_context_id: str,
    ) -> VerifiedScenarioContext: ...


class RejectingScenarioContextResolver:
    """Fail-closed resolver used when an assembly has no trusted context source."""

    def resolve_scenario_context(
        self,
        scenario_context_id: str,
    ) -> VerifiedScenarioContext:
        del scenario_context_id
        raise ValueError("scenario context resolver is not configured")


class _ScenarioToolInputBase(ContractModel):
    run_id: str
    scenario_context_id: str = Field(min_length=1, max_length=200)
    route_id: str = Field(min_length=1, max_length=200)
    cell: ScenarioCellDescriptor

    @field_validator("run_id")
    @classmethod
    def run_id_is_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("run_id must be a UUID string") from exc
        return value

    @field_validator("scenario_context_id", "route_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scenario tool identifiers must not be blank")
        return normalized


class CompareOperationScenariosToolInput(_ScenarioToolInputBase):
    baseline: OperationScenario
    comparisons: tuple[OperationScenario, ...] = Field(min_length=1, max_length=8)
    current_state_reference: ScenarioStateReference | None = None

    @model_validator(mode="after")
    def scenario_ids_are_unique(self) -> Self:
        identifiers = [self.baseline.scenario_id, *(item.scenario_id for item in self.comparisons)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("baseline and comparison scenario IDs must be unique")
        return self


class ProjectStorageLifetimeToolInput(_ScenarioToolInputBase):
    scenario: OperationScenario
    current_state_reference: ScenarioStateReference | None = None
    new_observation_reference: ScenarioStateReference | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _execution_time(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scenario tool clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _catalog_provenance(created_at: datetime) -> ProvenanceRecord:
    catalog = load_packaged_blast_route_catalog()
    return ProvenanceRecord(
        source_id="blast-lite-route-catalog-v1",
        source_kind=SourceKind.SIMULATED,
        uri="package://quanxin_life.scenarios/manifests/blast_lite_routes_v1.json",
        sha256=sha256_canonical(catalog.model_dump(mode="json")),
        description="Reviewed BLAST-Lite candidate route catalog.",
        created_at=created_at,
    )


def _route_provenance(
    route: BlastRouteManifest,
    *,
    created_at: datetime,
) -> ProvenanceRecord:
    catalog = load_packaged_blast_route_catalog()
    return ProvenanceRecord(
        source_id=route.route_id,
        source_kind=SourceKind.SIMULATED,
        uri=f"{catalog.upstream_repository}/tree/{catalog.upstream_commit}",
        sha256=route.upstream_model_source_sha256,
        description=(
            "Pinned BLAST-Lite source for a candidate reference scenario; "
            "not a product-specific lifetime validation."
        ),
        created_at=created_at,
    )


def _cell_matches(
    declared: ScenarioCellDescriptor,
    trusted: ScenarioCellDescriptor,
) -> bool:
    return (
        declared.chemistry.casefold() == trusted.chemistry.casefold()
        and math.isclose(
            declared.nominal_capacity_ah,
            trusted.nominal_capacity_ah,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and declared.cell_format == trusted.cell_format
    )


def _resolve_route_and_context(
    *,
    scenario_context_id: str,
    route_id: str,
    declared_cell: ScenarioCellDescriptor,
    context_resolver: ScenarioContextResolver,
) -> tuple[VerifiedScenarioContext | None, BlastRouteManifest | None, list[str]]:
    catalog = load_packaged_blast_route_catalog()
    route = next((item for item in catalog.routes if item.route_id == route_id), None)
    try:
        context = context_resolver.resolve_scenario_context(scenario_context_id)
    except (TypeError, ValueError):
        return None, route, ["SCENARIO_CONTEXT_NOT_FOUND"]
    if context.scenario_context_id != scenario_context_id:
        return context, route, ["SCENARIO_CONTEXT_MISMATCH"]
    if not _cell_matches(declared_cell, context.cell):
        return context, route, ["SCENARIO_CONTEXT_MISMATCH"]
    try:
        authorized = catalog.authorize_reference_use(
            route_id=route_id,
            chemistry=context.cell.chemistry,
            nominal_capacity_ah=context.cell.nominal_capacity_ah,
            cell_format=context.cell.cell_format,
            trusted_reference_use=context.trusted_reference_use,
        )
    except BlastRouteRejected as exc:
        return context, route, [str(exc)]
    return context, authorized, []


def _projection_payload(projection: ScenarioProjection) -> dict[str, object]:
    payload = projection.model_dump(mode="json")
    payload.pop("model_class", None)
    payload.update(
        {
            "final_natural_year": projection.natural_years[-1],
            "final_equivalent_full_cycles": projection.equivalent_full_cycles[-1],
            "final_soh": projection.soh[-1],
        }
    )
    payload["reference_model"] = _REFERENCE_MODEL_LABEL
    payload["scenario_assumptions"] = [
        "Beginning-of-life BLAST initialization.",
        "Monthly aggregation of active cycling and SOC residence.",
        "Configured rest is assigned to upper SOC; remaining idle time to lower SOC.",
        "Long-horizon values are deterministic scenarios, not confidence intervals.",
    ]
    return payload


def _result(
    *,
    result_id: str,
    tool_name: StandardToolName,
    tool_version: str,
    artifact_type: str,
    artifact: dict[str, object],
    input_value: ContractModel,
    context: VerifiedScenarioContext | None,
    route: BlastRouteManifest | None,
    warnings: list[str],
    created_at: datetime,
) -> ToolResult:
    provenance = list(context.provenance) if context is not None else []
    provenance.append(
        _route_provenance(route, created_at=created_at)
        if route is not None
        else _catalog_provenance(created_at)
    )
    return ToolResult(
        result_id=result_id,
        tool_name=tool_name.value,
        tool_version=tool_version,
        model_version=(
            route.route_version if route is not None else "blast-lite-route-unresolved-v1"
        ),
        data_version=(
            context.data_version if context is not None else "scenario-context-unresolved-v1"
        ),
        feature_version=SCENARIO_FEATURE_VERSION,
        input_hash=sha256_canonical(input_value.model_dump(mode="json")),
        values={"artifact_type": artifact_type, "artifact": artifact},
        uncertainty=dict(_UNCERTAINTY),
        warnings=list(dict.fromkeys(warnings)),
        provenance=provenance,
        created_at=created_at,
    )


def _rejected_artifact(
    *,
    result_id: str,
    run_id: str,
    scenario_context_id: str,
    route_id: str,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "status": "REJECTED",
        "run_id": run_id,
        "result_id": result_id,
        "scenario_context_id": scenario_context_id,
        "route_id": route_id,
        "evidence_level": EvidenceLevel.PHYSICS_REFERENCE.value,
        "rejection_reasons": reasons,
    }


def execute_compare_operation_scenarios_tool(
    input_value: CompareOperationScenariosToolInput,
    *,
    context_resolver: ScenarioContextResolver,
    runner: BlastScenarioRunner | None = None,
    allow_candidate_execution: bool = False,
    clock: Clock = _utc_now,
) -> ToolResult:
    validated = CompareOperationScenariosToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    created_at = _execution_time(clock)
    result_id = str(uuid4())
    context, route, reasons = _resolve_route_and_context(
        scenario_context_id=validated.scenario_context_id,
        route_id=validated.route_id,
        declared_cell=validated.cell,
        context_resolver=context_resolver,
    )
    if not reasons and not allow_candidate_execution:
        reasons.append("ROUTE_NOT_ACTIVATED")
    projections: list[ScenarioProjection] = []
    if not reasons:
        scenario_runner = runner or BlastScenarioRunner()
        try:
            if route is None:  # pragma: no cover - authorization invariant
                raise BlastScenarioRejected("UNKNOWN_BLAST_ROUTE")
            projections.append(
                scenario_runner.run(
                    route=route,
                    scenario=validated.baseline,
                    initial_state_reference=validated.current_state_reference,
                )
            )
            projections.extend(
                scenario_runner.run(
                    route=route,
                    scenario=scenario,
                    initial_state_reference=validated.current_state_reference,
                )
                for scenario in validated.comparisons
            )
        except BlastScenarioRejected as exc:
            reasons.append(str(exc))
    if reasons:
        artifact = _rejected_artifact(
            result_id=result_id,
            run_id=validated.run_id,
            scenario_context_id=validated.scenario_context_id,
            route_id=validated.route_id,
            reasons=reasons,
        )
        warnings = reasons
    else:
        artifact = {
            "status": "COMPLETED",
            "run_id": validated.run_id,
            "result_id": result_id,
            "scenario_context_id": validated.scenario_context_id,
            "route_id": validated.route_id,
            "reference_model": _REFERENCE_MODEL_LABEL,
            "evidence_level": EvidenceLevel.PHYSICS_REFERENCE.value,
            "baseline": _projection_payload(projections[0]),
            "comparisons": [
                _projection_payload(projection) for projection in projections[1:]
            ],
        }
        warnings = [
            "CANDIDATE_ROUTE_RESEARCH_USE_ONLY",
            *(warning for projection in projections for warning in projection.warnings),
        ]
    return _result(
        result_id=result_id,
        tool_name=StandardToolName.COMPARE_OPERATION_SCENARIOS,
        tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        artifact_type=COMPARE_OPERATION_SCENARIOS_ARTIFACT_TYPE,
        artifact=artifact,
        input_value=validated,
        context=context,
        route=route,
        warnings=warnings,
        created_at=created_at,
    )


def execute_project_storage_lifetime_tool(
    input_value: ProjectStorageLifetimeToolInput,
    *,
    context_resolver: ScenarioContextResolver,
    runner: BlastScenarioRunner | None = None,
    allow_candidate_execution: bool = False,
    clock: Clock = _utc_now,
) -> ToolResult:
    validated = ProjectStorageLifetimeToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    created_at = _execution_time(clock)
    result_id = str(uuid4())
    context, route, reasons = _resolve_route_and_context(
        scenario_context_id=validated.scenario_context_id,
        route_id=validated.route_id,
        declared_cell=validated.cell,
        context_resolver=context_resolver,
    )
    if not reasons and not allow_candidate_execution:
        reasons.append("ROUTE_NOT_ACTIVATED")
    if not reasons and validated.new_observation_reference is not None:
        reasons.append("UNSUPPORTED_BLAST_STATE_INITIALIZATION")
    projection: ScenarioProjection | None = None
    if not reasons:
        scenario_runner = runner or BlastScenarioRunner()
        try:
            if route is None:  # pragma: no cover - authorization invariant
                raise BlastScenarioRejected("UNKNOWN_BLAST_ROUTE")
            projection = scenario_runner.run(
                route=route,
                scenario=validated.scenario,
                initial_state_reference=validated.current_state_reference,
            )
        except BlastScenarioRejected as exc:
            reasons.append(str(exc))
    if reasons:
        artifact = _rejected_artifact(
            result_id=result_id,
            run_id=validated.run_id,
            scenario_context_id=validated.scenario_context_id,
            route_id=validated.route_id,
            reasons=reasons,
        )
        warnings = reasons
    else:
        if projection is None:  # pragma: no cover - execution invariant
            raise RuntimeError("scenario projection was not produced")
        artifact = {
            "status": "COMPLETED",
            "run_id": validated.run_id,
            "result_id": result_id,
            "scenario_context_id": validated.scenario_context_id,
            "route_id": validated.route_id,
            "reference_model": _REFERENCE_MODEL_LABEL,
            "evidence_level": EvidenceLevel.PHYSICS_REFERENCE.value,
            "projection": _projection_payload(projection),
        }
        warnings = ["CANDIDATE_ROUTE_RESEARCH_USE_ONLY", *projection.warnings]
    return _result(
        result_id=result_id,
        tool_name=StandardToolName.PROJECT_STORAGE_LIFETIME,
        tool_version=PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
        artifact_type=PROJECT_STORAGE_LIFETIME_ARTIFACT_TYPE,
        artifact=artifact,
        input_value=validated,
        context=context,
        route=route,
        warnings=warnings,
        created_at=created_at,
    )


def register_compare_operation_scenarios_tool(
    registry: ToolRegistry,
    *,
    context_resolver: ScenarioContextResolver,
    runner: BlastScenarioRunner | None = None,
    allow_candidate_execution: bool = False,
    clock: Clock = _utc_now,
) -> RegisteredTool[CompareOperationScenariosToolInput]:
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.COMPARE_OPERATION_SCENARIOS,
            tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
            input_model=CompareOperationScenariosToolInput,
            executor=lambda value: execute_compare_operation_scenarios_tool(
                value,
                context_resolver=context_resolver,
                runner=runner,
                allow_candidate_execution=allow_candidate_execution,
                clock=clock,
            ),
        )
    )


def register_project_storage_lifetime_tool(
    registry: ToolRegistry,
    *,
    context_resolver: ScenarioContextResolver,
    runner: BlastScenarioRunner | None = None,
    allow_candidate_execution: bool = False,
    clock: Clock = _utc_now,
) -> RegisteredTool[ProjectStorageLifetimeToolInput]:
    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PROJECT_STORAGE_LIFETIME,
            tool_version=PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
            input_model=ProjectStorageLifetimeToolInput,
            executor=lambda value: execute_project_storage_lifetime_tool(
                value,
                context_resolver=context_resolver,
                runner=runner,
                allow_candidate_execution=allow_candidate_execution,
                clock=clock,
            ),
        )
    )


__all__ = [
    "COMPARE_OPERATION_SCENARIOS_ARTIFACT_TYPE",
    "COMPARE_OPERATION_SCENARIOS_TOOL_VERSION",
    "PROJECT_STORAGE_LIFETIME_ARTIFACT_TYPE",
    "PROJECT_STORAGE_LIFETIME_TOOL_VERSION",
    "CompareOperationScenariosToolInput",
    "ProjectStorageLifetimeToolInput",
    "RejectingScenarioContextResolver",
    "ScenarioContextResolver",
    "execute_compare_operation_scenarios_tool",
    "execute_project_storage_lifetime_tool",
    "register_compare_operation_scenarios_tool",
    "register_project_storage_lifetime_tool",
]
