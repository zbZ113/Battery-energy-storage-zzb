"""Fail-closed display authorization for candidate BLAST scenario results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict

from quanxin_life.core import EvidenceLevel, ToolResult
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import AUDITED_REPORT_TOOL_VERSION
from quanxin_life.scenarios import BlastRouteManifest, load_packaged_blast_route_catalog
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
)

from .cards import AuditedResultAuthorization, AuditedResultAuthorizer

_SCENARIO_TOOL_VERSIONS = {
    "compare_operation_scenarios": COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    "project_storage_lifetime": PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
}
_UNRESOLVED_ROUTE = "unresolved-scenario-route"
_SUPPORTED_DOMAIN = (
    "manifest-bounded non-product-specific LFP reference scenario"
)


class _AuthorizationBase(TypedDict):
    route_id: str
    activation_status: str
    evidence_level: EvidenceLevel
    supported_domain: str


class _RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class BlastScenarioResultAuthorizer:
    """Authorize only internally consistent, explicitly enabled candidate results."""

    def __init__(self, *, allow_candidate_results: bool = False) -> None:
        if not isinstance(allow_candidate_results, bool):
            raise TypeError("allow_candidate_results must be a bool")
        self._allow_candidate_results = allow_candidate_results

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        try:
            checked = ToolResult.model_validate(result.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError):
            return _rejected("SCENARIO_RESULT_INVALID")
        expected_version = _SCENARIO_TOOL_VERSIONS.get(checked.tool_name)
        artifact = checked.values.get("artifact")
        if expected_version != checked.tool_version or not isinstance(artifact, Mapping):
            return _rejected("SCENARIO_RESULT_CONTRACT_MISMATCH")
        route_id = artifact.get("route_id")
        if not isinstance(route_id, str) or not route_id.strip():
            return _rejected("SCENARIO_RESULT_MANIFEST_MISMATCH")
        route = next(
            (
                item
                for item in load_packaged_blast_route_catalog().routes
                if item.route_id == route_id
            ),
            None,
        )
        if route is None:
            return _rejected(
                "SCENARIO_RESULT_MANIFEST_MISMATCH",
                route_id=route_id,
            )
        base = _authorization_base(route)
        if not _matches_manifest(checked, artifact=artifact, route=route):
            return _rejected(
                "SCENARIO_RESULT_MANIFEST_MISMATCH",
                **base,
            )
        status = artifact.get("status")
        if status == "REJECTED":
            reasons = artifact.get("rejection_reasons")
            reason = (
                reasons[0]
                if isinstance(reasons, list)
                and reasons
                and isinstance(reasons[0], str)
                and reasons[0].strip()
                else "SCENARIO_TOOL_REJECTED"
            )
            return _rejected(reason, **base)
        if status != "COMPLETED":
            return _rejected("SCENARIO_RESULT_INVALID", **base)
        if "CANDIDATE_ROUTE_RESEARCH_USE_ONLY" not in checked.warnings:
            return _rejected("SCENARIO_RESULT_MANIFEST_MISMATCH", **base)
        if not self._allow_candidate_results:
            return _rejected(
                "CANDIDATE_SCENARIO_RESULT_NOT_APPROVED",
                **base,
            )
        return AuditedResultAuthorization(
            allowed=True,
            rejection_reason=None,
            **base,
        )


class AuditedScenarioResultAuthorizer:
    """Authorize scenario results and reports through one candidate gate."""

    def __init__(
        self,
        *,
        result_resolver: _RegisteredResultResolver,
        allow_candidate_results: bool = False,
    ) -> None:
        if not callable(getattr(result_resolver, "resolve_registered_result", None)):
            raise TypeError("result_resolver must resolve registered ToolResults")
        self._result_resolver = result_resolver
        self._scenario_authorizer = BlastScenarioResultAuthorizer(
            allow_candidate_results=allow_candidate_results
        )

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        if result.tool_name in _SCENARIO_TOOL_VERSIONS:
            return self._scenario_authorizer.authorize(result)
        if (
            result.tool_name != "generate_audited_report"
            or result.tool_version != AUDITED_REPORT_TOOL_VERSION
            or result.model_version != REPORTING_VERSION
        ):
            return _rejected("AUDITED_RESULT_NOT_SUPPORTED")
        upstream_ids = result.values.get("upstream_result_ids")
        if (
            not isinstance(upstream_ids, list)
            or len(upstream_ids) != 1
            or not isinstance(upstream_ids[0], str)
        ):
            return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
        try:
            upstream = ToolResult.model_validate(
                self._result_resolver.resolve_registered_result(
                    upstream_ids[0]
                ).model_dump(mode="json")
            )
        except (AttributeError, TypeError, ValueError):
            return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
        authorization = self._scenario_authorizer.authorize(upstream)
        if not authorization.allowed:
            return authorization
        return AuditedResultAuthorization(
            allowed=True,
            route_id=authorization.route_id,
            activation_status=authorization.activation_status,
            evidence_level=authorization.evidence_level,
            supported_domain=authorization.supported_domain,
            rejection_reason=None,
        )


class ScenarioAwareAuditedResultAuthorizer:
    """Delegate BLAST scenario results to their candidate-specific gate."""

    def __init__(
        self,
        *,
        scenario_authorizer: BlastScenarioResultAuthorizer,
        fallback_authorizer: AuditedResultAuthorizer,
    ) -> None:
        self._scenario_authorizer = scenario_authorizer
        self._fallback_authorizer = fallback_authorizer

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        if result.tool_name in _SCENARIO_TOOL_VERSIONS:
            return self._scenario_authorizer.authorize(result)
        return self._fallback_authorizer.authorize(result)


def _matches_manifest(
    result: ToolResult,
    *,
    artifact: Mapping[str, object],
    route: BlastRouteManifest,
) -> bool:
    if (
        result.model_version != route.route_version
        or result.tool_name not in route.task_types
        or artifact.get("route_id") != route.route_id
        or artifact.get("evidence_level") != EvidenceLevel.PHYSICS_REFERENCE.value
        or route.lifecycle_status != "REGISTERED_CANDIDATE"
        or route.activation_status != "NOT_ACTIVATED"
        or route.evidence_level is not EvidenceLevel.PHYSICS_REFERENCE
        or not any(
            record.sha256 == route.upstream_model_source_sha256
            for record in result.provenance
        )
    ):
        return False
    if artifact.get("status") == "REJECTED":
        return True
    projections: list[object]
    if result.tool_name == "project_storage_lifetime":
        projections = [artifact.get("projection")]
    else:
        comparisons = artifact.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            return False
        projections = [artifact.get("baseline"), *comparisons]
    for projection in projections:
        if not isinstance(projection, Mapping):
            return False
        if (
            projection.get("route_id") != route.route_id
            or projection.get("route_version") != route.route_version
            or projection.get("evidence_level")
            != EvidenceLevel.PHYSICS_REFERENCE.value
            or projection.get("reference_model")
            != "BLAST-Lite LFP reference scenario"
        ):
            return False
    return True


def _authorization_base(route: BlastRouteManifest) -> _AuthorizationBase:
    return {
        "route_id": route.route_id,
        "activation_status": route.lifecycle_status,
        "evidence_level": EvidenceLevel.PHYSICS_REFERENCE,
        "supported_domain": _SUPPORTED_DOMAIN,
    }


def _rejected(
    reason: str,
    *,
    route_id: str = _UNRESOLVED_ROUTE,
    activation_status: str = "NOT_ACTIVATED",
    evidence_level: EvidenceLevel = EvidenceLevel.PHYSICS_REFERENCE,
    supported_domain: str = _SUPPORTED_DOMAIN,
) -> AuditedResultAuthorization:
    return AuditedResultAuthorization(
        allowed=False,
        route_id=route_id,
        activation_status=activation_status,
        evidence_level=evidence_level,
        supported_domain=supported_domain,
        rejection_reason=reason,
    )


__all__ = [
    "AuditedScenarioResultAuthorizer",
    "BlastScenarioResultAuthorizer",
    "ScenarioAwareAuditedResultAuthorizer",
]
