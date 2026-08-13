"""Fail-closed display authorization for candidate BLAST scenario results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict
from uuid import UUID

from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    CycleLifePrediction,
    EvidenceLevel,
    PredictionTarget,
    SourceKind,
    ToolResult,
)
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import AUDITED_REPORT_TOOL_VERSION
from quanxin_life.scenarios import BlastRouteManifest, load_packaged_blast_route_catalog
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
    AdvancedSOHInference,
)
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
_ADVANCED_RUL_DOMAIN = "activated project-bound MATR official cycle-life route"
_ADVANCED_RUL_UNRESOLVED_ROUTE = "unresolved-advanced-rul-route"
_ADVANCED_RUL_SHA_FIELDS = (
    "raw_sequence_input_sha256",
    "transform_config_sha256",
    "source_manifest_hash",
    "normalization_statistics_sha256",
    "artifact_manifest_sha256",
    "ledger_head_sha256",
)
_ADVANCED_SOH_DOMAIN = "activated project-bound MATR finite SOH route"
_ADVANCED_SOH_UNRESOLVED_ROUTE = "unresolved-advanced-soh-route"
_ADVANCED_SOH_SHA_FIELDS = _ADVANCED_RUL_SHA_FIELDS


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
    """Authorize audited scenario and active Advanced RUL/SOH results."""

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
        if result.tool_name == "predict_cycle_life":
            return _authorize_advanced_rul(result)
        if result.tool_name == "predict_soh_trajectory":
            return _authorize_advanced_soh(result)
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
        if upstream.tool_name in _SCENARIO_TOOL_VERSIONS:
            authorization = self._scenario_authorizer.authorize(upstream)
        elif upstream.tool_name == "predict_cycle_life":
            authorization = _authorize_advanced_rul(upstream)
            if authorization.allowed and not _matches_advanced_rul_report(
                result,
                upstream,
            ):
                return _advanced_rul_rejected(
                    "ADVANCED_RUL_REPORT_CONTRACT_MISMATCH",
                    route_id=authorization.route_id,
                )
        elif upstream.tool_name == "predict_soh_trajectory":
            authorization = _authorize_advanced_soh(upstream)
            if authorization.allowed and not _matches_advanced_soh_report(
                result,
                upstream,
            ):
                return _advanced_soh_rejected(
                    "ADVANCED_SOH_REPORT_CONTRACT_MISMATCH",
                    route_id=authorization.route_id,
                )
        else:
            return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
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


def _authorize_advanced_rul(result: ToolResult) -> AuditedResultAuthorization:
    try:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        artifact_value = checked.values.get("artifact")
        if not isinstance(artifact_value, Mapping):
            raise ValueError("Advanced RUL artifact is invalid")
        artifact = artifact_value
        prediction_value = artifact.get("cycle_life_prediction")
        prediction = CycleLifePrediction.model_validate(prediction_value)
        route_role_value = artifact.get("route_role")
        task_value = artifact.get("task")
        if not isinstance(route_role_value, str) or not isinstance(task_value, str):
            raise ValueError("Advanced RUL route identity is invalid")
        route_role = AdvancedModelRouteRole(route_role_value)
        task = AdvancedModelTask(task_value)
        artifact_id = _uuid(artifact.get("artifact_id"))
        decision_event_id = _uuid(artifact.get("decision_event_id"))
        _uuid(artifact.get("record_batch_id"))
        _uuid(artifact.get("upstream_result_id"))
    except (AttributeError, TypeError, ValueError):
        return _advanced_rul_rejected("ADVANCED_RUL_RESULT_CONTRACT_MISMATCH")
    if (
        checked.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
        or checked.values.get("artifact_type")
        != ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE
        or prediction.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        or not prediction.right_censored
        or prediction.observed_cycle is not None
        or task is not AdvancedModelTask.RUL
        or artifact.get("output_target")
        != PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value
        or artifact.get("artifact_kind")
        not in {"cyclepatch_direct", "cyclepatch_batlinet"}
        or artifact.get("dataset_id") != prediction.dataset_id
        or artifact.get("cell_id") != prediction.cell_id
        or artifact.get("cutoff_cycle") != prediction.cutoff_cycle
        or artifact.get("split_version") != prediction.split_version
        or checked.model_version != prediction.model_version
        or checked.data_version != prediction.data_version
        or checked.feature_version != prediction.feature_version
        or artifact.get("derived_remaining_cycles")
        != prediction.derived_remaining_cycles
        or not _approved_rul_role(prediction.cutoff_cycle, route_role)
        or not _positive_int(artifact.get("ledger_sequence_number"))
        or any(not _sha256(artifact.get(name)) for name in _ADVANCED_RUL_SHA_FIELDS)
        or not _matches_advanced_rul_provenance(
            checked,
            artifact_id=artifact_id,
            artifact_manifest_sha256=artifact.get("artifact_manifest_sha256"),
        )
    ):
        return _advanced_rul_rejected("ADVANCED_RUL_RESULT_CONTRACT_MISMATCH")
    return AuditedResultAuthorization(
        allowed=True,
        route_id=decision_event_id,
        activation_status="ACTIVE",
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
        supported_domain=_ADVANCED_RUL_DOMAIN,
        rejection_reason=None,
    )


def _authorize_advanced_soh(result: ToolResult) -> AuditedResultAuthorization:
    try:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        artifact_value = checked.values.get("artifact")
        if not isinstance(artifact_value, Mapping):
            raise ValueError("Advanced SOH artifact is invalid")
        artifact = artifact_value
        inference = AdvancedSOHInference.model_validate(
            {
                **{
                    name: artifact.get(name)
                    for name in AdvancedSOHInference.model_fields
                },
                "data_version": checked.data_version,
                "feature_version": checked.feature_version,
                "model_version": checked.model_version,
            }
        )
        artifact_id = _uuid(artifact.get("artifact_id"))
        decision_event_id = _uuid(artifact.get("decision_event_id"))
        _uuid(artifact.get("record_batch_id"))
        _uuid(artifact.get("upstream_result_id"))
    except (AttributeError, TypeError, ValueError):
        return _advanced_soh_rejected("ADVANCED_SOH_RESULT_CONTRACT_MISMATCH")
    uncertainty = checked.uncertainty
    if (
        checked.tool_version != ADVANCED_SOH_PREDICTION_TOOL_VERSION
        or checked.values.get("artifact_type")
        != ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE
        or inference.task is not AdvancedModelTask.SOH
        or inference.output_target != "soh_trajectory"
        or inference.route_role
        not in {
            AdvancedModelRouteRole.MEAN_ACCURACY,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
        }
        or artifact.get("horizon_end_cycle") != inference.prediction_cycles[-1]
        or inference.prediction_cycles[-1] != 500
        or checked.model_version != inference.model_version
        or checked.data_version != inference.data_version
        or checked.feature_version != inference.feature_version
        or not isinstance(uncertainty, Mapping)
        or uncertainty.get("finite_horizon_only") is not True
        or uncertainty.get("conformal_interval_included") is not False
        or not _positive_int(artifact.get("ledger_sequence_number"))
        or any(not _sha256(artifact.get(name)) for name in _ADVANCED_SOH_SHA_FIELDS)
        or not _matches_advanced_model_provenance(
            checked,
            artifact_id=artifact_id,
            artifact_manifest_sha256=artifact.get("artifact_manifest_sha256"),
        )
    ):
        return _advanced_soh_rejected("ADVANCED_SOH_RESULT_CONTRACT_MISMATCH")
    return AuditedResultAuthorization(
        allowed=True,
        route_id=decision_event_id,
        activation_status="ACTIVE",
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
        supported_domain=_ADVANCED_SOH_DOMAIN,
        rejection_reason=None,
    )


def _matches_advanced_rul_report(report: ToolResult, upstream: ToolResult) -> bool:
    context = report.values.get("upstream_context")
    if not isinstance(context, list) or len(context) != 1:
        return False
    expected_context = {
        "result_id": upstream.result_id,
        "tool_name": upstream.tool_name,
        "tool_version": upstream.tool_version,
        "model_version": upstream.model_version or "",
        "data_version": upstream.data_version or "",
        "feature_version": upstream.feature_version or "",
        "input_hash": upstream.input_hash,
    }
    return (
        report.values.get("report_kind") == "lifetime_decision"
        and report.values.get("claim_ids") == ["lifetime_prediction"]
        and context[0] == expected_context
    )


def _matches_advanced_soh_report(report: ToolResult, upstream: ToolResult) -> bool:
    return _matches_advanced_rul_report(report, upstream)


def _approved_rul_role(cutoff_cycle: int, role: AdvancedModelRouteRole) -> bool:
    if cutoff_cycle == 20:
        return role is AdvancedModelRouteRole.DEFAULT
    return cutoff_cycle in {50, 100, 150} and role in {
        AdvancedModelRouteRole.POINT_ACCURACY,
        AdvancedModelRouteRole.COVERAGE,
    }


def _matches_advanced_rul_provenance(
    result: ToolResult,
    *,
    artifact_id: str,
    artifact_manifest_sha256: object,
) -> bool:
    return _matches_advanced_model_provenance(
        result,
        artifact_id=artifact_id,
        artifact_manifest_sha256=artifact_manifest_sha256,
    )


def _matches_advanced_model_provenance(
    result: ToolResult,
    *,
    artifact_id: str,
    artifact_manifest_sha256: object,
) -> bool:
    return any(
        record.source_kind is SourceKind.PREDICTED
        and record.source_id == f"advanced-model-{artifact_id}"
        and record.uri == f"artifact://advanced-model/{artifact_id}"
        and record.sha256 == artifact_manifest_sha256
        for record in result.provenance
    )


def _uuid(value: object) -> str:
    return str(UUID(value))  # type: ignore[arg-type]


def _sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _advanced_rul_rejected(
    reason: str,
    *,
    route_id: str = _ADVANCED_RUL_UNRESOLVED_ROUTE,
) -> AuditedResultAuthorization:
    return AuditedResultAuthorization(
        allowed=False,
        route_id=route_id,
        activation_status="NOT_ACTIVATED",
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
        supported_domain=_ADVANCED_RUL_DOMAIN,
        rejection_reason=reason,
    )


def _advanced_soh_rejected(
    reason: str,
    *,
    route_id: str = _ADVANCED_SOH_UNRESOLVED_ROUTE,
) -> AuditedResultAuthorization:
    return AuditedResultAuthorization(
        allowed=False,
        route_id=route_id,
        activation_status="NOT_ACTIVATED",
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
        supported_domain=_ADVANCED_SOH_DOMAIN,
        rejection_reason=reason,
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
