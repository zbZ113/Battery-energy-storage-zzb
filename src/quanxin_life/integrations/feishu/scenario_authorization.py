"""Fail-closed display authorization for candidate BLAST scenario results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Protocol, TypedDict, cast
from uuid import UUID

from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    CycleLifePrediction,
    EvidenceLevel,
    PredictionTarget,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import DataQualityReport
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_VERSION,
    RECOMMENDATION_REPORT_RENDERER_VERSION,
)
from quanxin_life.reporting.engineering_recommendation import (
    render_engineering_recommendation_markdown,
)
from quanxin_life.scenarios import BlastRouteManifest, load_packaged_blast_route_catalog
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
    AdvancedSOHInference,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
)
from quanxin_life.tools.cell_metadata_evidence import (
    validate_versioned_cell_metadata_evidence,
)
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
)
from quanxin_life.tools.engineering_recommendation import (
    ENGINEERING_RECOMMENDATION_FEATURE_VERSION,
    ENGINEERING_RECOMMENDATION_MODEL_VERSION,
    ENGINEERING_RECOMMENDATION_TOOL_VERSION,
    EngineeringRecommendationToolInput,
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
_DATA_QUALITY_DOMAIN = "registered battery cycle-data validation rules"
_ENGINEERING_RECOMMENDATION_DOMAIN = (
    "same-root reviewed ToolResult engineering decision support"
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
        if result.tool_name == "validate_battery_data":
            return _authorize_data_quality(result)
        if result.tool_name == "predict_cycle_life":
            return _authorize_advanced_rul(result)
        if result.tool_name == "predict_soh_trajectory":
            return _authorize_advanced_soh(result)
        if result.tool_name == "make_engineering_recommendation":
            return _authorize_engineering_recommendation(result)
        if (
            result.tool_name != "generate_audited_report"
            or result.tool_version != AUDITED_REPORT_TOOL_VERSION
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
            if result.model_version != REPORTING_VERSION:
                return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
            authorization = self._scenario_authorizer.authorize(upstream)
        elif upstream.tool_name == "predict_cycle_life":
            if result.model_version != REPORTING_VERSION:
                return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
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
            if result.model_version != REPORTING_VERSION:
                return _rejected("AUDITED_REPORT_UPSTREAM_INVALID")
            authorization = _authorize_advanced_soh(upstream)
            if authorization.allowed and not _matches_advanced_soh_report(
                result,
                upstream,
            ):
                return _advanced_soh_rejected(
                    "ADVANCED_SOH_REPORT_CONTRACT_MISMATCH",
                    route_id=authorization.route_id,
                )
        elif upstream.tool_name == "make_engineering_recommendation":
            authorization = _authorize_engineering_recommendation(upstream)
            if authorization.allowed and not _matches_recommendation_report(
                result,
                upstream,
            ):
                return _rejected(
                    "ENGINEERING_RECOMMENDATION_REPORT_CONTRACT_MISMATCH",
                    route_id=authorization.route_id,
                    activation_status=authorization.activation_status,
                    evidence_level=authorization.evidence_level,
                    supported_domain=authorization.supported_domain,
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
        validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=checked.values.get("artifact_type"),
            legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
        )
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
        not in ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES
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


def _authorize_data_quality(result: ToolResult) -> AuditedResultAuthorization:
    try:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        if set(checked.values) != {
            "dataset_id",
            "blocked",
            "quality_score",
            "issue_count",
            "issues",
        }:
            raise ValueError("data-quality values are incomplete")
        report = DataQualityReport.model_validate(
            {
                "dataset_id": checked.values.get("dataset_id"),
                "issues": checked.values.get("issues"),
            }
        )
    except (AttributeError, TypeError, ValueError):
        return _data_quality_rejected("DATA_QUALITY_RESULT_CONTRACT_MISMATCH")
    expected_warnings = list(dict.fromkeys(issue.code for issue in report.issues))
    if (
        checked.tool_version != DATA_QUALITY_TOOL_VERSION
        or checked.model_version != DATA_QUALITY_MODEL_VERSION
        or checked.values.get("blocked") is not report.blocked
        or checked.values.get("quality_score") != report.quality_score
        or checked.values.get("issue_count") != len(report.issues)
        or checked.uncertainty is not None
        or checked.warnings != expected_warnings
        or not any(
            record.source_kind is SourceKind.OBSERVED
            for record in checked.provenance
        )
    ):
        return _data_quality_rejected("DATA_QUALITY_RESULT_CONTRACT_MISMATCH")
    return AuditedResultAuthorization(
        allowed=True,
        route_id=DATA_QUALITY_MODEL_VERSION,
        activation_status="ACTIVE",
        evidence_level=EvidenceLevel.DATA_DIRECT,
        supported_domain=_DATA_QUALITY_DOMAIN,
        rejection_reason=None,
    )


def _data_quality_rejected(reason: str) -> AuditedResultAuthorization:
    return _rejected(
        reason,
        route_id=DATA_QUALITY_MODEL_VERSION,
        activation_status="ACTIVE",
        evidence_level=EvidenceLevel.DATA_DIRECT,
        supported_domain=_DATA_QUALITY_DOMAIN,
    )


def _authorize_advanced_soh(result: ToolResult) -> AuditedResultAuthorization:
    try:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        artifact_value = checked.values.get("artifact")
        if not isinstance(artifact_value, Mapping):
            raise ValueError("Advanced SOH artifact is invalid")
        artifact = artifact_value
        validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=checked.values.get("artifact_type"),
            legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
        )
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
        not in ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES
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


def _authorize_engineering_recommendation(
    result: ToolResult,
) -> AuditedResultAuthorization:
    try:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        values = checked.values
        if set(values) != {
            "recommendation",
            "ruleset_id",
            "ruleset_version",
            "ruleset_manifest_sha256",
            "authorized_upstream_result_ids",
            "authorized_result_selectors",
            "evaluated_value_paths",
            "threshold_evidence",
            "reason_codes",
            "resolution_issues",
        }:
            raise ValueError("recommendation values are incomplete")
        ruleset_id = _safe_recommendation_text(values.get("ruleset_id"))
        _safe_recommendation_text(values.get("ruleset_version"))
        if not _sha256(values.get("ruleset_manifest_sha256")):
            raise ValueError("ruleset manifest is invalid")
        result_ids_value = values.get("authorized_upstream_result_ids")
        if not isinstance(result_ids_value, list):
            raise ValueError("authorized result IDs are invalid")
        input_value = EngineeringRecommendationToolInput(
            result_ids=tuple(result_ids_value),
            ruleset_id=ruleset_id,
        )
        if list(input_value.result_ids) != result_ids_value:
            raise ValueError("authorized result IDs are not canonical")
        selectors = _recommendation_selectors(
            values.get("authorized_result_selectors")
        )
        evaluated_paths = _recommendation_text_list(
            values.get("evaluated_value_paths")
        )
        evidence = _recommendation_threshold_evidence(
            values.get("threshold_evidence"),
            result_ids=frozenset(input_value.result_ids),
            selectors=selectors,
        )
        if tuple(item["value_path"] for item in evidence) != evaluated_paths:
            raise ValueError("evaluated recommendation paths are inconsistent")
        reason_codes = _recommendation_text_list(values.get("reason_codes"))
        resolution_issues = _recommendation_text_list(
            values.get("resolution_issues")
        )
        if checked.warnings != [*reason_codes, *resolution_issues]:
            raise ValueError("recommendation warnings are inconsistent")
        failed_codes = tuple(
            dict.fromkeys(
                cast(str, item["recheck_reason_code"])
                for item in evidence
                if item["comparison_passed"] is False
            )
        )
        outcome = values.get("recommendation")
        if outcome in {"ADOPTABLE", "RECHECK_REQUIRED"} and not evidence:
            raise ValueError("resolved recommendation requires rule evidence")
        if resolution_issues:
            valid_outcome = outcome == "UNRESOLVED" and bool(reason_codes)
        elif failed_codes:
            valid_outcome = (
                outcome == "RECHECK_REQUIRED" and reason_codes == failed_codes
            )
        else:
            valid_outcome = outcome == "ADOPTABLE" and not reason_codes
        if not valid_outcome:
            raise ValueError("recommendation outcome is inconsistent")
    except (AttributeError, TypeError, ValueError):
        return _engineering_recommendation_rejected(
            "ENGINEERING_RECOMMENDATION_RESULT_CONTRACT_MISMATCH"
        )
    if (
        checked.tool_version != ENGINEERING_RECOMMENDATION_TOOL_VERSION
        or checked.model_version != ENGINEERING_RECOMMENDATION_MODEL_VERSION
        or checked.feature_version != ENGINEERING_RECOMMENDATION_FEATURE_VERSION
        or checked.input_hash
        != sha256_canonical(input_value.model_dump(mode="json"))
        or checked.uncertainty is not None
    ):
        return _engineering_recommendation_rejected(
            "ENGINEERING_RECOMMENDATION_RESULT_CONTRACT_MISMATCH",
            route_id=ruleset_id,
        )
    return AuditedResultAuthorization(
        allowed=True,
        route_id=ruleset_id,
        activation_status="REVIEWED_RULESET",
        evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
        supported_domain=_ENGINEERING_RECOMMENDATION_DOMAIN,
        rejection_reason=None,
    )


def _engineering_recommendation_rejected(
    reason: str,
    *,
    route_id: str = "unresolved-engineering-recommendation-ruleset",
) -> AuditedResultAuthorization:
    return _rejected(
        reason,
        route_id=route_id,
        activation_status="NOT_ACTIVATED",
        evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
        supported_domain=_ENGINEERING_RECOMMENDATION_DOMAIN,
    )


def _recommendation_selectors(value: object) -> frozenset[tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("recommendation selectors are invalid")
    selectors: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "result_tool_name",
            "result_tool_version",
        }:
            raise ValueError("recommendation selector is invalid")
        selectors.append(
            (
                _safe_recommendation_text(item.get("result_tool_name")),
                _safe_recommendation_text(item.get("result_tool_version")),
            )
        )
    if len(selectors) != len(set(selectors)):
        raise ValueError("recommendation selectors are duplicated")
    return frozenset(selectors)


def _recommendation_threshold_evidence(
    value: object,
    *,
    result_ids: frozenset[str],
    selectors: frozenset[tuple[str, str]],
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError("recommendation threshold evidence is invalid")
    checked: list[Mapping[str, object]] = []
    rule_ids: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "rule_id",
            "result_id",
            "result_tool_name",
            "result_tool_version",
            "value_path",
            "comparator",
            "threshold",
            "actual_value",
            "comparison_passed",
            "recheck_reason_code",
        }:
            raise ValueError("recommendation threshold entry is invalid")
        rule_id = _safe_recommendation_text(item.get("rule_id"))
        result_id = _safe_recommendation_text(item.get("result_id"))
        selector = (
            _safe_recommendation_text(item.get("result_tool_name")),
            _safe_recommendation_text(item.get("result_tool_version")),
        )
        value_path = _safe_recommendation_text(item.get("value_path"))
        comparator = _safe_recommendation_text(item.get("comparator"))
        threshold = _finite_recommendation_number(item.get("threshold"))
        actual = _finite_recommendation_number(item.get("actual_value"))
        comparison_passed = item.get("comparison_passed")
        _safe_recommendation_text(item.get("recheck_reason_code"))
        if (
            rule_id in rule_ids
            or result_id not in result_ids
            or selector not in selectors
            or not value_path.startswith("values.")
            or not isinstance(comparison_passed, bool)
            or comparison_passed
            is not _recommendation_compare(actual, comparator, threshold)
        ):
            raise ValueError("recommendation threshold evidence is inconsistent")
        rule_ids.add(rule_id)
        checked.append(item)
    return tuple(checked)


def _recommendation_compare(actual: float, comparator: str, threshold: float) -> bool:
    comparisons = {
        "LT": actual < threshold,
        "LTE": actual <= threshold,
        "EQ": actual == threshold,
        "NE": actual != threshold,
        "GTE": actual >= threshold,
        "GT": actual > threshold,
    }
    try:
        return comparisons[comparator]
    except KeyError as exc:
        raise ValueError("recommendation comparator is invalid") from exc


def _finite_recommendation_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("recommendation numeric evidence is invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("recommendation numeric evidence is not finite")
    return number


def _recommendation_text_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("recommendation text evidence is invalid")
    normalized = tuple(_safe_recommendation_text(item) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("recommendation text evidence is duplicated")
    return normalized


def _safe_recommendation_text(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 500
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("recommendation text evidence is invalid")
    return normalized


def _matches_recommendation_report(
    report: ToolResult,
    upstream: ToolResult,
) -> bool:
    markdown = report.values.get("markdown")
    try:
        expected_markdown = render_engineering_recommendation_markdown(upstream)
    except (TypeError, ValueError):
        return False
    return (
        report.model_version == RECOMMENDATION_REPORT_RENDERER_VERSION
        and report.data_version == upstream.data_version
        and report.feature_version == upstream.feature_version
        and report.input_hash
        == sha256_canonical(
            {
                "schema_version": "engineering-recommendation-report-input-v1",
                "analysis_result_id": upstream.result_id,
                "analysis_input_hash": upstream.input_hash,
            }
        )
        and report.values.get("report_id") == report.result_id
        and report.values.get("rendering_version")
        == RECOMMENDATION_REPORT_RENDERER_VERSION
        and markdown == expected_markdown
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
