"""Fail-closed contract checks shared by Feishu and Aily delivery."""

from __future__ import annotations

from quanxin_life.core import ToolResult

from .cards import AuditedResultAuthorization, AuditedResultAuthorizer
from .workflow import FeishuAnalysisTask

_TASK_TOOL_NAMES = {
    FeishuAnalysisTask.PREDICT_CYCLE_LIFE: "predict_cycle_life",
    FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: "predict_soh_trajectory",
    FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: "compare_operation_scenarios",
    FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME: "project_storage_lifetime",
    FeishuAnalysisTask.INGEST_OBSERVED_SOH: "ingest_newly_observed_soh",
    FeishuAnalysisTask.UPDATE_TRAJECTORY: "update_cell_parameters",
    FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION: (
        "make_engineering_recommendation"
    ),
}


class FeishuDeliveryResultContractError(ValueError):
    """Raised before external delivery when audited result identities disagree."""


def validate_delivery_results(
    *,
    task: FeishuAnalysisTask,
    expected_analysis_result_id: str | None,
    expected_report_result_id: str | None,
    analysis_result: ToolResult,
    report_result: ToolResult,
    authorizer: AuditedResultAuthorizer,
) -> AuditedResultAuthorization:
    """Validate exact job, analysis, report and authorization relationships."""

    expected_tool_name = _TASK_TOOL_NAMES[task]
    if analysis_result.tool_name != expected_tool_name:
        raise FeishuDeliveryResultContractError(
            "analysis result task does not match the durable job"
        )
    if (
        expected_analysis_result_id is None
        or analysis_result.result_id != expected_analysis_result_id
    ):
        raise FeishuDeliveryResultContractError(
            "analysis result does not match the durable job slot"
        )
    if (
        expected_report_result_id is None
        or report_result.result_id != expected_report_result_id
    ):
        raise FeishuDeliveryResultContractError(
            "report result does not match the durable job slot"
        )
    if report_result.tool_name != "generate_audited_report":
        raise FeishuDeliveryResultContractError(
            "report result is not an audited report"
        )
    if report_result.values.get("upstream_result_ids") != [analysis_result.result_id]:
        raise FeishuDeliveryResultContractError(
            "report is not bound to the exact analysis result"
        )
    analysis_authorization = authorizer.authorize(analysis_result)
    if not analysis_authorization.allowed:
        raise FeishuDeliveryResultContractError(
            "analysis result is not authorized for delivery"
        )
    report_authorization = authorizer.authorize(report_result)
    if not report_authorization.allowed:
        raise FeishuDeliveryResultContractError(
            "report result is not authorized for delivery"
        )
    if (
        report_authorization.route_id != analysis_authorization.route_id
        or report_authorization.activation_status
        != analysis_authorization.activation_status
        or report_authorization.evidence_level
        is not analysis_authorization.evidence_level
        or report_authorization.supported_domain
        != analysis_authorization.supported_domain
    ):
        raise FeishuDeliveryResultContractError(
            "report authorization does not match the analysis result"
        )
    return analysis_authorization


__all__ = [
    "FeishuDeliveryResultContractError",
    "validate_delivery_results",
]
