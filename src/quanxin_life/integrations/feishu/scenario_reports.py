"""Ledger-only report inputs for audited BLAST scenario results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.core import ToolResult
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.audited_report import (
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
)

from .workflow import FeishuAnalysisTask


class FeishuScenarioReportJob(Protocol):
    @property
    def task_type(self) -> FeishuAnalysisTask: ...


_TASK_RESULT_CONTRACTS = {
    FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: (
        StandardToolName.COMPARE_OPERATION_SCENARIOS.value,
        COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    ),
    FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME: (
        StandardToolName.PROJECT_STORAGE_LIFETIME.value,
        PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
    ),
}
_REQUIRED_NUMERIC_FIELDS = (
    "final_natural_year",
    "final_equivalent_full_cycles",
    "final_soh",
)
_MILESTONE_YEARS = ("15", "20", "25")


class FeishuScenarioReportResultFactory:
    """Invoke the existing audited report tool using fixed ToolResult paths."""

    def __init__(self, service: ToolInvocationService) -> None:
        if service.audit_ledger is None:
            raise ValueError("scenario report factory requires an audit ledger")
        self._service = service

    def __call__(
        self,
        job: FeishuScenarioReportJob,
        result: ToolResult,
    ) -> ToolResult:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        expected = _TASK_RESULT_CONTRACTS.get(job.task_type)
        if expected is None or (checked.tool_name, checked.tool_version) != expected:
            raise ValueError("scenario report result does not match the Feishu task")
        references = tuple(
            NumericEvidenceReference(result_id=checked.result_id, json_path=path)
            for path in _numeric_paths(checked, task=job.task_type)
        )
        report_input = GenerateAuditedReportToolInput(
            report_kind=ReportKind.STORAGE_LIFETIME_SCENARIO,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.SCENARIO_PROJECTION,
                    numeric_evidence=references,
                ),
            ),
            upstream_result_ids=(checked.result_id,),
        )
        return self._service.invoke_for_agent(
            ToolInvocation(
                tool_name=StandardToolName.GENERATE_AUDITED_REPORT,
                input_value=report_input.model_dump(mode="json"),
            ),
            allowed_tool_names={StandardToolName.GENERATE_AUDITED_REPORT},
        )


def _numeric_paths(
    result: ToolResult,
    *,
    task: FeishuAnalysisTask,
) -> list[str]:
    artifact = _mapping(result.values.get("artifact"), label="scenario artifact")
    if artifact.get("status") != "COMPLETED":
        raise ValueError("scenario report requires a completed ToolResult")
    if task is FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME:
        projections = [
            (
                "values.artifact.projection",
                _mapping(artifact.get("projection"), label="scenario projection"),
            )
        ]
    else:
        baseline = _mapping(artifact.get("baseline"), label="baseline projection")
        comparisons = artifact.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError("scenario comparison result has no comparison projections")
        projections = [("values.artifact.baseline", baseline)]
        projections.extend(
            (
                f"values.artifact.comparisons.{index}",
                _mapping(item, label="comparison projection"),
            )
            for index, item in enumerate(comparisons)
        )

    paths: list[str] = []
    for prefix, projection in projections:
        for field_name in _REQUIRED_NUMERIC_FIELDS:
            if field_name not in projection:
                raise ValueError("scenario projection is missing a report scalar")
            paths.append(f"{prefix}.{field_name}")
        milestones = projection.get("milestone_soh")
        if not isinstance(milestones, Mapping):
            raise ValueError("scenario projection milestones are invalid")
        paths.extend(
            f"{prefix}.milestone_soh.{year}"
            for year in _MILESTONE_YEARS
            if year in milestones
        )
        eol = _mapping(projection.get("eol"), label="scenario EOL outcome")
        if eol.get("status") == "REACHED":
            if eol.get("natural_year") is None or eol.get("equivalent_full_cycles") is None:
                raise ValueError("reached scenario EOL outcome is incomplete")
            paths.extend(
                (
                    f"{prefix}.eol.natural_year",
                    f"{prefix}.eol.equivalent_full_cycles",
                )
            )
    return paths


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is invalid")
    return value


__all__ = [
    "FeishuScenarioReportJob",
    "FeishuScenarioReportResultFactory",
]
