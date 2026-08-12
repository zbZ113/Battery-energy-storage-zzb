from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.scenario_reports import (
    FeishuScenarioReportResultFactory,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.tools import ToolRegistry
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


class _Job:
    task_type = FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS


def _projection(*, scenario_id: str, final_soh: float) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "scenario_version": f"{scenario_id}-v1",
        "final_natural_year": 20.0,
        "final_equivalent_full_cycles": 6000.0,
        "final_soh": final_soh,
        "milestone_soh": {"15": final_soh + 0.03, "20": final_soh},
        "eol": {
            "status": "REACHED",
            "natural_year": 18.5,
            "equivalent_full_cycles": 5550.0,
        },
    }


def _result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
        tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        model_version="blast-lite-route-v1",
        data_version="scenario-data-v1",
        feature_version="blast-scenario-v1",
        input_hash="a" * 64,
        values={
            "artifact": {
                "status": "COMPLETED",
                "baseline": _projection(scenario_id="baseline", final_soh=0.88),
                "comparisons": [
                    _projection(scenario_id="warmer", final_soh=0.82)
                ],
            }
        },
        warnings=["CANDIDATE_ROUTE_RESEARCH_USE_ONLY"],
        provenance=[
            ProvenanceRecord(
                source_id="scenario-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://scenario/source",
                sha256="b" * 64,
                description="Verified scenario report fixture",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_scenario_report_factory_resolves_baseline_and_comparison_values_from_ledger() -> None:
    analysis = _result()
    ledger = AuditLedger((analysis,))
    registry = ToolRegistry()
    register_generate_audited_report_tool(
        registry,
        audit_ledger=ledger,
        clock=lambda: NOW,
    )
    factory = FeishuScenarioReportResultFactory(
        ToolInvocationService(registry=registry, audit_ledger=ledger)
    )

    report = factory(_Job(), analysis)  # type: ignore[arg-type]

    assert ledger.resolve_registered_result(report.result_id) == report
    markdown = report.values["markdown"]
    assert "values.artifact.baseline.final_soh" in markdown
    assert "values.artifact.comparisons.0.final_soh" in markdown
    assert "values.artifact.comparisons.0.eol.natural_year" in markdown
    assert "PHYSICS_REFERENCE" in markdown
    assert "confidence interval" in markdown
