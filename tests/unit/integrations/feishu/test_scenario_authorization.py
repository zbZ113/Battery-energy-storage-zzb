from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.audit import AuditLedger
from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu import (
    AuditedScenarioResultAuthorizer,
    BlastScenarioResultAuthorizer,
)
from quanxin_life.scenarios import (
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSegment,
    VerifiedScenarioContext,
)
from quanxin_life.tools.audited_report import (
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
    execute_generate_audited_report_tool,
)
from quanxin_life.tools.blast_scenarios import (
    CompareOperationScenariosToolInput,
    execute_compare_operation_scenarios_tool,
)

NOW = datetime(2026, 8, 11, 14, 0, tzinfo=UTC)


class _Resolver:
    def __init__(self, context: VerifiedScenarioContext) -> None:
        self._context = context

    def resolve_scenario_context(self, scenario_context_id: str) -> VerifiedScenarioContext:
        if scenario_context_id != self._context.scenario_context_id:
            raise ValueError("unknown context")
        return self._context


def _scenario(scenario_id: str) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=1,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="year-1",
                start_year=0,
                end_year=1,
                temperature_c=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def _result() -> ToolResult:
    context_id = str(uuid4())
    cell = ScenarioCellDescriptor(
        chemistry="LFP/graphite",
        nominal_capacity_ah=250.0,
        cell_format="prismatic",
    )
    context = VerifiedScenarioContext(
        scenario_context_id=context_id,
        cell=cell,
        trusted_reference_use=False,
        data_version="scenario-data-v1",
        provenance=(
            ProvenanceRecord(
                source_id="scenario-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://scenario/source",
                sha256="a" * 64,
                description="Verified scenario source.",
                created_at=NOW,
            ),
        ),
    )
    return execute_compare_operation_scenarios_tool(
        CompareOperationScenariosToolInput(
            run_id=context_id,
            scenario_context_id=context_id,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            cell=cell,
            baseline=_scenario("baseline"),
            comparisons=(_scenario("comparison"),),
        ),
        context_resolver=_Resolver(context),
        allow_candidate_execution=True,
        clock=lambda: NOW,
    )


def test_scenario_authorizer_is_fail_closed_for_candidate_results_by_default() -> None:
    authorization = BlastScenarioResultAuthorizer().authorize(_result())

    assert authorization.allowed is False
    assert authorization.rejection_reason == "CANDIDATE_SCENARIO_RESULT_NOT_APPROVED"
    assert authorization.activation_status == "REGISTERED_CANDIDATE"


def test_scenario_authorizer_allows_only_manifest_bound_candidate_results() -> None:
    result = _result()

    authorization = BlastScenarioResultAuthorizer(
        allow_candidate_results=True
    ).authorize(result)

    assert authorization.allowed is True
    assert authorization.route_id == result.values["artifact"]["route_id"]
    assert authorization.activation_status == "REGISTERED_CANDIDATE"
    assert authorization.evidence_level is EvidenceLevel.PHYSICS_REFERENCE
    assert "non-product-specific" in authorization.supported_domain


def test_scenario_authorizer_rejects_route_or_model_version_tampering() -> None:
    result = _result().model_copy(update={"model_version": "tampered-route-v1"})

    authorization = BlastScenarioResultAuthorizer(
        allow_candidate_results=True
    ).authorize(result)

    assert authorization.allowed is False
    assert authorization.rejection_reason == "SCENARIO_RESULT_MANIFEST_MISMATCH"


def test_scenario_authorizer_propagates_an_audited_tool_rejection_without_values() -> None:
    completed = _result()
    rejected = completed.model_copy(
        update={
            "values": {
                "artifact_type": completed.values["artifact_type"],
                "artifact": {
                    "status": "REJECTED",
                    "run_id": completed.values["artifact"]["run_id"],
                    "result_id": completed.result_id,
                    "scenario_context_id": completed.values["artifact"][
                        "scenario_context_id"
                    ],
                    "route_id": completed.values["artifact"]["route_id"],
                    "evidence_level": EvidenceLevel.PHYSICS_REFERENCE.value,
                    "rejection_reasons": ["SCENARIO_OUTSIDE_SUPPORTED_RANGE"],
                },
            },
            "warnings": ["SCENARIO_OUTSIDE_SUPPORTED_RANGE"],
        }
    )

    authorization = BlastScenarioResultAuthorizer(
        allow_candidate_results=True
    ).authorize(rejected)

    assert authorization.allowed is False
    assert authorization.rejection_reason == "SCENARIO_OUTSIDE_SUPPORTED_RANGE"


def _report(ledger: AuditLedger, scenario_result: ToolResult) -> ToolResult:
    projection = scenario_result.values["artifact"]["baseline"]
    assert projection["final_soh"] is not None
    return execute_generate_audited_report_tool(
        GenerateAuditedReportToolInput(
            report_kind=ReportKind.STORAGE_LIFETIME_SCENARIO,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.SCENARIO_PROJECTION,
                    numeric_evidence=(
                        NumericEvidenceReference(
                            result_id=scenario_result.result_id,
                            json_path="values.artifact.baseline.final_soh",
                        ),
                    ),
                ),
            ),
            upstream_result_ids=(scenario_result.result_id,),
        ),
        audit_ledger=ledger,
        clock=lambda: NOW,
    )


def test_audited_scenario_authorizer_applies_the_same_gate_to_reports() -> None:
    ledger = AuditLedger()
    scenario_result = ledger.register_result(_result())
    report = ledger.register_result(_report(ledger, scenario_result))

    blocked = AuditedScenarioResultAuthorizer(
        result_resolver=ledger,
        allow_candidate_results=False,
    )
    allowed = AuditedScenarioResultAuthorizer(
        result_resolver=ledger,
        allow_candidate_results=True,
    )

    assert blocked.authorize(scenario_result).allowed is False
    assert blocked.authorize(report).allowed is False
    assert allowed.authorize(scenario_result).allowed is True
    report_authorization = allowed.authorize(report)
    assert report_authorization.allowed is True
    assert report_authorization.evidence_level is EvidenceLevel.PHYSICS_REFERENCE


def test_audited_scenario_authorizer_rejects_a_report_with_unbound_upstream() -> None:
    ledger = AuditLedger()
    scenario_result = ledger.register_result(_result())
    report = _report(ledger, scenario_result).model_copy(
        update={"values": {"upstream_result_ids": [str(uuid4())]}}
    )

    authorization = AuditedScenarioResultAuthorizer(
        result_resolver=ledger,
        allow_candidate_results=True,
    ).authorize(report)

    assert authorization.allowed is False
    assert authorization.rejection_reason == "AUDITED_REPORT_UPSTREAM_INVALID"
