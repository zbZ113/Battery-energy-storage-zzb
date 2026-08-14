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
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
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
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
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


def _data_quality_result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="validate_battery_data",
        tool_version=DATA_QUALITY_TOOL_VERSION,
        model_version=DATA_QUALITY_MODEL_VERSION,
        data_version="registered-batch-v1",
        feature_version="canonical-csv-v1",
        input_hash="d" * 64,
        values={
            "dataset_id": "batch-safe",
            "blocked": True,
            "quality_score": 0.0,
            "issue_count": 1,
            "issues": [
                {
                    "code": "MISSING_REQUIRED_CYCLE",
                    "severity": "blocking",
                    "message": "Required cycle is missing.",
                    "cell_id": "cell-1",
                    "cycle_index": None,
                }
            ],
        },
        uncertainty=None,
        warnings=["MISSING_REQUIRED_CYCLE"],
        provenance=[
            ProvenanceRecord(
                source_id="batch-safe",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:batch-safe",
                sha256="e" * 64,
                description="Registered canonical CSV batch.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_audited_authorizer_accepts_only_consistent_data_quality_results() -> None:
    result = _data_quality_result()
    authorizer = AuditedScenarioResultAuthorizer(
        result_resolver=AuditLedger((result,))
    )

    allowed = authorizer.authorize(result)
    tampered = authorizer.authorize(
        result.model_copy(update={"model_version": "unreviewed-validator"})
    )

    assert allowed.allowed is True
    assert allowed.evidence_level is EvidenceLevel.DATA_DIRECT
    assert tampered.allowed is False
    assert tampered.rejection_reason == "DATA_QUALITY_RESULT_CONTRACT_MISMATCH"


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


def _advanced_rul_result() -> ToolResult:
    artifact_id = str(uuid4())
    decision_event_id = str(uuid4())
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version=ADVANCED_RUL_PREDICTION_TOOL_VERSION,
        model_version="cyclepatch-direct-cutoff-20-seed-38",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="c" * 64,
        values={
            "artifact_type": ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": str(uuid4()),
                "dataset_id": "MATR",
                "cell_id": "MATR_b3c34",
                "cutoff_cycle": 20,
                "cycle_life_prediction": {
                    "dataset_id": "MATR",
                    "cell_id": "MATR_b3c34",
                    "cutoff_cycle": 20,
                    "target": "matr_official_cycle_life",
                    "predicted_cycle": 720.0,
                    "observed_cycle": None,
                    "right_censored": True,
                    "feature_version": "cyclepatch-multichannel-v1",
                    "split_version": "matr-cell-split-v1",
                    "model_version": "cyclepatch-direct-cutoff-20-seed-38",
                    "data_version": "matr-three-batch-v1",
                },
                "derived_remaining_cycles": 700.0,
                "upstream_result_id": str(uuid4()),
                "raw_sequence_input_sha256": "1" * 64,
                "transform_config_sha256": "2" * 64,
                "source_manifest_hash": "3" * 64,
                "split_version": "matr-cell-split-v1",
                "normalization_statistics_sha256": "4" * 64,
                "task": "RUL",
                "route_role": "DEFAULT",
                "output_target": "matr_official_cycle_life",
                "artifact_kind": "cyclepatch_direct",
                "artifact_id": artifact_id,
                "artifact_manifest_sha256": "5" * 64,
                "decision_event_id": decision_event_id,
                "ledger_sequence_number": 7,
                "ledger_head_sha256": "6" * 64,
            },
        },
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id=f"advanced-model-{artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{artifact_id}",
                sha256="5" * 64,
                description="Verified active Advanced RUL runtime used for inference",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_audited_authorizer_allows_formal_advanced_rul_and_its_report() -> None:
    ledger = AuditLedger()
    result = ledger.register_result(_advanced_rul_result())
    report = ledger.register_result(
        execute_generate_audited_report_tool(
            GenerateAuditedReportToolInput(
                report_kind=ReportKind.LIFETIME_DECISION,
                claims=(
                    AuditedReportClaimReference(
                        claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                        numeric_evidence=(
                            NumericEvidenceReference(
                                result_id=result.result_id,
                                json_path=(
                                    "values.artifact.cycle_life_prediction."
                                    "predicted_cycle"
                                ),
                            ),
                            NumericEvidenceReference(
                                result_id=result.result_id,
                                json_path="values.artifact.derived_remaining_cycles",
                            ),
                        ),
                    ),
                ),
                upstream_result_ids=(result.result_id,),
            ),
            audit_ledger=ledger,
            clock=lambda: NOW,
        )
    )
    authorizer = AuditedScenarioResultAuthorizer(
        result_resolver=ledger,
        allow_candidate_results=False,
    )

    result_authorization = authorizer.authorize(result)
    report_authorization = authorizer.authorize(report)

    assert result_authorization.allowed is True
    assert result_authorization.activation_status == "ACTIVE"
    assert result_authorization.evidence_level is EvidenceLevel.MODEL_INFERENCE
    assert result_authorization.route_id == result.values["artifact"]["decision_event_id"]
    assert report_authorization == result_authorization


def test_audited_authorizer_rejects_incomplete_advanced_rul_evidence() -> None:
    result = _advanced_rul_result().model_copy(
        update={"values": {"artifact_type": ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE}}
    )

    authorization = AuditedScenarioResultAuthorizer(
        result_resolver=AuditLedger((result,)),
    ).authorize(result)

    assert authorization.allowed is False
    assert authorization.rejection_reason == "ADVANCED_RUL_RESULT_CONTRACT_MISMATCH"


def _advanced_soh_result() -> ToolResult:
    artifact_id = str(uuid4())
    decision_event_id = str(uuid4())
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_soh_trajectory",
        tool_version=ADVANCED_SOH_PREDICTION_TOOL_VERSION,
        model_version="hybridpatch-v2-cutoff-50-seed-38",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="7" * 64,
        values={
            "artifact_type": ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": str(uuid4()),
                "dataset_id": "MATR",
                "cell_id": "MATR_b3c34",
                "cutoff_cycle": 50,
                "prediction_cycles": [51, 500],
                "predicted_soh": [0.99, 0.87],
                "horizon_end_cycle": 500,
                "upstream_result_id": str(uuid4()),
                "raw_sequence_input_sha256": "1" * 64,
                "transform_config_sha256": "2" * 64,
                "source_manifest_hash": "3" * 64,
                "split_version": "matr-cell-split-v1",
                "normalization_statistics_sha256": "4" * 64,
                "task": "SOH",
                "route_role": "MEAN_ACCURACY",
                "output_target": "soh_trajectory",
                "artifact_kind": "hybridpatch_v2",
                "artifact_id": artifact_id,
                "artifact_manifest_sha256": "5" * 64,
                "decision_event_id": decision_event_id,
                "ledger_sequence_number": 7,
                "ledger_head_sha256": "6" * 64,
            },
        },
        uncertainty={
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        },
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id=f"advanced-model-{artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{artifact_id}",
                sha256="5" * 64,
                description="Verified active Advanced SOH runtime used for inference",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_audited_authorizer_accepts_only_complete_finite_soh_evidence() -> None:
    result = _advanced_soh_result()
    authorizer = AuditedScenarioResultAuthorizer(
        result_resolver=AuditLedger((result,)),
    )

    allowed = authorizer.authorize(result)
    tampered = authorizer.authorize(
        result.model_copy(
            update={
                "values": {
                    **result.values,
                    "artifact": {
                        **result.values["artifact"],
                        "prediction_cycles": [51, 499],
                    },
                }
            }
        )
    )

    assert allowed.allowed is True
    assert allowed.activation_status == "ACTIVE"
    assert allowed.evidence_level is EvidenceLevel.MODEL_INFERENCE
    assert tampered.allowed is False
    assert tampered.rejection_reason == "ADVANCED_SOH_RESULT_CONTRACT_MISMATCH"
