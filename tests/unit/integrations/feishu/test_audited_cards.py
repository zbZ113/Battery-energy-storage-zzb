from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.cards import (
    AuditedCardBuilder,
    AuditedCardError,
    AuditedResultAuthorization,
    FeishuCardStatus,
    build_status_card,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    execute_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="uploaded-csv",
            source_kind=SourceKind.OBSERVED,
            uri="feishu://message/om_source/resource/file_source",
            sha256="a" * 64,
            description="Verified Feishu CSV attachment",
            created_at=NOW,
        ),
    )


class _Authorizer:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        del result
        return AuditedResultAuthorization(
            allowed=self.allowed,
            route_id="quality-validation",
            activation_status="ACTIVE" if self.allowed else "NOT_ACTIVATED",
            evidence_level=EvidenceLevel.DATA_DIRECT,
            supported_domain="validated-upload",
            rejection_reason=None if self.allowed else "MODEL_ROUTE_NOT_ACTIVATED",
        )


class _BindingVerifier:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str]] = []

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool:
        self.calls.append((run_id, result_id))
        return self.allowed


class _ScenarioAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "compare_operation_scenarios"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            activation_status="REGISTERED_CANDIDATE",
            evidence_level=EvidenceLevel.PHYSICS_REFERENCE,
            supported_domain="manifest-bounded-reference-scenario",
        )


def _builder(
    ledger: AuditLedger,
    *,
    authorizer: _Authorizer | None = None,
    binding_verifier: _BindingVerifier | None = None,
) -> AuditedCardBuilder:
    return AuditedCardBuilder(
        ledger,
        authorizer=authorizer or _Authorizer(),
        binding_verifier=binding_verifier or _BindingVerifier(),
    )


def _validation_result() -> ToolResult:
    return execute_validate_battery_data_tool(
        ValidateBatteryDataToolInput(
            records=(),
            data_version="uploaded-data-v1",
            feature_version="raw-cycle-v1",
            provenance=_provenance(),
            validated_at=NOW,
        )
    )


def test_audited_card_reads_allowlisted_values_from_registered_tool_result() -> None:
    result = _validation_result()
    ledger = AuditLedger((result,))

    card = _builder(ledger).build_result_card(
        run_id="run-safe",
        result_id=result.result_id,
    )
    rendered = json.dumps(card, ensure_ascii=False)

    assert result.result_id in rendered
    assert "values.blocked" in rendered
    assert "values.quality_score" in rendered
    assert str(result.values["quality_score"]) in rendered
    assert result.model_version in rendered
    assert result.data_version in rendered
    assert result.feature_version in rendered
    assert EvidenceLevel.DATA_DIRECT.value in rendered


def test_audited_card_public_api_rejects_caller_supplied_numeric_value() -> None:
    result = _validation_result()
    builder = _builder(AuditLedger((result,)))

    with pytest.raises(TypeError):
        builder.build_result_card(  # type: ignore[call-arg]
            run_id="run-safe",
            result_id=result.result_id,
            value=object(),
        )


def test_missing_or_unsupported_result_is_rejected() -> None:
    with pytest.raises(AuditedCardError, match="registered"):
        _builder(AuditLedger()).build_result_card(
            run_id="run-safe",
            result_id=str(uuid4()),
        )

    unsupported = _validation_result().model_copy(
        update={"result_id": str(uuid4()), "tool_version": "unknown-version"}
    )
    with pytest.raises(AuditedCardError, match="allowlisted"):
        _builder(
            AuditLedger((unsupported,)), authorizer=_Authorizer()
        ).build_result_card(run_id="run-safe", result_id=unsupported.result_id)


def test_inactive_model_result_returns_rejection_card_before_reading_values() -> None:
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="advanced-cycle-life-prediction-tool-v1",
        model_version="candidate-route-v1",
        data_version="registered-data-v1",
        feature_version="registered-feature-v1",
        input_hash="b" * 64,
        values={"untrusted_numeric_payload": object().__class__.__name__},
        uncertainty=None,
        warnings=[],
        provenance=list(_provenance()),
        created_at=NOW,
    )

    card = _builder(
        AuditLedger((result,)), authorizer=_Authorizer(allowed=False)
    ).build_result_card(run_id="run-safe", result_id=result.result_id)
    rendered = json.dumps(card, ensure_ascii=False)

    assert "MODEL_ROUTE_NOT_ACTIVATED" in rendered
    assert "untrusted_numeric_payload" not in rendered


def test_scenario_card_reads_only_fixed_scalar_summaries_and_embeds_curve() -> None:
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="compare_operation_scenarios",
        tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        model_version="blast-lite-route-v1",
        data_version="scenario-data-v1",
        feature_version="blast-scenario-v1",
        input_hash="b" * 64,
        values={
            "artifact": {
                "status": "COMPLETED",
                "baseline": {
                    "scenario_id": "baseline",
                    "scenario_version": "baseline-v1",
                    "final_natural_year": 20.0,
                    "final_equivalent_full_cycles": 6000.0,
                    "final_soh": 0.88,
                    "milestone_soh": {"15": 0.91, "20": 0.88},
                    "eol_threshold": 0.8,
                    "eol": {
                        "status": "NOT_REACHED",
                        "natural_year": None,
                        "equivalent_full_cycles": None,
                    },
                    "support": {"status": "SUPPORTED"},
                    "natural_years": [0.0, 20.0],
                    "soh": [1.0, 0.88],
                },
                "comparisons": [
                    {
                        "scenario_id": "warmer",
                        "scenario_version": "warmer-v1",
                        "final_natural_year": 20.0,
                        "final_equivalent_full_cycles": 6000.0,
                        "final_soh": 0.81,
                        "milestone_soh": {"15": 0.86, "20": 0.81},
                        "eol_threshold": 0.8,
                        "eol": {
                            "status": "REACHED",
                            "natural_year": 19.2,
                            "equivalent_full_cycles": 5760.0,
                        },
                        "support": {"status": "NEAR_BOUNDARY"},
                        "natural_years": [0.0, 20.0],
                        "soh": [1.0, 0.81],
                    }
                ],
            }
        },
        warnings=["CANDIDATE_ROUTE_RESEARCH_USE_ONLY"],
        provenance=list(_provenance()),
        created_at=NOW,
    )
    card = _builder(
        AuditLedger((result,)),
        authorizer=_ScenarioAuthorizer(),  # type: ignore[arg-type]
    ).build_result_card(
        run_id="run-safe",
        result_id=result.result_id,
        image_key="img-scenario-safe",
    )
    rendered = json.dumps(card, ensure_ascii=False)

    assert "values.artifact.baseline.final_soh" in rendered
    assert "values.artifact.comparisons.0.final_soh" in rendered
    assert "values.artifact.comparisons.0.eol.natural_year" in rendered
    assert "PHYSICS_REFERENCE" in rendered
    assert "CANDIDATE_ROUTE_RESEARCH_USE_ONLY" in rendered
    assert "img-scenario-safe" in rendered
    assert "values.artifact.baseline.soh" not in rendered
    assert "natural_years" not in rendered


def test_unbound_result_is_rejected_before_ledger_resolution() -> None:
    result = _validation_result()
    verifier = _BindingVerifier(allowed=False)

    with pytest.raises(AuditedCardError, match="not bound"):
        _builder(
            AuditLedger((result,)), binding_verifier=verifier
        ).build_result_card(run_id="run-safe", result_id=result.result_id)

    assert verifier.calls == [("run-safe", result.result_id)]


@pytest.mark.parametrize("status", tuple(FeishuCardStatus))
def test_status_cards_are_reference_only(status: FeishuCardStatus) -> None:
    rendered = json.dumps(
        build_status_card(status=status, run_id="run-safe", result_id=None),
        ensure_ascii=False,
    )

    assert "run-safe" in rendered
    assert "SOH" not in rendered
    assert "RUL" not in rendered
    assert "置信区间" not in rendered
