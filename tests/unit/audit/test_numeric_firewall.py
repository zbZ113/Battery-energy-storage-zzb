from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import (
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)


def _result(*, values: dict[str, object] | None = None) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="life-tool-v1",
        model_version="xgboost-v1",
        data_version="matr-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"cell_id": "cell-a"}),
        values=values or {"lifetime": {"predicted_eol_cycle": 333.0}},
        provenance=[
            ProvenanceRecord(
                source_id="audit-test-source",
                source_kind=SourceKind.PREDICTED,
                uri="test://audit/result",
                sha256=sha256_canonical({"audit": "source"}),
                description="Audit firewall test provenance",
                created_at=datetime(2026, 7, 13, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def test_numeric_evidence_must_exactly_match_a_registered_tool_result_value() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger, NumericEvidence

    result = _result()
    ledger = AuditLedger((result,))

    evidence = NumericEvidence(
        result_id=result.result_id,
        json_path="values.lifetime.predicted_eol_cycle",
        reported_value=333.0,
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
    )

    resolved = ledger.verify_numeric_evidence(evidence)
    assert resolved == 333.0


def test_numeric_firewall_resolves_only_explicit_nonnegative_list_indices() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger

    result = _result(
        values={
            "artifact": {
                "comparisons": [
                    {"final_soh": 0.91},
                    {"final_soh": 0.87},
                ]
            }
        }
    )
    ledger = AuditLedger((result,))

    assert (
        ledger.resolve_numeric_value(
            result.result_id,
            "values.artifact.comparisons.1.final_soh",
        )
        == 0.87
    )
    for path in (
        "values.artifact.comparisons.-1.final_soh",
        "values.artifact.comparisons.one.final_soh",
        "values.artifact.comparisons.2.final_soh",
    ):
        with pytest.raises(ValueError, match="does not exist"):
            ledger.resolve_numeric_value(result.result_id, path)


def test_audit_ledger_resolves_only_registered_results_as_detached_contract_values() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger

    result = _result()
    ledger = AuditLedger((result,))

    resolved = ledger.resolve_registered_result(result.result_id)
    assert resolved == result

    resolved.values["lifetime"]["predicted_eol_cycle"] = 999.0  # type: ignore[index]
    assert ledger.resolve_registered_result(result.result_id).values["lifetime"] == {
        "predicted_eol_cycle": 333.0
    }

    with pytest.raises(ValueError, match="not registered"):
        ledger.resolve_registered_result(str(uuid4()))

    with pytest.raises(ValueError, match="UUID"):
        ledger.resolve_registered_result("not-a-uuid")


def test_numeric_firewall_rejects_mismatched_unknown_or_non_numeric_evidence() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger, NumericEvidence

    result = _result(values={"status": "completed", "lifetime": {"eol": 333.0}})
    ledger = AuditLedger((result,))

    with pytest.raises(ValueError, match="does not match"):
        ledger.verify_numeric_evidence(
            NumericEvidence(
                result_id=result.result_id,
                json_path="values.lifetime.eol",
                reported_value=334.0,
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
            )
        )
    with pytest.raises(ValueError, match="not registered"):
        ledger.verify_numeric_evidence(
            NumericEvidence(
                result_id=str(uuid4()),
                json_path="values.lifetime.eol",
                reported_value=333.0,
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
            )
        )
    with pytest.raises(ValueError, match="numeric"):
        ledger.verify_numeric_evidence(
            NumericEvidence(
                result_id=result.result_id,
                json_path="values.status",
                reported_value=1.0,
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
            )
        )


def test_audit_ledger_revalidates_tool_results_before_accepting_them() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger

    tampered = ToolResult.model_construct(
        result_id="not-a-uuid",
        tool_name="predict_cycle_life",
        tool_version="life-tool-v1",
        model_version="xgboost-v1",
        data_version="matr-v1",
        feature_version="early-cycle-v1",
        input_hash="0" * 64,
        values={"lifetime": {"predicted_eol_cycle": 333.0}},
        provenance=[],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="ToolResult"):
        AuditLedger((tampered,))


def test_empty_audit_ledger_accepts_append_only_registered_results() -> None:
    from quanxin_life.audit.numeric_firewall import AuditLedger

    result = _result(values={"lifetime": {"eol": 333.0}})
    ledger = AuditLedger()

    registered = ledger.register_result(result)

    assert registered == result
    assert registered is not result
    assert ledger.resolve_registered_result(result.result_id) == result

    with pytest.raises(ValueError, match="duplicate ToolResult result_id"):
        ledger.register_result(result)


def test_audit_ledger_duplicate_registration_is_atomic_across_threads() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from quanxin_life.audit import AuditLedger, DuplicateAuditResultError

    result = _result(values={"lifetime": {"eol": 333.0}})
    ledger = AuditLedger()

    def register() -> str:
        try:
            ledger.register_result(result)
        except DuplicateAuditResultError:
            return "duplicate"
        return "registered"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = tuple(pool.map(lambda _: register(), range(32)))

    assert outcomes.count("registered") == 1
    assert outcomes.count("duplicate") == 31


def test_audit_ledger_restores_identical_result_but_rejects_conflicting_content() -> None:
    from quanxin_life.audit import AuditLedger, DuplicateAuditResultError

    result = _result(values={"lifetime": {"eol": 333.0}})
    ledger = AuditLedger((result,))

    assert ledger.ensure_result(result) == result

    conflicting = result.model_copy(update={"values": {"lifetime": {"eol": 334.0}}})
    with pytest.raises(DuplicateAuditResultError, match="conflicting ToolResult"):
        ledger.ensure_result(conflicting)


def test_numeric_evidence_rejects_boolean_reported_values() -> None:
    from quanxin_life.audit.numeric_firewall import NumericEvidence

    with pytest.raises(ValueError, match="finite numeric"):
        NumericEvidence(
            result_id=str(uuid4()),
            json_path="values.lifetime.predicted_eol_cycle",
            reported_value=True,
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
        )
