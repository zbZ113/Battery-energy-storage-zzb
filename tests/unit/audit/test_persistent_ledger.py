from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical


def _result(*, value: float = 321.0) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="life-tool-v1",
        model_version="variance-v1",
        data_version="verified-data-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"cell_id": "cell-a"}),
        values={"lifetime": {"predicted_eol_cycle": value}},
        provenance=(
            ProvenanceRecord(
                source_id="persistent-ledger-test",
                source_kind=SourceKind.PREDICTED,
                uri="test://persistent-ledger/result",
                sha256=sha256_canonical({"source": "persistent-ledger-test"}),
                description="Persistent audit ledger test evidence",
                created_at=datetime(2026, 7, 15, tzinfo=UTC),
            ),
        ),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def test_jsonl_audit_ledger_survives_process_restart(tmp_path: Path) -> None:
    from quanxin_life.audit import JsonlAuditLedger

    journal = tmp_path / "audit" / "tool-results.jsonl"
    result = _result()

    first = JsonlAuditLedger(journal)
    assert first.register_result(result) == result

    restored = JsonlAuditLedger(journal)
    assert restored.resolve_registered_result(result.result_id) == result
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_audit_ledger_does_not_append_idempotent_restore(tmp_path: Path) -> None:
    from quanxin_life.audit import JsonlAuditLedger

    journal = tmp_path / "tool-results.jsonl"
    result = _result()
    ledger = JsonlAuditLedger(journal)

    ledger.ensure_result(result)
    ledger.ensure_result(result)

    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_audit_ledger_rejects_conflicting_result_without_mutating_journal(
    tmp_path: Path,
) -> None:
    from quanxin_life.audit import DuplicateAuditResultError, JsonlAuditLedger

    journal = tmp_path / "tool-results.jsonl"
    result = _result()
    ledger = JsonlAuditLedger(journal)
    ledger.register_result(result)
    original = journal.read_bytes()

    conflicting = result.model_copy(
        update={"values": {"lifetime": {"predicted_eol_cycle": 999.0}}}
    )
    with pytest.raises(DuplicateAuditResultError, match="conflicting ToolResult"):
        ledger.ensure_result(conflicting)

    assert journal.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        "not-json\n",
        '{"result_id":"not-a-valid-tool-result"}\n',
        "\n",
    ],
)
def test_jsonl_audit_ledger_fails_closed_on_corrupt_journal(
    tmp_path: Path,
    payload: str,
) -> None:
    from quanxin_life.audit import InvalidAuditResultError, JsonlAuditLedger

    journal = tmp_path / "tool-results.jsonl"
    journal.write_text(payload, encoding="utf-8")

    with pytest.raises(InvalidAuditResultError, match="audit journal"):
        JsonlAuditLedger(journal)


def test_jsonl_audit_ledger_rejects_symlink_journal(tmp_path: Path) -> None:
    from quanxin_life.audit import InvalidAuditResultError, JsonlAuditLedger

    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    try:
        journal.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable in this environment")

    with pytest.raises(InvalidAuditResultError, match="symbolic link"):
        JsonlAuditLedger(journal)
