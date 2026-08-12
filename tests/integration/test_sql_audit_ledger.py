from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from quanxin_life.audit import (
    AuditLedgerError,
    DuplicateAuditResultError,
    SqlAuditLedger,
)
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    GlobalToolResultBindingRecord,
    ProvenanceRecordRow,
    ToolResultRecord,
)

NOW = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)


def _session_factory(tmp_path: Path) -> SessionFactory:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'global-ledger.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def _result(*, result_id: str | None = None, value: float = 0.91) -> ToolResult:
    return ToolResult(
        result_id=result_id or str(uuid4()),
        tool_name="project_storage_lifetime",
        tool_version="blast-scenario-tool-v1",
        model_version="blast-lite-reference-v1",
        data_version="scenario-context-v1",
        feature_version="operation-scenario-contract-v1",
        input_hash=sha256_canonical({"scenario_context_id": "context-1"}),
        values={
            "milestones": {"15": {"soh": value}},
            "eol": {"status": "not_reached"},
        },
        warnings=["PHYSICS_SCENARIO_LONG_HORIZON"],
        provenance=[
            ProvenanceRecord(
                source_id="source-b",
                source_kind=SourceKind.SIMULATED,
                uri="vendor://blast-lite/source-b",
                sha256="b" * 64,
                description="Pinned BLAST-Lite source B.",
                created_at=NOW,
            ),
            ProvenanceRecord(
                source_id="source-a",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:batch-1",
                sha256="a" * 64,
                description="Verified canonical batch.",
                created_at=NOW,
            ),
        ],
        created_at=NOW,
    )


def test_global_result_and_provenance_survive_ledger_restart(tmp_path: Path) -> None:
    session_factory = _session_factory(tmp_path)
    registered = SqlAuditLedger(session_factory).register_result(_result())

    restored = SqlAuditLedger(session_factory).resolve_registered_result(
        registered.result_id
    )

    assert restored == registered
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ToolResultRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ProvenanceRecordRow)) == 2
        assert (
            session.scalar(
                select(func.count()).select_from(GlobalToolResultBindingRecord)
            )
            == 1
        )


def test_global_ledger_rejects_duplicate_and_conflicting_result_ids(
    tmp_path: Path,
) -> None:
    session_factory = _session_factory(tmp_path)
    ledger = SqlAuditLedger(session_factory)
    registered = ledger.register_result(_result())

    with pytest.raises(DuplicateAuditResultError, match="duplicate ToolResult"):
        ledger.register_result(registered)

    assert ledger.ensure_result(registered) == registered
    with pytest.raises(DuplicateAuditResultError, match="conflicting ToolResult"):
        ledger.ensure_result(_result(result_id=registered.result_id, value=0.89))


def test_global_ledger_detects_legal_json_payload_tampering(tmp_path: Path) -> None:
    session_factory = _session_factory(tmp_path)
    registered = SqlAuditLedger(session_factory).register_result(_result())
    with session_factory.begin() as session:
        row = session.get(ToolResultRecord, registered.result_id)
        assert row is not None
        row.values_json = {
            "milestones": {"15": {"soh": 0.99}},
            "eol": {"status": "not_reached"},
        }

    with pytest.raises(AuditLedgerError, match="integrity"):
        SqlAuditLedger(session_factory).resolve_registered_result(
            registered.result_id
        )


def test_global_ledger_detects_provenance_tampering(tmp_path: Path) -> None:
    session_factory = _session_factory(tmp_path)
    registered = SqlAuditLedger(session_factory).register_result(_result())
    with session_factory.begin() as session:
        row = session.scalar(
            select(ProvenanceRecordRow).where(
                ProvenanceRecordRow.tool_result_id == registered.result_id
            )
        )
        assert row is not None
        row.description = "Altered provenance that still satisfies the schema."

    with pytest.raises(AuditLedgerError, match="integrity"):
        SqlAuditLedger(session_factory).resolve_registered_result(
            registered.result_id
        )


def test_api_and_worker_instances_share_numeric_resolution(tmp_path: Path) -> None:
    session_factory = _session_factory(tmp_path)
    api_ledger = SqlAuditLedger(session_factory)
    registered = api_ledger.register_result(_result(value=0.905))

    worker_ledger = SqlAuditLedger(session_factory)

    assert worker_ledger.resolve_numeric_value(
        registered.result_id,
        "values.milestones.15.soh",
    ) == pytest.approx(0.905)
