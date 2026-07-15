from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from quanxin_life.application.ingestion import (
    CANONICAL_CYCLE_CSV_FIELDS,
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_payload(*, cell_id: str = "cell-1", last_cycle: int = 20) -> bytes:
    header = ",".join(CANONICAL_CYCLE_CSV_FIELDS)
    rows = [
        f"UPLOAD,{cell_id},1,0,0,3.1,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},1,1,1,3.2,-1,25,1.1,1.0,0.01,true,true",
        f"UPLOAD,{cell_id},{last_cycle},0,0,3.1,-1,25,1.0,0.9,0.02,true,true",
        f"UPLOAD,{cell_id},{last_cycle},1,1,3.2,-1,25,1.0,0.9,0.02,true,true",
    ]
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _registration(payload: bytes) -> CanonicalCsvBatchRegistration:
    digest = _sha256_bytes(payload)
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="cell-1",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="upload://canonical/cell-1.csv",
            source_sha256=digest,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="upload-data-v1",
        split_version="upload-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="canonical-upload-cell-1",
                source_kind=SourceKind.OBSERVED,
                uri="upload://canonical/cell-1.csv",
                sha256=digest,
                description="Canonical CSV uploaded by the test fixture",
                created_at=datetime(2026, 7, 15, tzinfo=UTC),
            ),
        ),
    )


def test_registers_verified_canonical_csv_as_resolvable_batch() -> None:
    payload = _csv_payload()
    store = InMemoryVerifiedEarlyCycleBatchStore()

    batch_id = store.register_canonical_csv(payload, registration=_registration(payload))
    resolved = store.resolve_verified_early_cycle_batch(batch_id)

    assert resolved.record_batch_id == batch_id
    assert resolved.metadata.cell_id == "cell-1"
    assert resolved.source_manifest_hash == _sha256_bytes(payload)
    assert len(resolved.records) == 4
    assert resolved.records[-1].cycle_index == 20


def test_registration_is_idempotent_and_resolver_returns_detached_batch() -> None:
    payload = _csv_payload()
    store = InMemoryVerifiedEarlyCycleBatchStore()
    registration = _registration(payload)

    first_id = store.register_canonical_csv(payload, registration=registration)
    second_id = store.register_canonical_csv(payload, registration=registration)
    first = store.resolve_verified_early_cycle_batch(first_id)
    second = store.resolve_verified_early_cycle_batch(first_id)

    assert first_id == second_id
    assert first == second
    assert first is not second


def test_rejects_payload_without_matching_observed_sha256() -> None:
    payload = _csv_payload()
    registration = _registration(payload)
    wrong = registration.model_copy(
        update={
            "provenance": (
                registration.provenance[0].model_copy(update={"sha256": "0" * 64}),
            )
        }
    )

    with pytest.raises(ValueError, match="SHA-256"):
        InMemoryVerifiedEarlyCycleBatchStore().register_canonical_csv(
            payload,
            registration=wrong,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"dataset_id,cell_id\nUPLOAD,cell-1\n", "header"),
        (_csv_payload(cell_id="other-cell"), "identity"),
        (_csv_payload(last_cycle=21), "cutoff"),
        (_csv_payload().replace(b",true,true", b",maybe,true", 1), "boolean"),
    ],
)
def test_rejects_untrusted_csv_that_breaks_the_canonical_contract(
    payload: bytes,
    message: str,
) -> None:
    registration = _registration(payload)

    with pytest.raises(ValueError, match=message):
        InMemoryVerifiedEarlyCycleBatchStore().register_canonical_csv(
            payload,
            registration=registration,
        )
