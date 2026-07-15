"""Safe canonical-CSV registration for the local competition application.

The parser accepts only the public :class:`CycleRecord` columns, validates every
row through Pydantic, binds the bytes to observed provenance, and exposes the
result through the existing trusted early-cycle batch resolver protocol.  It
does not deserialize executable artifacts or derive battery-health values.
"""

from __future__ import annotations

import csv
import hashlib
import io
from threading import RLock

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

CANONICAL_CYCLE_CSV_FIELDS = (
    "dataset_id",
    "cell_id",
    "cycle_index",
    "sample_index",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "internal_resistance_ohm",
    "diagnostic",
    "valid",
)
MAX_CANONICAL_CSV_BYTES = 25 * 1024 * 1024
_OPTIONAL_FIELDS = {
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "internal_resistance_ohm",
}
_BOOLEAN_FIELDS = {"diagnostic", "valid"}
_TRUE_VALUES = {"true", "1"}
_FALSE_VALUES = {"false", "0"}


class CanonicalCsvBatchRegistration(ContractModel):
    """Trusted metadata supplied alongside one canonical CSV upload."""

    metadata: CellMetadata
    feature_config: EarlyCycleFeatureConfig
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("data_version", "split_version")
    @classmethod
    def require_nonblank_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("registration versions must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_observed_source(self) -> CanonicalCsvBatchRegistration:
        if not any(item.source_kind is SourceKind.OBSERVED for item in self.provenance):
            raise ValueError("registration provenance must include an OBSERVED source")
        return self


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_boolean(value: str, *, field_name: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"canonical CSV boolean field {field_name!r} is invalid at row {row_number}"
    )


def _parse_canonical_records(payload: bytes) -> tuple[CycleRecord, ...]:
    if not payload:
        raise ValueError("canonical CSV payload must not be empty")
    if len(payload) > MAX_CANONICAL_CSV_BYTES:
        raise ValueError("canonical CSV payload exceeds the configured size limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("canonical CSV payload must be UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames is None or tuple(reader.fieldnames) != CANONICAL_CYCLE_CSV_FIELDS:
        raise ValueError("canonical CSV header does not match the CycleRecord contract")

    records: list[CycleRecord] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"canonical CSV row {row_number} contains extra columns")
            normalized: dict[str, object] = {}
            for field_name in CANONICAL_CYCLE_CSV_FIELDS:
                raw_value = row.get(field_name)
                if raw_value is None:
                    raise ValueError(
                        f"canonical CSV row {row_number} is missing {field_name!r}"
                    )
                if field_name in _OPTIONAL_FIELDS and not raw_value.strip():
                    normalized[field_name] = None
                elif field_name in _BOOLEAN_FIELDS:
                    normalized[field_name] = _parse_boolean(
                        raw_value,
                        field_name=field_name,
                        row_number=row_number,
                    )
                else:
                    normalized[field_name] = raw_value.strip()
            records.append(CycleRecord.model_validate(normalized))
    except csv.Error as exc:
        raise ValueError("canonical CSV syntax is invalid") from exc
    if not records:
        raise ValueError("canonical CSV must contain at least one data row")
    return tuple(records)


class InMemoryVerifiedEarlyCycleBatchStore:
    """Thread-safe resolver for verified upload batches used by the MVP app."""

    def __init__(self) -> None:
        self._batches: dict[str, VerifiedEarlyCycleBatch] = {}
        self._lock = RLock()

    def register_canonical_csv(
        self,
        payload: bytes,
        *,
        registration: CanonicalCsvBatchRegistration,
    ) -> str:
        """Validate and register one source-bound canonical CSV batch."""

        validated = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        payload_hash = _payload_sha256(payload)
        if validated.metadata.source_sha256 != payload_hash:
            raise ValueError("metadata source SHA-256 does not match canonical CSV payload")
        observed_hashes = {
            item.sha256
            for item in validated.provenance
            if item.source_kind is SourceKind.OBSERVED
        }
        if payload_hash not in observed_hashes:
            raise ValueError("observed provenance SHA-256 does not match canonical CSV payload")

        records = _parse_canonical_records(payload)
        batch_id = "canonical-csv-" + sha256_canonical(
            {
                "payload_sha256": payload_hash,
                "metadata": validated.metadata.model_dump(mode="json"),
                "feature_config": validated.feature_config.model_dump(mode="json"),
                "data_version": validated.data_version,
                "split_version": validated.split_version,
            }
        )
        batch = VerifiedEarlyCycleBatch(
            record_batch_id=batch_id,
            records=records,
            metadata=validated.metadata,
            feature_config=validated.feature_config,
            data_version=validated.data_version,
            split_version=validated.split_version,
            source_manifest_hash=payload_hash,
            provenance=validated.provenance,
        )
        detached = VerifiedEarlyCycleBatch.model_validate(batch.model_dump(mode="json"))
        with self._lock:
            existing = self._batches.get(batch_id)
            if existing is not None and existing != detached:
                raise ValueError("canonical CSV batch identifier collision")
            self._batches[batch_id] = detached
        return batch_id

    def resolve_verified_early_cycle_batch(
        self,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        """Resolve a detached batch or fail explicitly for an unknown ID."""

        with self._lock:
            try:
                batch = self._batches[record_batch_id]
            except KeyError as exc:
                raise KeyError(f"unknown verified record batch: {record_batch_id}") from exc
            return VerifiedEarlyCycleBatch.model_validate(batch.model_dump(mode="json"))
