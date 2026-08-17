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
import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Protocol

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
SELF_DESCRIBED_FEISHU_CSV_REGISTRATION_MODE = "SELF_DESCRIBED_FEISHU_CSV_V1"
SELF_DESCRIBED_MODEL_SUPPORT_STATUS = "UNREVIEWED_SELF_DESCRIBED"
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


def project_model_registration_rejection_code(
    registration: CanonicalCsvBatchRegistration,
) -> str | None:
    """Return the explicit model gate for an unreviewed self-described upload."""

    checked = CanonicalCsvBatchRegistration.model_validate(
        registration.model_dump(mode="json")
    )
    parameters = checked.metadata.ingestion_parameters
    if (
        parameters.get("registration_mode")
        == SELF_DESCRIBED_FEISHU_CSV_REGISTRATION_MODE
        or parameters.get("model_support_status")
        == SELF_DESCRIBED_MODEL_SUPPORT_STATUS
    ):
        return "PROJECT_MODEL_DOMAIN_NOT_SUPPORTED"
    return None


class VerifiedEarlyCycleBatchStore(Protocol):
    """Storage contract shared by in-memory and restart-safe application stores."""

    def register_canonical_csv(
        self,
        payload: bytes,
        *,
        registration: CanonicalCsvBatchRegistration,
    ) -> str: ...

    def resolve_verified_early_cycle_batch(
        self,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch: ...


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

        batch_id, batch = _build_verified_batch(payload, registration=registration)
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


_CANONICAL_BATCH_ID = re.compile(r"canonical-csv-[0-9a-f]{64}\Z")


def _build_verified_batch(
    payload: bytes,
    *,
    registration: CanonicalCsvBatchRegistration,
) -> tuple[str, VerifiedEarlyCycleBatch]:
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
    return batch_id, batch


class FileSystemVerifiedEarlyCycleBatchStore:
    """Restart-safe canonical CSV store that revalidates source bytes on every read."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()
        if self._root.exists() and self._root.is_symlink():
            raise ValueError("verified batch store root must not be a symbolic link")
        if self._root.exists() and not self._root.is_dir():
            raise ValueError("verified batch store root must be a directory")
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def register_canonical_csv(
        self,
        payload: bytes,
        *,
        registration: CanonicalCsvBatchRegistration,
    ) -> str:
        validated = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        batch_id, _ = _build_verified_batch(payload, registration=validated)
        csv_path, registration_path = self._paths(batch_id)
        registration_payload = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with self._lock:
            if csv_path.exists() != registration_path.exists():
                raise ValueError("verified batch store contains an incomplete artifact pair")
            if csv_path.exists() and registration_path.exists():
                existing = self.resolve_verified_early_cycle_batch(batch_id)
                if existing.record_batch_id != batch_id:
                    raise ValueError("canonical CSV batch identifier collision")
                return batch_id
            self._atomic_write(csv_path, payload)
            self._atomic_write(registration_path, registration_payload)
        return batch_id

    def resolve_verified_early_cycle_batch(
        self,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        csv_path, registration_path = self._paths(record_batch_id)
        with self._lock:
            if not csv_path.is_file() or not registration_path.is_file():
                raise KeyError(f"unknown verified record batch: {record_batch_id}")
            if csv_path.is_symlink() or registration_path.is_symlink():
                raise ValueError("verified batch artifacts must not be symbolic links")
            payload = csv_path.read_bytes()
            try:
                registration = CanonicalCsvBatchRegistration.model_validate_json(
                    registration_path.read_text(encoding="utf-8")
                )
            except (UnicodeError, ValueError) as exc:
                raise ValueError("verified batch registration is invalid") from exc
            rebuilt_id, batch = _build_verified_batch(payload, registration=registration)
            if rebuilt_id != record_batch_id:
                raise ValueError("verified batch content does not match record_batch_id")
            return VerifiedEarlyCycleBatch.model_validate(batch.model_dump(mode="json"))

    def _paths(self, record_batch_id: str) -> tuple[Path, Path]:
        if not isinstance(record_batch_id, str) or not _CANONICAL_BATCH_ID.fullmatch(
            record_batch_id
        ):
            raise ValueError("record_batch_id is not a canonical CSV batch identifier")
        return (
            self._root / f"{record_batch_id}.csv",
            self._root / f"{record_batch_id}.registration.json",
        )

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=self._root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
