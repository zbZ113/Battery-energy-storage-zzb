import hashlib
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quanxin_life.core import CellMetadata, canonical_json_bytes
from quanxin_life.data.schemas import CycleRecord, DataQualitySeverity
from quanxin_life.data.validation import validate_cycle_records

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_STORAGE_VERSION = "parquet-json-v1.0.0"

CYCLE_RECORD_ARROW_SCHEMA = pa.schema(
    [
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("cycle_index", pa.int32(), nullable=False),
        pa.field("sample_index", pa.int32(), nullable=False),
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("voltage_v", pa.float64(), nullable=False),
        pa.field("current_a", pa.float64(), nullable=False),
        pa.field("temperature_c", pa.float64(), nullable=True),
        pa.field("charge_capacity_ah", pa.float64(), nullable=True),
        pa.field("discharge_capacity_ah", pa.float64(), nullable=True),
        pa.field("internal_resistance_ohm", pa.float64(), nullable=True),
        pa.field("diagnostic", pa.bool_(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
    ]
)


class ProcessedCellManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    row_count: int = Field(gt=0)
    storage_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    parquet_relative_path: str = Field(min_length=1)
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_relative_path: str = Field(min_length=1)
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_relative_path: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


@dataclass(frozen=True)
class VerifiedCellArtifacts:
    metadata: CellMetadata
    row_count: int
    parquet_path: Path
    metadata_path: Path
    manifest_path: Path


def _validate_safe_identifier(value: str) -> None:
    stem = value.split(".", maxsplit=1)[0].upper()
    if (
        not _SAFE_IDENTIFIER.fullmatch(value)
        or value.endswith(".")
        or stem in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("cell_id must be a safe file identifier")


def _safe_artifact_path(root: Path, relative_path: Path) -> Path:
    if (
        relative_path.is_absolute()
        or "\\" in relative_path.as_posix()
        or any(part in {".", ".."} for part in relative_path.parts)
    ):
        raise ValueError("artifact path must be a safe relative path")
    resolved_root = root.resolve(strict=False)
    resolved_path = (resolved_root / relative_path).resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("artifact path escapes output root")
    return resolved_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_parquet(path: Path, records: Sequence[CycleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="python") for record in records]
    table = pa.Table.from_pylist(payload, schema=CYCLE_RECORD_ARROW_SCHEMA)
    pq.write_table(table, path, compression="zstd", version="2.6")


def write_cell_artifacts(
    output_root: Path,
    metadata: CellMetadata,
    records: Sequence[CycleRecord],
) -> ProcessedCellManifest:
    """Publish one validated cell; the manifest is the final commit marker."""
    _validate_safe_identifier(metadata.cell_id)
    if not records:
        raise ValueError("at least one cycle record is required")
    if any(
        record.cell_id != metadata.cell_id or record.dataset_id != metadata.dataset_id
        for record in records
    ):
        raise ValueError("cycle records do not match metadata cell and dataset")

    quality = validate_cycle_records(records)
    if quality.blocked:
        codes = ", ".join(
            issue.code
            for issue in quality.issues
            if issue.severity == DataQualitySeverity.BLOCKING
        )
        raise ValueError(f"blocked cycle records cannot be persisted: {codes}")

    root = Path(output_root)
    manifest_relative = Path("manifests") / f"{metadata.cell_id}.json"
    manifest_path = _safe_artifact_path(root, manifest_relative)

    staging_root = _safe_artifact_path(root, Path(".staging") / uuid4().hex)
    staged_parquet = staging_root / "cell.parquet"
    staged_metadata = staging_root / "metadata.json"
    staged_manifest = staging_root / "manifest.json"
    try:
        _write_parquet(staged_parquet, records)
        _write_json(staged_metadata, metadata.model_dump(mode="json"))
        parquet_sha256 = _sha256_file(staged_parquet)
        metadata_sha256 = _sha256_file(staged_metadata)
        parquet_relative = (
            Path("cells") / metadata.cell_id / f"{parquet_sha256}.parquet"
        )
        metadata_relative = (
            Path("metadata") / metadata.cell_id / f"{metadata_sha256}.json"
        )
        parquet_path = _safe_artifact_path(root, parquet_relative)
        metadata_path = _safe_artifact_path(root, metadata_relative)
        manifest = ProcessedCellManifest(
            dataset_id=metadata.dataset_id,
            cell_id=metadata.cell_id,
            row_count=len(records),
            storage_version=_STORAGE_VERSION,
            schema_version=metadata.schema_version,
            parquet_relative_path=parquet_relative.as_posix(),
            parquet_sha256=parquet_sha256,
            metadata_relative_path=metadata_relative.as_posix(),
            metadata_sha256=metadata_sha256,
            manifest_relative_path=manifest_relative.as_posix(),
        )
        _write_json(staged_manifest, manifest.model_dump(mode="json"))

        # Validate all publication destinations before exposing any artifact.
        for parent in (parquet_path.parent, metadata_path.parent, manifest_path.parent):
            parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_parquet, parquet_path)
        os.replace(staged_metadata, metadata_path)
        os.replace(staged_manifest, manifest_path)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _verified_relative_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.as_posix() != relative:
        raise ValueError("manifest contains a non-canonical artifact path")
    return _safe_artifact_path(root, candidate)


def _validate_manifest_path_contract(manifest: ProcessedCellManifest) -> None:
    expected_parquet = (
        Path("cells") / manifest.cell_id / f"{manifest.parquet_sha256}.parquet"
    ).as_posix()
    expected_metadata = (
        Path("metadata") / manifest.cell_id / f"{manifest.metadata_sha256}.json"
    ).as_posix()
    expected_manifest = (Path("manifests") / f"{manifest.cell_id}.json").as_posix()
    if (
        manifest.parquet_relative_path != expected_parquet
        or manifest.metadata_relative_path != expected_metadata
        or manifest.manifest_relative_path != expected_manifest
    ):
        raise ValueError("manifest artifact path contract mismatch")


def verify_cell_artifacts(
    output_root: Path, manifest: ProcessedCellManifest
) -> VerifiedCellArtifacts:
    """Verify a published cell's commit marker, hashes, schemas and identities."""
    _validate_safe_identifier(manifest.cell_id)
    _validate_manifest_path_contract(manifest)
    root = Path(output_root)
    parquet_path = _verified_relative_path(root, manifest.parquet_relative_path)
    metadata_path = _verified_relative_path(root, manifest.metadata_relative_path)
    manifest_path = _verified_relative_path(root, manifest.manifest_relative_path)

    if not manifest_path.is_file():
        raise ValueError("manifest commit marker is missing")
    persisted_manifest = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())
    if persisted_manifest != manifest:
        raise ValueError("manifest commit marker does not match requested manifest")
    if not parquet_path.is_file() or _sha256_file(parquet_path) != manifest.parquet_sha256:
        raise ValueError("parquet SHA-256 mismatch")
    if not metadata_path.is_file() or _sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError("metadata SHA-256 mismatch")

    metadata = CellMetadata.model_validate_json(metadata_path.read_bytes())
    if (
        metadata.dataset_id != manifest.dataset_id
        or metadata.cell_id != manifest.cell_id
        or metadata.schema_version != manifest.schema_version
        or manifest.storage_version != _STORAGE_VERSION
    ):
        raise ValueError("metadata and manifest contracts do not match")

    table = pq.read_table(parquet_path)
    if table.schema != CYCLE_RECORD_ARROW_SCHEMA:
        raise ValueError("parquet schema mismatch")
    if table.num_rows != manifest.row_count:
        raise ValueError("parquet row count mismatch")
    if set(table.column("dataset_id").to_pylist()) != {manifest.dataset_id}:
        raise ValueError("parquet dataset_id does not match manifest")
    if set(table.column("cell_id").to_pylist()) != {manifest.cell_id}:
        raise ValueError("parquet cell_id does not match manifest")

    return VerifiedCellArtifacts(
        metadata=metadata,
        row_count=table.num_rows,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )
