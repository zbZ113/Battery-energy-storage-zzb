import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from quanxin_life.core import CellMetadata
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.validation import validate_cycle_records

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_STORAGE_VERSION = "parquet-json-v1.0.0"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(encoded)


def _write_parquet(path: Path, records: Sequence[CycleRecord]) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - only without the data extra
        raise RuntimeError("Parquet storage requires the 'data' optional dependencies") from exc

    schema = pa.schema(
        [
            ("dataset_id", pa.string()),
            ("cell_id", pa.string()),
            ("cycle_index", pa.int32()),
            ("sample_index", pa.int32()),
            ("time_s", pa.float64()),
            ("voltage_v", pa.float64()),
            ("current_a", pa.float64()),
            ("temperature_c", pa.float64()),
            ("charge_capacity_ah", pa.float64()),
            ("discharge_capacity_ah", pa.float64()),
            ("internal_resistance_ohm", pa.float64()),
            ("diagnostic", pa.bool_()),
            ("valid", pa.bool_()),
        ]
    )
    payload = [record.model_dump(mode="python") for record in records]
    table = pa.Table.from_pylist(payload, schema=schema)
    pq.write_table(table, path, compression="zstd", version="2.6")


def write_cell_artifacts(
    output_root: Path,
    metadata: CellMetadata,
    records: Sequence[CycleRecord],
) -> ProcessedCellManifest:
    """Persist one validated cell without executable serialization formats."""
    if not _SAFE_IDENTIFIER.fullmatch(metadata.cell_id):
        raise ValueError("cell_id must be a safe file identifier")
    if not records:
        raise ValueError("at least one cycle record is required")
    if any(
        record.cell_id != metadata.cell_id or record.dataset_id != metadata.dataset_id
        for record in records
    ):
        raise ValueError("cycle records do not match metadata cell and dataset")

    quality = validate_cycle_records(records)
    if quality.blocked:
        codes = ", ".join(issue.code for issue in quality.issues if issue.severity == "blocking")
        raise ValueError(f"blocked cycle records cannot be persisted: {codes}")

    root = Path(output_root)
    parquet_relative = Path("cells") / f"{metadata.cell_id}.parquet"
    metadata_relative = Path("metadata") / f"{metadata.cell_id}.json"
    manifest_relative = Path("manifests") / f"{metadata.cell_id}.json"
    parquet_path = root / parquet_relative
    metadata_path = root / metadata_relative
    manifest_path = root / manifest_relative

    _atomic_write(parquet_path, lambda temporary: _write_parquet(temporary, records))
    _atomic_write(
        metadata_path,
        lambda temporary: _write_json(temporary, metadata.model_dump(mode="json")),
    )
    manifest = ProcessedCellManifest(
        dataset_id=metadata.dataset_id,
        cell_id=metadata.cell_id,
        row_count=len(records),
        storage_version=_STORAGE_VERSION,
        schema_version=metadata.schema_version,
        parquet_relative_path=parquet_relative.as_posix(),
        parquet_sha256=_sha256_file(parquet_path),
        metadata_relative_path=metadata_relative.as_posix(),
        metadata_sha256=_sha256_file(metadata_path),
        manifest_relative_path=manifest_relative.as_posix(),
    )
    _atomic_write(
        manifest_path,
        lambda temporary: _write_json(temporary, manifest.model_dump(mode="json")),
    )
    return manifest
