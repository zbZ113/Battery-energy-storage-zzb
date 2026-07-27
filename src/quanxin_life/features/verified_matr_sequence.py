"""Verified MATR processed-cell loading shared by training and workers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quanxin_life.core import CellMetadata
from quanxin_life.data.matr_pipeline import MatrCellConversionEvidence
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.storage import (
    CYCLE_RECORD_ARROW_SCHEMA,
    ProcessedCellManifest,
    verify_cell_artifacts,
)
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.features.multichannel_cycle import (
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)


class EarlyCycleSequenceCachePort(Protocol):
    def get_or_build(
        self,
        context: Any,
        builder: Callable[[], EarlyCycleSequence],
    ) -> EarlyCycleSequence: ...


@dataclass(frozen=True, slots=True)
class VerifiedMatrEarlySequenceLoad:
    sequence: EarlyCycleSequence
    masked_cycle_indices: tuple[int, ...]


def load_verified_matr_early_sequence(
    *,
    processed_root: Path,
    evidence: MatrCellConversionEvidence,
    reference_capacity_ah: float,
    raw_sha256: str,
    config: MultichannelCycleConfig,
    data_version: str,
    cache: EarlyCycleSequenceCachePort | None = None,
) -> VerifiedMatrEarlySequenceLoad:
    """Load cutoff-bounded rows after verifying all processed-cell evidence."""

    root = processed_root.resolve(strict=True)
    relative = _safe_relative(
        evidence.manifest_relative_path,
        "processed manifest",
    )
    manifest_path = (root / Path(*relative.parts)).resolve(strict=True)
    if not manifest_path.is_relative_to(root) or manifest_path.is_symlink():
        raise ValueError("processed manifest must remain inside processed_root")
    manifest = ProcessedCellManifest.model_validate_json(
        manifest_path.read_bytes()
    )
    if (
        manifest.cell_id != evidence.cell_id
        or manifest.parquet_sha256 != evidence.parquet_sha256
        or manifest.metadata_sha256 != evidence.metadata_sha256
        or manifest.row_count != evidence.row_count
    ):
        raise ValueError(
            "processed cell manifest differs from conversion evidence"
        )
    verified = verify_cell_artifacts(root, manifest)
    metadata_bytes = verified.metadata_path.read_bytes()
    if (
        hashlib.sha256(metadata_bytes).hexdigest()
        != manifest.metadata_sha256
    ):
        raise ValueError("consumed metadata SHA-256 mismatch")
    metadata = CellMetadata.model_validate_json(metadata_bytes)
    parquet_bytes = verified.parquet_path.read_bytes()
    if (
        hashlib.sha256(parquet_bytes).hexdigest()
        != manifest.parquet_sha256
    ):
        raise ValueError("consumed Parquet SHA-256 mismatch")
    table = pq.read_table(pa.BufferReader(parquet_bytes))
    if (
        table.schema != CYCLE_RECORD_ARROW_SCHEMA
        or table.num_rows != manifest.row_count
        or set(table.column("dataset_id").to_pylist())
        != {manifest.dataset_id}
        or set(table.column("cell_id").to_pylist()) != {manifest.cell_id}
    ):
        raise ValueError(
            "consumed Parquet schema or identity is invalid"
        )
    if (
        metadata.source_sha256 != raw_sha256
        or metadata.official_life_label
        != evidence.official_life_label
        or metadata.reference_capacity_ah
        != evidence.reference_capacity_ah
        or reference_capacity_ah != evidence.reference_capacity_ah
    ):
        raise ValueError(
            "processed cell metadata differs from registered MATR evidence"
        )
    if not math.isfinite(reference_capacity_ah) or reference_capacity_ah <= 0:
        raise ValueError(
            "reference_capacity_ah must be finite and positive"
        )
    filtered_table = table.filter(
        pc.less_equal(
            table["cycle_index"],
            pa.scalar(config.cutoff_cycle),
        )
    )
    capacity_table = filtered_table.select(
        [
            "cycle_index",
            "charge_capacity_ah",
            "discharge_capacity_ah",
        ]
    )
    masked_cycles = _capacity_outlier_cycles_from_rows(
        capacity_table.to_pylist(),
        reference_capacity_ah,
    )

    def build() -> EarlyCycleSequence:
        records = tuple(
            CycleRecord.model_validate(row)
            for row in filtered_table.to_pylist()
        )
        if not records or any(
            record.cycle_index > config.cutoff_cycle
            for record in records
        ):
            raise ValueError(
                "early MATR input must contain only cutoff-bounded rows"
            )
        filtered_records = tuple(
            record
            for record in records
            if record.cycle_index not in masked_cycles
        )
        return build_early_cycle_sequence(
            filtered_records,
            config=config,
            data_version=data_version,
        )

    if cache is None:
        sequence = build()
    else:
        from quanxin_life.training.advanced_cache import (
            EarlySequenceCacheContext,
        )

        context = EarlySequenceCacheContext.from_multichannel_config(
            source_parquet_sha256=manifest.parquet_sha256,
            dataset_id=manifest.dataset_id,
            cell_id=manifest.cell_id,
            data_version=data_version,
            config=config,
        )
        sequence = cache.get_or_build(context, build)
    return VerifiedMatrEarlySequenceLoad(
        sequence=sequence,
        masked_cycle_indices=tuple(sorted(masked_cycles)),
    )


def _capacity_outlier_cycles_from_rows(
    rows: Iterable[dict[str, object]],
    reference_capacity_ah: float,
) -> frozenset[int]:
    masked: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("capacity audit rows must be mappings")
        capacities = (
            row.get("charge_capacity_ah"),
            row.get("discharge_capacity_ah"),
        )
        if any(
            isinstance(capacity, int | float)
            and not isinstance(capacity, bool)
            and math.isfinite(capacity)
            and capacity / reference_capacity_ah > 1.5
            for capacity in capacities
        ):
            cycle = row.get("cycle_index")
            if (
                isinstance(cycle, bool)
                or not isinstance(cycle, int)
                or cycle < 0
            ):
                raise ValueError(
                    "capacity audit cycle_index must be non-negative"
                )
            masked.add(cycle)
    return frozenset(masked)


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"{label} must be a safe relative POSIX path"
        )
    return path


__all__ = [
    "EarlyCycleSequenceCachePort",
    "VerifiedMatrEarlySequenceLoad",
    "load_verified_matr_early_sequence",
]
