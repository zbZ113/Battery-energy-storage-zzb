"""Reviewed MATR batch conversion into verified per-cell Parquet artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.matr import iter_matr_batch
from quanxin_life.data.labels import DiagnosticPoint, derive_eol80
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.data.split import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    build_stratified_cell_split,
)
from quanxin_life.data.storage import verify_cell_artifacts, write_cell_artifacts
from quanxin_life.data.validation import validate_cycle_records


class MatrCellConversionEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    raw_cell_id: str = Field(min_length=1)
    official_life_label: int | None = Field(default=None, ge=0)
    official_life_right_censored: bool
    protocol_id: str = Field(min_length=1)
    reference_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    row_count: int = Field(gt=0)
    cycle_count: int = Field(gt=0)
    quality_issue_counts: dict[str, int]
    manifest_relative_path: str = Field(min_length=1)
    parquet_sha256: Sha256
    metadata_sha256: Sha256


class MatrBatchConversionReport(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-batch-conversion-v1"] = "matr-batch-conversion-v1"
    dataset_id: Literal["MATR"] = "MATR"
    batch_index: int = Field(ge=1)
    batch_date: date
    raw_relative_path: str = Field(min_length=1)
    raw_size_bytes: int = Field(gt=0)
    raw_sha256: Sha256
    source_uri: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_uri: str | None = None
    paper_doi: str | None = None
    adapter_version: str = Field(min_length=1)
    time_unit: Literal["seconds", "minutes"]
    max_cycle_index: int = Field(ge=5)
    cell_count: int = Field(gt=0)
    total_row_count: int = Field(gt=0)
    quality_issue_counts: dict[str, int]
    cells: tuple[MatrCellConversionEvidence, ...] = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


class MatrSplitEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-split-evidence-v1"] = "matr-split-evidence-v1"
    split_version: str = Field(min_length=1)
    source_report_sha256: Sha256
    seed: int
    ratios: tuple[float, float, float, float]
    quantile_count: int = Field(ge=2)
    cell_strata: dict[str, str]
    split_manifest: SplitManifest
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def split_created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


class MatrEOL80CellEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    raw_cell_id: str = Field(min_length=1)
    reference_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    observed_cycle_count: int = Field(gt=0)
    minimum_observed_soh: float = Field(ge=0, allow_inf_nan=False)
    official_life_label: int | None = Field(default=None, ge=0)
    unified_eol80_cycle: int | None = Field(default=None, ge=0)
    unified_confirmation_cycles: tuple[int, ...] = ()
    unified_right_censored: bool


class MatrEOL80LabelAudit(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-eol80-label-audit-v1"] = "matr-eol80-label-audit-v1"
    dataset_id: Literal["MATR"] = "MATR"
    source_report_sha256: Sha256
    raw_sha256: Sha256
    threshold: float = Field(default=0.8, gt=0, lt=1)
    confirmations: int = Field(default=3, ge=1)
    cell_count: int = Field(gt=0)
    unified_event_count: int = Field(ge=0)
    unified_right_censored_count: int = Field(ge=0)
    cells: tuple[MatrEOL80CellEvidence, ...] = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def label_created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def label_counts_match_cells(self) -> MatrEOL80LabelAudit:
        if self.cell_count != len(self.cells):
            raise ValueError("cell_count must match MATR EOL80 cell evidence")
        if self.unified_event_count + self.unified_right_censored_count != self.cell_count:
            raise ValueError("MATR EOL80 event and censoring counts must cover all cells")
        return self


def build_matr_life_strata(
    cells: Sequence[MatrCellConversionEvidence],
    *,
    quantile_count: int = 4,
) -> dict[str, str]:
    """Combine protocol and event-only life ranks; censoring remains separate."""

    if quantile_count < 2:
        raise ValueError("quantile_count must be at least two")
    if not cells:
        raise ValueError("at least one MATR cell is required for stratification")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("MATR stratification cell identifiers must be unique")

    observed = sorted(
        (cell for cell in cells if not cell.official_life_right_censored),
        key=lambda cell: (
            cell.official_life_label if cell.official_life_label is not None else -1,
            cell.cell_id,
        ),
    )
    if any(cell.official_life_label is None for cell in observed):
        raise ValueError("observed MATR life strata require an official life label")
    strata: dict[str, str] = {}
    for rank, cell in enumerate(observed):
        life_quantile = min((rank * quantile_count) // len(observed) + 1, quantile_count)
        strata[cell.cell_id] = f"{cell.protocol_id}|life-q{life_quantile}"
    for cell in cells:
        if cell.official_life_right_censored:
            if cell.official_life_label is not None:
                raise ValueError("right-censored MATR cells cannot contain an event label")
            strata[cell.cell_id] = f"{cell.protocol_id}|right-censored"
    return dict(sorted(strata.items()))


def build_matr_split_evidence(
    report: MatrBatchConversionReport,
    *,
    split_version: str,
    created_at: datetime,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float, float] = DEFAULT_RATIOS,
    quantile_count: int = 4,
) -> MatrSplitEvidence:
    """Bind a deterministic cell-disjoint split to one conversion report."""

    if report.cell_count != len(report.cells):
        raise ValueError("MATR report cell_count does not match its cell evidence")
    strata = build_matr_life_strata(report.cells, quantile_count=quantile_count)
    manifest = build_stratified_cell_split(
        report.dataset_id,
        strata,
        seed=seed,
        ratios=ratios,
    )
    return MatrSplitEvidence(
        split_version=split_version,
        source_report_sha256=sha256_canonical(report.model_dump(mode="json")),
        seed=seed,
        ratios=ratios,
        quantile_count=quantile_count,
        cell_strata=strata,
        split_manifest=manifest,
        created_at=created_at,
    )


def audit_matr_eol80_labels(
    *,
    raw_path: Path,
    raw_manifest: RawFileManifest,
    conversion_report: MatrBatchConversionReport,
    created_at: datetime,
) -> MatrEOL80LabelAudit:
    """Derive the project EOL80 definition from full summary capacity trajectories."""

    if raw_manifest.sha256 != conversion_report.raw_sha256:
        raise ValueError("MATR label audit manifest differs from the conversion report")
    verify_raw_file(raw_path, raw_manifest)
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MATR label audit requires the 'data' optional dependencies") from exc

    cell_evidence: list[MatrEOL80CellEvidence] = []
    with h5py.File(raw_path, "r") as handle:
        if "batch" not in handle or "summary" not in handle["batch"]:
            raise ValueError("MATR label audit requires the batch summary references")
        batch = handle["batch"]
        for cell in conversion_report.cells:
            match = re.fullmatch(
                rf"b{conversion_report.batch_index}c(?P<index>\d+)",
                cell.raw_cell_id,
            )
            if match is None:
                raise ValueError("MATR label audit found an invalid raw_cell_id")
            cell_index = int(match.group("index"))
            if cell_index >= batch["summary"].shape[0]:
                raise ValueError("MATR label audit raw_cell_id exceeds the batch size")
            summary = handle[batch["summary"][cell_index, 0]]
            if "QDischarge" not in summary:
                raise ValueError("MATR label audit requires summary QDischarge")
            capacities = _resolve_hdf5_numeric(handle, summary["QDischarge"])
            if len(capacities) < 6 or any(
                not math.isfinite(value) or value <= 0 for value in capacities
            ):
                raise ValueError("MATR label audit found invalid summary capacities")
            soh = [value / cell.reference_capacity_ah for value in capacities]
            label = derive_eol80(
                (
                    DiagnosticPoint(cycle_index=index, soh=value)
                    for index, value in enumerate(soh)
                    if index > 0
                ),
                threshold=0.8,
                confirmations=3,
            )
            cell_evidence.append(
                MatrEOL80CellEvidence(
                    cell_id=cell.cell_id,
                    raw_cell_id=cell.raw_cell_id,
                    reference_capacity_ah=cell.reference_capacity_ah,
                    observed_cycle_count=len(capacities),
                    minimum_observed_soh=min(soh[1:]),
                    official_life_label=cell.official_life_label,
                    unified_eol80_cycle=label.eol_cycle,
                    unified_confirmation_cycles=label.confirmation_cycles,
                    unified_right_censored=label.right_censored,
                )
            )

    if len(cell_evidence) != conversion_report.cell_count:
        raise ValueError("MATR label audit did not cover every converted cell")
    event_count = sum(not cell.unified_right_censored for cell in cell_evidence)
    return MatrEOL80LabelAudit(
        source_report_sha256=sha256_canonical(conversion_report.model_dump(mode="json")),
        raw_sha256=raw_manifest.sha256,
        cell_count=len(cell_evidence),
        unified_event_count=event_count,
        unified_right_censored_count=len(cell_evidence) - event_count,
        cells=tuple(cell_evidence),
        created_at=created_at,
    )


def _resolve_hdf5_numeric(handle: Any, dataset: Any) -> list[float]:
    values = dataset[()]
    if values.dtype.kind != "O":
        return [float(item) for item in values.reshape(-1)]
    resolved: list[float] = []
    for reference in values.reshape(-1):
        referenced = handle[reference][()]
        resolved.extend(float(item) for item in referenced.reshape(-1))
    return resolved


def convert_matr_batch(
    *,
    raw_path: Path,
    raw_manifest: RawFileManifest,
    output_root: Path,
    batch_index: int,
    batch_date: date,
    time_unit: Literal["seconds", "minutes"],
    max_cycle_index: int,
    created_at: datetime | None = None,
) -> MatrBatchConversionReport:
    """Convert a reviewed batch once, publishing and re-verifying each cell."""

    if max_cycle_index < 5:
        raise ValueError("MATR conversion requires cycles 1 through 5 for reference capacity")
    raw_path = Path(raw_path)
    output_root = Path(output_root)
    cell_evidence: list[MatrCellConversionEvidence] = []
    total_quality_counts: dict[str, int] = {}
    adapter_versions: set[str] = set()

    for metadata, records in iter_matr_batch(
        raw_path,
        raw_manifest,
        batch_index=batch_index,
        time_unit=time_unit,
        max_cycle_index=max_cycle_index,
        expected_batch_date=batch_date,
    ):
        if (
            metadata.raw_cell_id is None
            or metadata.protocol_id is None
            or metadata.reference_capacity_ah is None
            or metadata.adapter_version is None
        ):
            raise ValueError("MATR conversion produced incomplete governed cell metadata")
        official_life_right_censored = metadata.ingestion_parameters.get(
            "official_life_right_censored"
        )
        if not isinstance(official_life_right_censored, bool):
            raise ValueError("MATR metadata is missing the right-censoring decision")
        if official_life_right_censored == (metadata.official_life_label is not None):
            raise ValueError("MATR official life label conflicts with right-censoring metadata")

        quality = validate_cycle_records(records)
        cell_quality_counts: dict[str, int] = {}
        for issue in quality.issues:
            cell_quality_counts[issue.code] = cell_quality_counts.get(issue.code, 0) + 1
            total_quality_counts[issue.code] = total_quality_counts.get(issue.code, 0) + 1

        processed = write_cell_artifacts(output_root, metadata, records)
        verified = verify_cell_artifacts(output_root, processed)
        if verified.row_count != len(records):
            raise ValueError("verified MATR row count differs from the converted records")
        adapter_versions.add(metadata.adapter_version)
        cell_evidence.append(
            MatrCellConversionEvidence(
                cell_id=metadata.cell_id,
                raw_cell_id=metadata.raw_cell_id,
                official_life_label=metadata.official_life_label,
                official_life_right_censored=official_life_right_censored,
                protocol_id=metadata.protocol_id,
                reference_capacity_ah=metadata.reference_capacity_ah,
                row_count=processed.row_count,
                cycle_count=len({record.cycle_index for record in records}),
                quality_issue_counts=dict(sorted(cell_quality_counts.items())),
                manifest_relative_path=processed.manifest_relative_path,
                parquet_sha256=processed.parquet_sha256,
                metadata_sha256=processed.metadata_sha256,
            )
        )

    if not cell_evidence:
        raise ValueError("MATR batch conversion produced no cells")
    if len(adapter_versions) != 1:
        raise ValueError("MATR batch conversion mixed adapter versions")

    return MatrBatchConversionReport(
        batch_index=batch_index,
        batch_date=batch_date,
        raw_relative_path=raw_manifest.relative_path,
        raw_size_bytes=raw_path.stat().st_size,
        raw_sha256=raw_manifest.sha256,
        source_uri=raw_manifest.source_uri,
        license_name=raw_manifest.license_name,
        license_uri=raw_manifest.license_uri,
        paper_doi=raw_manifest.paper_doi,
        adapter_version=next(iter(adapter_versions)),
        time_unit=time_unit,
        max_cycle_index=max_cycle_index,
        cell_count=len(cell_evidence),
        total_row_count=sum(cell.row_count for cell in cell_evidence),
        quality_issue_counts=dict(sorted(total_quality_counts.items())),
        cells=tuple(cell_evidence),
        created_at=created_at or datetime.now(UTC),
    )
