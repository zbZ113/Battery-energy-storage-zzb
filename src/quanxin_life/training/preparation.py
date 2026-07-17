"""Idempotent preparation and verification of the approved MATR training inputs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from quanxin_life.core import sha256_canonical
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrSupervisionArtifact,
    build_matr_split_evidence,
    build_matr_supervision_artifact,
    convert_matr_batch,
)
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.data.storage import ProcessedCellManifest, verify_cell_artifacts
from quanxin_life.training.suite import MatrRunConfig

_APPROVED_BATCH_INDEX = 3
_APPROVED_BATCH_DATE = date(2018, 4, 12)
_APPROVED_TIME_UNIT: Literal["minutes"] = "minutes"
_SUPERVISION_HORIZON = 500


def prepare_matr_training_inputs(
    *,
    project_root: Path,
    config: MatrRunConfig,
) -> dict[str, Any]:
    """Create missing governed inputs and verify every cached byte before reuse."""

    root = project_root.resolve(strict=True)
    paths = config.paths
    raw_path = _inside(root, paths.raw_mat)
    raw_manifest = RawFileManifest.model_validate_json(
        _inside(root, paths.raw_manifest).read_bytes()
    )
    verify_raw_file(raw_path, raw_manifest)

    processed_root = _inside(root, paths.processed_root, must_exist=False)
    conversion_path = _inside(root, paths.conversion_report, must_exist=False)
    conversion_created = not conversion_path.exists()
    if conversion_created:
        conversion = convert_matr_batch(
            raw_path=raw_path,
            raw_manifest=raw_manifest,
            output_root=processed_root,
            batch_index=_APPROVED_BATCH_INDEX,
            batch_date=_APPROVED_BATCH_DATE,
            time_unit=_APPROVED_TIME_UNIT,
            max_cycle_index=max(config.suite.cutoffs),
        )
        _write_json_atomic(conversion_path, conversion.model_dump(mode="json"))
    else:
        conversion = MatrBatchConversionReport.model_validate_json(
            conversion_path.read_bytes()
        )
    _verify_conversion(
        conversion,
        raw_manifest=raw_manifest,
        processed_root=processed_root,
        required_cutoff=max(config.suite.cutoffs),
    )

    split_path = _inside(root, paths.split_manifest, must_exist=False)
    split_created = not split_path.exists()
    if split_created:
        split_evidence = build_matr_split_evidence(
            conversion,
            split_version=config.suite.split_version,
            created_at=datetime.now(UTC),
        )
        split = split_evidence.split_manifest
        _write_json_atomic(split_path, split.model_dump(mode="json"))
    else:
        split = SplitManifest.model_validate_json(split_path.read_bytes())
    _verify_split(split, conversion)

    supervision_root = _inside(root, paths.supervision_root, must_exist=False)
    supervision_path = _inside(root, paths.supervision_report, must_exist=False)
    supervision_created = not supervision_path.exists()
    if supervision_created:
        supervision = build_matr_supervision_artifact(
            raw_path=raw_path,
            raw_manifest=raw_manifest,
            conversion_report=conversion,
            output_root=supervision_root,
            horizon_cycle=_SUPERVISION_HORIZON,
            created_at=datetime.now(UTC),
        )
        _write_json_atomic(supervision_path, supervision.model_dump(mode="json"))
    else:
        supervision = MatrSupervisionArtifact.model_validate_json(
            supervision_path.read_bytes()
        )
    _verify_supervision(
        supervision,
        conversion=conversion,
        raw_manifest=raw_manifest,
        supervision_root=supervision_root,
    )
    return {
        "status": "MATR_TRAINING_INPUTS_READY",
        "raw_sha256": raw_manifest.sha256,
        "cell_count": conversion.cell_count,
        "max_feature_cycle": conversion.max_cycle_index,
        "supervision_horizon_cycle": supervision.horizon_cycle,
        "conversion_created": conversion_created,
        "split_created": split_created,
        "supervision_created": supervision_created,
    }


def _verify_conversion(
    report: MatrBatchConversionReport,
    *,
    raw_manifest: RawFileManifest,
    processed_root: Path,
    required_cutoff: int,
) -> None:
    if (
        report.raw_sha256 != raw_manifest.sha256
        or report.raw_relative_path != raw_manifest.relative_path
        or report.batch_index != _APPROVED_BATCH_INDEX
        or report.batch_date != _APPROVED_BATCH_DATE
        or report.time_unit != _APPROVED_TIME_UNIT
        or report.max_cycle_index < required_cutoff
    ):
        raise ValueError("MATR conversion cache does not match the approved raw context")
    if report.cell_count != len(report.cells):
        raise ValueError("MATR conversion cache cell count is inconsistent")
    for evidence in report.cells:
        manifest_path = _inside(
            processed_root,
            evidence.manifest_relative_path,
        )
        manifest = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())
        verified = verify_cell_artifacts(processed_root, manifest)
        if (
            manifest.cell_id != evidence.cell_id
            or manifest.parquet_sha256 != evidence.parquet_sha256
            or manifest.metadata_sha256 != evidence.metadata_sha256
            or verified.row_count != evidence.row_count
        ):
            raise ValueError("MATR conversion evidence does not match cached cell artifacts")


def _verify_split(split: SplitManifest, conversion: MatrBatchConversionReport) -> None:
    partitions = (split.train, split.validation, split.calibration, split.test)
    flattened = [cell_id for partition in partitions for cell_id in partition]
    if len(flattened) != len(set(flattened)):
        raise ValueError("MATR split assigns one cell to multiple partitions")
    expected = {cell.cell_id for cell in conversion.cells}
    if split.dataset_id != "MATR" or set(flattened) != expected:
        raise ValueError("MATR split does not cover the approved conversion cells exactly")


def _verify_supervision(
    artifact: MatrSupervisionArtifact,
    *,
    conversion: MatrBatchConversionReport,
    raw_manifest: RawFileManifest,
    supervision_root: Path,
) -> None:
    if (
        artifact.raw_sha256 != raw_manifest.sha256
        or artifact.source_report_sha256
        != sha256_canonical(conversion.model_dump(mode="json"))
        or artifact.horizon_cycle != _SUPERVISION_HORIZON
        or {cell.cell_id for cell in artifact.cells}
        != {cell.cell_id for cell in conversion.cells}
    ):
        raise ValueError("MATR supervision cache does not match the approved context")
    parquet_path = _inside(supervision_root, artifact.parquet_relative_path)
    if _sha256_file(parquet_path) != artifact.parquet_sha256:
        raise ValueError("MATR supervision Parquet SHA-256 mismatch")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MATR preparation requires the data dependencies") from exc
    metadata = pq.read_metadata(parquet_path)
    required_columns = {
        "dataset_id",
        "cell_id",
        "cycle_index",
        "discharge_capacity_ah",
        "reference_capacity_ah",
        "soh",
    }
    if metadata.num_rows != artifact.row_count or set(metadata.schema.names) != required_columns:
        raise ValueError("MATR supervision Parquet schema or row count mismatch")


def _inside(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = (root / Path(relative)).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise ValueError("MATR preparation path escapes the project root")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
