"""Verified MATR early-input assembly kept separate from future supervision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.data.matr_pipeline import MatrSupervisionArtifact
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.data.storage import ProcessedCellManifest, verify_cell_artifacts
from quanxin_life.features.curve_tensor import (
    CurveTensorConfig,
    build_discharge_curve_tensor,
)
from quanxin_life.features.early_cycle import (
    EarlyCycleFeatureConfig,
    extract_early_cycle_features,
)
from quanxin_life.models.cpmlp import curve_tensor_to_tensors
from quanxin_life.training.tasks import (
    CycleLifeCurveBatch,
    HybridTrajectoryBatch,
)

_HYBRID_FEATURE_NAMES = (
    "capacity_first_ah",
    "capacity_last_ah",
    "capacity_delta_ah",
    "capacity_relative_change",
    "capacity_slope_ah_per_cycle",
)


@dataclass(frozen=True)
class MatrCurveCohorts:
    train: CycleLifeCurveBatch
    validation: CycleLifeCurveBatch
    calibration: CycleLifeCurveBatch
    test: CycleLifeCurveBatch


@dataclass(frozen=True)
class MatrHybridCohorts:
    train: HybridTrajectoryBatch
    validation: HybridTrajectoryBatch
    calibration: HybridTrajectoryBatch
    test: HybridTrajectoryBatch


def load_matr_cycle_life_curve_cohorts(
    *,
    processed_root: Path,
    supervision_root: Path,
    supervision: MatrSupervisionArtifact,
    split_manifest: SplitManifest,
    cutoff_cycle: int,
    voltage_min_v: float,
    voltage_max_v: float,
    voltage_grid_step_v: float,
) -> MatrCurveCohorts:
    """Build four curve cohorts without exposing the supervision file to features."""

    if split_manifest.dataset_id != "MATR" or supervision.dataset_id != "MATR":
        raise ValueError("MATR curve loading requires MATR split and supervision contracts")
    _verify_supervision_bytes(supervision_root, supervision)
    supervision_by_cell = {cell.cell_id: cell for cell in supervision.cells}
    if len(supervision_by_cell) != len(supervision.cells):
        raise ValueError("MATR supervision cell identifiers must be unique")
    split_cells = set(
        (
            *split_manifest.train,
            *split_manifest.validation,
            *split_manifest.calibration,
            *split_manifest.test,
        )
    )
    if split_cells != set(supervision_by_cell):
        raise ValueError("MATR split cells must exactly match supervision cells")

    manifest_directory = Path(processed_root) / "manifests"
    manifest_paths = tuple(sorted(manifest_directory.glob("*.json")))
    if {path.stem for path in manifest_paths} != split_cells:
        raise ValueError("processed MATR manifest inventory must exactly match split cells")
    manifests = {
        path.stem: ProcessedCellManifest.model_validate_json(path.read_bytes())
        for path in manifest_paths
    }
    feature_config = CurveTensorConfig(
        cutoff_cycle=cutoff_cycle,
        voltage_grid_step_v=voltage_grid_step_v,
        voltage_min_v=voltage_min_v,
        voltage_max_v=voltage_max_v,
    )
    prepared: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for cell_id in sorted(split_cells):
        manifest = manifests[cell_id]
        verified = verify_cell_artifacts(processed_root, manifest)
        label = supervision_by_cell[cell_id]
        if verified.metadata.source_sha256 != supervision.raw_sha256:
            raise ValueError("early input and supervision raw source hashes differ")
        if (
            verified.metadata.official_life_label != label.official_life_label
            or verified.metadata.reference_capacity_ah != label.reference_capacity_ah
        ):
            raise ValueError("early metadata and supervision label evidence differ")
        if label.official_life_right_censored:
            continue
        if label.official_life_label is None:
            raise ValueError("observed MATR official life requires a label")
        records = _read_cutoff_records(verified.parquet_path, cutoff_cycle=cutoff_cycle)
        curve = build_discharge_curve_tensor(records, config=feature_config)
        values, mask = curve_tensor_to_tensors(curve)
        prepared[cell_id] = (values, mask, float(label.official_life_label))

    return MatrCurveCohorts(
        train=_build_batch(split_manifest.train, prepared, cutoff_cycle=cutoff_cycle),
        validation=_build_batch(
            split_manifest.validation, prepared, cutoff_cycle=cutoff_cycle
        ),
        calibration=_build_batch(
            split_manifest.calibration, prepared, cutoff_cycle=cutoff_cycle
        ),
        test=_build_batch(split_manifest.test, prepared, cutoff_cycle=cutoff_cycle),
    )


def load_matr_hybrid_trajectory_cohorts(
    *,
    processed_root: Path,
    supervision_root: Path,
    supervision: MatrSupervisionArtifact,
    split_manifest: SplitManifest,
    cutoff_cycle: int,
) -> MatrHybridCohorts:
    """Join cutoff-safe early features to real SOH labels through cycle 500."""

    if split_manifest.dataset_id != "MATR" or supervision.dataset_id != "MATR":
        raise ValueError("MATR Hybrid loading requires MATR split and supervision contracts")
    supervision_path = _verify_supervision_bytes(supervision_root, supervision)
    table = pq.read_table(supervision_path)
    expected_columns = (
        "dataset_id",
        "cell_id",
        "cycle_index",
        "discharge_capacity_ah",
        "reference_capacity_ah",
        "soh",
    )
    if tuple(table.schema.names) != expected_columns or table.num_rows != supervision.row_count:
        raise ValueError("MATR supervision Parquet schema or row count is invalid")
    trajectory_rows: dict[str, dict[int, float]] = {}
    for row in table.to_pylist():
        if row["dataset_id"] != "MATR":
            raise ValueError("MATR supervision contains another dataset")
        cell_id = str(row["cell_id"])
        cycle = int(row["cycle_index"])
        soh = float(row["soh"])
        if not 1 <= cycle <= 500 or not np.isfinite(soh) or soh <= 0:
            raise ValueError("MATR supervision contains an invalid SOH trajectory row")
        cell_rows = trajectory_rows.setdefault(cell_id, {})
        if cycle in cell_rows:
            raise ValueError("MATR supervision contains a duplicate cell-cycle row")
        cell_rows[cycle] = soh

    supervision_by_cell = {cell.cell_id: cell for cell in supervision.cells}
    split_cells = set(
        (
            *split_manifest.train,
            *split_manifest.validation,
            *split_manifest.calibration,
            *split_manifest.test,
        )
    )
    if set(trajectory_rows) != split_cells or set(supervision_by_cell) != split_cells:
        raise ValueError("MATR trajectory, split and supervision cells must match exactly")
    prediction_cycles = tuple(range(cutoff_cycle + 1, 501))
    prepared: dict[str, tuple[list[float], float, list[float]]] = {}
    for cell_id in sorted(split_cells):
        cycles = trajectory_rows[cell_id]
        if set(cycles) != set(range(1, 501)):
            raise ValueError("MATR Hybrid supervision requires every real cycle 1 through 500")
        manifest_path = Path(processed_root) / "manifests" / f"{cell_id}.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("processed MATR cell manifest is missing")
        manifest = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())
        verified = verify_cell_artifacts(processed_root, manifest)
        evidence = supervision_by_cell[cell_id]
        if verified.metadata.source_sha256 != supervision.raw_sha256:
            raise ValueError("early input and Hybrid supervision raw source hashes differ")
        if verified.metadata.reference_capacity_ah != evidence.reference_capacity_ah:
            raise ValueError("early input and Hybrid reference capacities differ")
        records = _read_cutoff_records(verified.parquet_path, cutoff_cycle=cutoff_cycle)
        features = extract_early_cycle_features(
            records,
            config=EarlyCycleFeatureConfig(cutoff_cycle=cutoff_cycle),
        )
        values = [features.values[name] for name in _HYBRID_FEATURE_NAMES]
        if any(value is None for value in values):
            raise ValueError("MATR Hybrid requires complete capacity trend features")
        numeric_features = [float(value) for value in values if value is not None]
        reference_capacity = evidence.reference_capacity_ah
        initial_soh = cycles[cutoff_cycle]
        if not np.isclose(
            initial_soh,
            numeric_features[1] / reference_capacity,
            rtol=1e-4,
            atol=1e-6,
        ):
            raise ValueError("early capacity and supervision SOH disagree at the cutoff")
        prepared[cell_id] = (
            numeric_features,
            initial_soh,
            [cycles[cycle] for cycle in prediction_cycles],
        )

    return MatrHybridCohorts(
        train=_build_hybrid_batch(
            split_manifest.train, prepared, cutoff_cycle, prediction_cycles
        ),
        validation=_build_hybrid_batch(
            split_manifest.validation, prepared, cutoff_cycle, prediction_cycles
        ),
        calibration=_build_hybrid_batch(
            split_manifest.calibration, prepared, cutoff_cycle, prediction_cycles
        ),
        test=_build_hybrid_batch(
            split_manifest.test, prepared, cutoff_cycle, prediction_cycles
        ),
    )


def _build_batch(
    partition_cell_ids: tuple[str, ...],
    prepared: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    *,
    cutoff_cycle: int,
) -> CycleLifeCurveBatch:
    observed_cell_ids = tuple(cell_id for cell_id in partition_cell_ids if cell_id in prepared)
    if not observed_cell_ids:
        raise ValueError("MATR partition has no observed official cycle-life labels")
    return CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=observed_cell_ids,
        curve_values=torch.stack([prepared[cell_id][0] for cell_id in observed_cell_ids]),
        observed_mask=torch.stack([prepared[cell_id][1] for cell_id in observed_cell_ids]),
        observed_cycles=torch.tensor(
            [prepared[cell_id][2] for cell_id in observed_cell_ids],
            dtype=torch.float32,
        ),
        cutoff_cycle=cutoff_cycle,
    )


def _read_cutoff_records(path: Path, *, cutoff_cycle: int) -> tuple[CycleRecord, ...]:
    table = pq.read_table(path, filters=[("cycle_index", "<=", cutoff_cycle)])
    records = tuple(CycleRecord.model_validate(row) for row in table.to_pylist())
    if not records or any(record.cycle_index > cutoff_cycle for record in records):
        raise ValueError("early MATR input must contain records only through the cutoff")
    return records


def _build_hybrid_batch(
    cell_ids: tuple[str, ...],
    prepared: dict[str, tuple[list[float], float, list[float]]],
    cutoff_cycle: int,
    prediction_cycles: tuple[int, ...],
) -> HybridTrajectoryBatch:
    return HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=cell_ids,
        features=torch.tensor([prepared[cell][0] for cell in cell_ids], dtype=torch.float32),
        initial_soh=torch.tensor(
            [prepared[cell][1] for cell in cell_ids], dtype=torch.float32
        ),
        target_soh=torch.tensor(
            [prepared[cell][2] for cell in cell_ids], dtype=torch.float32
        ),
        prediction_cycles=prediction_cycles,
        cutoff_cycle=cutoff_cycle,
    )


def _verify_supervision_bytes(
    supervision_root: Path, supervision: MatrSupervisionArtifact
) -> Path:
    relative = PurePosixPath(supervision.parquet_relative_path)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in supervision.parquet_relative_path
    ):
        raise ValueError("MATR supervision path must be safe and relative")
    root = Path(supervision_root).resolve(strict=True)
    path = (root / Path(*relative.parts)).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("MATR supervision file must remain inside its approved root")
    if _sha256_file(path) != supervision.parquet_sha256:
        raise ValueError("MATR supervision Parquet SHA-256 mismatch")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
