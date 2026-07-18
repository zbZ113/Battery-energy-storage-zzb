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

_MATR_TIME_MONOTONIC_TOLERANCE_S = 1e-9

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


@dataclass(frozen=True)
class MatrOfficialLifeEvidence:
    """Hash-context-bound scalar label evidence independent of trajectory eligibility."""

    cell_id: str
    official_life_label: int | None
    official_life_right_censored: bool
    reference_capacity_ah: float

    def __post_init__(self) -> None:
        if not self.cell_id or self.reference_capacity_ah <= 0:
            raise ValueError("MATR official life evidence is invalid")
        if self.official_life_right_censored == (
            self.official_life_label is not None
        ):
            raise ValueError("MATR official life label and censoring flag disagree")


def is_matr_official_life_future_target(
    evidence: MatrOfficialLifeEvidence,
    *,
    cutoff_cycle: int,
) -> bool:
    """Return whether an observed official life remains strictly after the cutoff."""

    if cutoff_cycle < 0:
        raise ValueError("cutoff_cycle must be non-negative")
    return (
        not evidence.official_life_right_censored
        and evidence.official_life_label is not None
        and evidence.official_life_label > cutoff_cycle
    )


def restrict_matr_split_to_cells(
    split: SplitManifest,
    selected_cell_ids: set[str],
) -> SplitManifest:
    """Filter task-ineligible cells without moving any selected cell across partitions."""

    if split.dataset_id != "MATR" or not selected_cell_ids:
        raise ValueError("restricted MATR split requires selected MATR cells")
    if not selected_cell_ids.issubset(set(split.all_cells)):
        raise ValueError("restricted MATR split contains cells outside the source split")
    partitions = tuple(
        tuple(cell for cell in partition if cell in selected_cell_ids)
        for partition in (split.train, split.validation, split.calibration, split.test)
    )
    if any(not partition for partition in partitions):
        raise ValueError("every restricted MATR partition must retain eligible cells")
    return SplitManifest(
        dataset_id="MATR",
        seed=split.seed,
        train=partitions[0],
        validation=partitions[1],
        calibration=partitions[2],
        test=partitions[3],
    )


def merge_matr_curve_cohorts(
    components: tuple[MatrCurveCohorts, ...],
) -> MatrCurveCohorts:
    """Join batch-local scalar cohorts while preserving split membership."""

    if len(components) != 3:
        raise ValueError("MATR three-batch curves require exactly three components")
    return MatrCurveCohorts(
        train=_merge_curve_batches(tuple(item.train for item in components)),
        validation=_merge_curve_batches(tuple(item.validation for item in components)),
        calibration=_merge_curve_batches(tuple(item.calibration for item in components)),
        test=_merge_curve_batches(tuple(item.test for item in components)),
    )


def merge_matr_hybrid_cohorts(
    components: tuple[MatrHybridCohorts, ...],
) -> MatrHybridCohorts:
    """Join only batch-local cells with genuine trajectories through cycle 500."""

    if len(components) != 3:
        raise ValueError("MATR three-batch Hybrid requires exactly three components")
    return MatrHybridCohorts(
        train=_merge_hybrid_batches(tuple(item.train for item in components)),
        validation=_merge_hybrid_batches(
            tuple(item.validation for item in components)
        ),
        calibration=_merge_hybrid_batches(
            tuple(item.calibration for item in components)
        ),
        test=_merge_hybrid_batches(tuple(item.test for item in components)),
    )


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
    evidence = tuple(
        MatrOfficialLifeEvidence(
            cell_id=cell.cell_id,
            official_life_label=cell.official_life_label,
            official_life_right_censored=cell.official_life_right_censored,
            reference_capacity_ah=cell.reference_capacity_ah,
        )
        for cell in supervision.cells
    )
    return load_matr_cycle_life_curve_cohorts_from_evidence(
        processed_root=processed_root,
        raw_sha256=supervision.raw_sha256,
        evidence=evidence,
        split_manifest=split_manifest,
        cutoff_cycle=cutoff_cycle,
        voltage_min_v=voltage_min_v,
        voltage_max_v=voltage_max_v,
        voltage_grid_step_v=voltage_grid_step_v,
    )


def load_matr_cycle_life_curve_cohorts_from_evidence(
    *,
    processed_root: Path,
    raw_sha256: str,
    evidence: tuple[MatrOfficialLifeEvidence, ...],
    split_manifest: SplitManifest,
    cutoff_cycle: int,
    voltage_min_v: float,
    voltage_max_v: float,
    voltage_grid_step_v: float,
) -> MatrCurveCohorts:
    """Build scalar cohorts from conversion evidence, not trajectory membership."""

    if split_manifest.dataset_id != "MATR":
        raise ValueError("MATR curve loading requires a MATR split")
    evidence_by_cell = {cell.cell_id: cell for cell in evidence}
    if len(evidence_by_cell) != len(evidence):
        raise ValueError("MATR official life evidence cell identifiers must be unique")
    split_cells = set(
        (
            *split_manifest.train,
            *split_manifest.validation,
            *split_manifest.calibration,
            *split_manifest.test,
        )
    )
    if split_cells != set(evidence_by_cell):
        raise ValueError("MATR split cells must exactly match scalar label evidence")

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
        time_monotonic_tolerance_s=_MATR_TIME_MONOTONIC_TOLERANCE_S,
    )
    prepared: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for cell_id in sorted(split_cells):
        label = evidence_by_cell[cell_id]
        if not is_matr_official_life_future_target(
            label,
            cutoff_cycle=cutoff_cycle,
        ):
            continue
        manifest = manifests[cell_id]
        verified = verify_cell_artifacts(processed_root, manifest)
        if verified.metadata.source_sha256 != raw_sha256:
            raise ValueError("early input and scalar evidence raw source hashes differ")
        if (
            verified.metadata.official_life_label != label.official_life_label
            or verified.metadata.reference_capacity_ah != label.reference_capacity_ah
        ):
            raise ValueError("early metadata and scalar label evidence differ")
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
            config=EarlyCycleFeatureConfig(
                cutoff_cycle=cutoff_cycle,
                time_monotonic_tolerance_s=_MATR_TIME_MONOTONIC_TOLERANCE_S,
            ),
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


def _merge_curve_batches(
    batches: tuple[CycleLifeCurveBatch, ...],
) -> CycleLifeCurveBatch:
    first = batches[0]
    cell_ids = tuple(cell for batch in batches for cell in batch.cell_ids)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("merged MATR curve cell identifiers must be globally unique")
    if any(
        batch.dataset_id != first.dataset_id
        or batch.target is not first.target
        or batch.cutoff_cycle != first.cutoff_cycle
        or batch.curve_values.shape[1:] != first.curve_values.shape[1:]
        for batch in batches[1:]
    ):
        raise ValueError("merged MATR curve batch schemas must agree")
    return CycleLifeCurveBatch(
        dataset_id=first.dataset_id,
        target=first.target,
        cell_ids=cell_ids,
        curve_values=torch.cat(tuple(batch.curve_values for batch in batches), dim=0),
        observed_mask=torch.cat(tuple(batch.observed_mask for batch in batches), dim=0),
        observed_cycles=torch.cat(
            tuple(batch.observed_cycles for batch in batches), dim=0
        ),
        cutoff_cycle=first.cutoff_cycle,
    )


def _merge_hybrid_batches(
    batches: tuple[HybridTrajectoryBatch, ...],
) -> HybridTrajectoryBatch:
    first = batches[0]
    cell_ids = tuple(cell for batch in batches for cell in batch.cell_ids)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("merged MATR Hybrid cell identifiers must be globally unique")
    if any(
        batch.dataset_id != first.dataset_id
        or batch.cutoff_cycle != first.cutoff_cycle
        or batch.prediction_cycles != first.prediction_cycles
        or batch.features.shape[1:] != first.features.shape[1:]
        or batch.target_soh.shape[1:] != first.target_soh.shape[1:]
        for batch in batches[1:]
    ):
        raise ValueError("merged MATR Hybrid batch schemas must agree")
    return HybridTrajectoryBatch(
        dataset_id=first.dataset_id,
        cell_ids=cell_ids,
        features=torch.cat(tuple(batch.features for batch in batches), dim=0),
        initial_soh=torch.cat(tuple(batch.initial_soh for batch in batches), dim=0),
        target_soh=torch.cat(tuple(batch.target_soh for batch in batches), dim=0),
        prediction_cycles=first.prediction_cycles,
        cutoff_cycle=first.cutoff_cycle,
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
