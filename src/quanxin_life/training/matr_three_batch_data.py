"""Verified assembly of the governed three-batch MATR training cohorts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from quanxin_life.data.matr_multibatch import (
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    combine_matr_batch_splits,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrSupervisionArtifact,
)
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.training.matr_data import (
    MatrCurveCohorts,
    MatrHybridCohorts,
    MatrOfficialLifeEvidence,
    is_matr_official_life_future_target,
    load_matr_cycle_life_curve_cohorts_from_evidence,
    load_matr_hybrid_trajectory_cohorts,
    merge_matr_curve_cohorts,
    merge_matr_hybrid_cohorts,
    restrict_matr_split_to_cells,
)


def load_matr_three_batch_training_cohorts(
    *,
    project_root: Path,
    manifest: MatrThreeBatchManifest,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    voltage_min_v: float = 2.0,
    voltage_max_v: float = 3.6,
    voltage_grid_step_v: float = 0.01,
) -> tuple[MatrCurveCohorts, MatrHybridCohorts]:
    """Load scalar and trajectory tasks from their distinct approved inventories."""

    root = project_root.resolve(strict=True)
    combined_path = _verified_file(
        root,
        manifest.combined_split_manifest,
        manifest.combined_split_sha256,
    )
    disk_combined = SplitManifest.model_validate_json(combined_path.read_bytes())
    if disk_combined != combined_split:
        raise ValueError("combined MATR split differs from the registered manifest")

    curve_components: list[MatrCurveCohorts] = []
    hybrid_components: list[MatrHybridCohorts] = []
    component_splits: list[SplitManifest] = []
    expected_scalar_count = 0
    for component in manifest.batches:
        conversion = MatrBatchConversionReport.model_validate_json(
            _verified_file(
                root,
                component.conversion_report,
                component.conversion_report_sha256,
            ).read_bytes()
        )
        split = SplitManifest.model_validate_json(
            _verified_file(
                root,
                component.split_manifest,
                component.split_manifest_sha256,
            ).read_bytes()
        )
        eligibility = MatrTrajectoryEligibilityAudit.model_validate_json(
            _verified_file(
                root,
                component.eligibility_report,
                component.eligibility_report_sha256,
            ).read_bytes()
        )
        supervision = MatrSupervisionArtifact.model_validate_json(
            _verified_file(
                root,
                component.supervision_report,
                component.supervision_report_sha256,
            ).read_bytes()
        )
        _validate_component(
            component_batch_index=component.batch_index,
            component_raw_sha256=component.raw_sha256,
            component_cell_count=component.cell_count,
            component_scalar_count=component.scalar_label_count,
            component_hybrid_count=component.hybrid_eligible_count,
            conversion=conversion,
            split=split,
            eligibility=eligibility,
            supervision=supervision,
        )
        scalar_evidence = tuple(
            MatrOfficialLifeEvidence(
                cell_id=cell.cell_id,
                official_life_label=cell.official_life_label,
                official_life_right_censored=cell.official_life_right_censored,
                reference_capacity_ah=cell.reference_capacity_ah,
            )
            for cell in conversion.cells
        )
        expected_scalar_count += sum(
            is_matr_official_life_future_target(
                cell,
                cutoff_cycle=cutoff_cycle,
            )
            for cell in scalar_evidence
        )
        curve_components.append(
            load_matr_cycle_life_curve_cohorts_from_evidence(
                processed_root=_inside(root, component.processed_root),
                raw_sha256=component.raw_sha256,
                evidence=scalar_evidence,
                split_manifest=split,
                cutoff_cycle=cutoff_cycle,
                voltage_min_v=voltage_min_v,
                voltage_max_v=voltage_max_v,
                voltage_grid_step_v=voltage_grid_step_v,
            )
        )
        hybrid_split = restrict_matr_split_to_cells(
            split,
            set(eligibility.eligible_cell_ids),
        )
        hybrid_components.append(
            load_matr_hybrid_trajectory_cohorts(
                processed_root=_inside(root, component.processed_root),
                supervision_root=_inside(root, component.supervision_root),
                supervision=supervision,
                split_manifest=hybrid_split,
                cutoff_cycle=cutoff_cycle,
            )
        )
        component_splits.append(split)

    rebuilt = combine_matr_batch_splits(tuple(component_splits))
    if rebuilt != combined_split:
        raise ValueError("combined MATR split does not preserve component assignments")
    curves = merge_matr_curve_cohorts(tuple(curve_components))
    hybrid = merge_matr_hybrid_cohorts(tuple(hybrid_components))
    scalar_count = sum(
        len(getattr(curves, partition).cell_ids)
        for partition in ("train", "validation", "calibration", "test")
    )
    hybrid_count = sum(
        len(getattr(hybrid, partition).cell_ids)
        for partition in ("train", "validation", "calibration", "test")
    )
    if scalar_count != expected_scalar_count:
        raise ValueError(
            "loaded MATR scalar count differs from cutoff-eligible source evidence"
        )
    if hybrid_count != manifest.hybrid_eligible_count:
        raise ValueError("loaded MATR Hybrid count differs from the three-batch manifest")
    return curves, hybrid


def _validate_component(
    *,
    component_batch_index: int,
    component_raw_sha256: str,
    component_cell_count: int,
    component_scalar_count: int,
    component_hybrid_count: int,
    conversion: MatrBatchConversionReport,
    split: SplitManifest,
    eligibility: MatrTrajectoryEligibilityAudit,
    supervision: MatrSupervisionArtifact,
) -> None:
    conversion_cells = {cell.cell_id for cell in conversion.cells}
    eligible = set(eligibility.eligible_cell_ids)
    excluded = {item.cell_id for item in eligibility.excluded}
    if (
        conversion.batch_index != component_batch_index
        or eligibility.batch_index != component_batch_index
        or conversion.raw_sha256 != component_raw_sha256
        or supervision.raw_sha256 != component_raw_sha256
        or conversion.cell_count != component_cell_count
        or split.dataset_id != "MATR"
        or set(split.all_cells) != conversion_cells
        or eligible | excluded != conversion_cells
        or eligible & excluded
        or len(eligible) != component_hybrid_count
        or sum(not cell.official_life_right_censored for cell in conversion.cells)
        != component_scalar_count
        or {cell.cell_id for cell in supervision.cells} != eligible
        or supervision.horizon_cycle != 500
    ):
        raise ValueError("MATR three-batch component context is inconsistent")


def _verified_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = _inside(root, relative)
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered MATR evidence must be a regular file")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("registered MATR evidence SHA-256 mismatch")
    return path


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("MATR three-batch path escapes the project root")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
