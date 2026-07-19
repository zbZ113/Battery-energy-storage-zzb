"""Leakage-safe MATR data assembly for advanced model selection and final runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.matr_multibatch import (
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    combine_matr_batch_splits,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrCellConversionEvidence,
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.data.storage import ProcessedCellManifest, verify_cell_artifacts
from quanxin_life.features.early_cycle_sequence import (
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)
from quanxin_life.features.multichannel_cycle import (
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)
from quanxin_life.models.cyclepatch import stack_early_cycle_sequences
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Inputs,
    HybridPatchV2Targets,
)
from quanxin_life.training.advanced_tasks import (
    AdvancedCycleLifeBatch,
    AdvancedTrajectoryBatch,
)


@dataclass(frozen=True)
class AdvancedSelectionMatrData:
    """Selection data exposes only train and validation supervision."""

    source_split: SplitManifest
    scalar_train: AdvancedCycleLifeBatch
    scalar_validation: AdvancedCycleLifeBatch
    scalar_normalizer: EarlyCycleNormalizer
    hybrid_train: AdvancedTrajectoryBatch
    hybrid_validation: AdvancedTrajectoryBatch
    hybrid_normalizer: EarlyCycleNormalizer


@dataclass(frozen=True)
class AdvancedFinalMatrData(AdvancedSelectionMatrData):
    """Final evaluation data adds held-out calibration and test partitions."""

    scalar_calibration: AdvancedCycleLifeBatch
    scalar_test: AdvancedCycleLifeBatch
    hybrid_calibration: AdvancedTrajectoryBatch
    hybrid_test: AdvancedTrajectoryBatch


ScalarSequence = tuple[EarlyCycleSequence, float]
HybridSequence = tuple[EarlyCycleSequence, dict[int, float]]


def load_advanced_matr_selection_data(
    *,
    project_root: Path,
    manifest: MatrThreeBatchManifest,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    feature_version: str,
) -> AdvancedSelectionMatrData:
    result = _load_advanced_matr_data(
        project_root=project_root,
        manifest=manifest,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
        feature_version=feature_version,
        partitions=("train", "validation"),
    )
    if isinstance(result, AdvancedFinalMatrData):
        raise RuntimeError("selection loading exposed held-out partitions")
    return result


def load_advanced_matr_final_data(
    *,
    project_root: Path,
    manifest: MatrThreeBatchManifest,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    feature_version: str,
) -> AdvancedFinalMatrData:
    result = _load_advanced_matr_data(
        project_root=project_root,
        manifest=manifest,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
        feature_version=feature_version,
        partitions=("train", "validation", "calibration", "test"),
    )
    if not isinstance(result, AdvancedFinalMatrData):
        raise RuntimeError("final loading did not expose held-out partitions")
    return result


def _load_advanced_matr_data(
    *,
    project_root: Path,
    manifest: MatrThreeBatchManifest,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    feature_version: str,
    partitions: tuple[str, ...],
) -> AdvancedSelectionMatrData | AdvancedFinalMatrData:
    root = project_root.resolve(strict=True)
    manifest = MatrThreeBatchManifest.model_validate(manifest.model_dump(mode="python"))
    combined_path = _verified_registered_file(
        root, manifest.combined_split_manifest, manifest.combined_split_sha256
    )
    if SplitManifest.model_validate_json(combined_path.read_bytes()) != combined_split:
        raise ValueError("combined split differs from the registered manifest")
    config = MultichannelCycleConfig(cutoff_cycle=cutoff_cycle, feature_version=feature_version)
    scalar_by_cell: dict[str, ScalarSequence] = {}
    hybrid_by_cell: dict[str, HybridSequence] = {}
    component_splits: list[SplitManifest] = []
    for component in manifest.batches:
        conversion = MatrBatchConversionReport.model_validate_json(
            _verified_registered_file(
                root, component.conversion_report, component.conversion_report_sha256
            ).read_bytes()
        )
        split = SplitManifest.model_validate_json(
            _verified_registered_file(
                root, component.split_manifest, component.split_manifest_sha256
            ).read_bytes()
        )
        eligibility = MatrTrajectoryEligibilityAudit.model_validate_json(
            _verified_registered_file(
                root, component.eligibility_report, component.eligibility_report_sha256
            ).read_bytes()
        )
        supervision = MatrSupervisionArtifact.model_validate_json(
            _verified_registered_file(
                root, component.supervision_report, component.supervision_report_sha256
            ).read_bytes()
        )
        _validate_component_contract(component, conversion, split, eligibility, supervision)
        component_splits.append(split)
        allowed = tuple(
            cell_id for partition in partitions for cell_id in getattr(split, partition)
        )
        evidence = {cell.cell_id: cell for cell in conversion.cells}
        if not set(allowed) <= set(evidence):
            raise ValueError("requested cells are absent from conversion evidence")
        scalar_cells = tuple(
            cell_id for cell_id in allowed if _is_scalar_target(evidence[cell_id], cutoff_cycle)
        )
        eligible = set(eligibility.eligible_cell_ids)
        hybrid_cells = tuple(cell_id for cell_id in allowed if cell_id in eligible)
        sequence_cells = tuple(dict.fromkeys((*scalar_cells, *hybrid_cells)))
        processed_root = _inside(root, component.processed_root)
        sequences = {
            cell_id: _load_early_sequence(
                processed_root=processed_root,
                evidence=evidence[cell_id],
                raw_sha256=component.raw_sha256,
                config=config,
                data_version=manifest.data_version,
            )
            for cell_id in sequence_cells
        }
        for cell_id in scalar_cells:
            label = evidence[cell_id].official_life_label
            if label is None:
                raise ValueError("observed official cycle-life target is missing")
            scalar_by_cell[cell_id] = (sequences[cell_id], float(label))
        if hybrid_cells:
            supervision_root = _inside(root, component.supervision_root)
            relative = _safe_relative(supervision.parquet_relative_path, "supervision parquet")
            supervision_path = (supervision_root / Path(*relative.parts)).resolve(strict=True)
            if not supervision_path.is_relative_to(supervision_root):
                raise ValueError("supervision parquet escapes supervision_root")
            trajectories = _load_supervision_rows(
                supervision_path=supervision_path,
                supervision=supervision,
                allowed_cell_ids=hybrid_cells,
                allowed_evidence={
                    cell_id: evidence[cell_id] for cell_id in hybrid_cells
                },
            )
            for cell_id in hybrid_cells:
                hybrid_by_cell[cell_id] = (sequences[cell_id], trajectories[cell_id])
    if combine_matr_batch_splits(tuple(component_splits)) != combined_split:
        raise ValueError("component splits do not reconstruct the combined split")
    scalar_partitions = {
        partition: tuple(
            scalar_by_cell[cell_id]
            for cell_id in getattr(combined_split, partition)
            if cell_id in scalar_by_cell
        )
        for partition in partitions
    }
    hybrid_partitions = {
        partition: tuple(
            hybrid_by_cell[cell_id]
            for cell_id in getattr(combined_split, partition)
            if cell_id in hybrid_by_cell
        )
        for partition in partitions
    }
    return _assemble_advanced_matr_data(
        source_split=combined_split,
        scalar_sequences=scalar_partitions,
        hybrid_sequences=hybrid_partitions,
        final=len(partitions) == 4,
    )


def _assemble_advanced_matr_data(
    *,
    source_split: SplitManifest,
    scalar_sequences: dict[str, tuple[ScalarSequence, ...]],
    hybrid_sequences: dict[str, tuple[HybridSequence, ...]],
    final: bool,
) -> AdvancedSelectionMatrData | AdvancedFinalMatrData:
    """Fit task-local train normalizers and construct closed partition batches."""

    expected = (
        {"train", "validation", "calibration", "test"}
        if final
        else {
            "train",
            "validation",
        }
    )
    if set(scalar_sequences) != expected or set(hybrid_sequences) != expected:
        raise ValueError("advanced MATR partitions do not match the requested run mode")
    scalar_train_sequences = tuple(item[0] for item in scalar_sequences["train"])
    hybrid_train_sequences = tuple(item[0] for item in hybrid_sequences["train"])
    scalar_normalizer = EarlyCycleNormalizer.fit(
        scalar_train_sequences,
        training_cell_ids=frozenset(sequence.cell_id for sequence in scalar_train_sequences),
    )
    hybrid_normalizer = EarlyCycleNormalizer.fit(
        hybrid_train_sequences,
        training_cell_ids=frozenset(sequence.cell_id for sequence in hybrid_train_sequences),
    )

    scalar_batches = {
        partition: _build_scalar_batch(entries, scalar_normalizer)
        for partition, entries in scalar_sequences.items()
    }
    hybrid_batches = {
        partition: _build_hybrid_batch(entries, hybrid_normalizer)
        for partition, entries in hybrid_sequences.items()
    }
    if not final:
        return AdvancedSelectionMatrData(
            source_split=source_split,
            scalar_train=scalar_batches["train"],
            scalar_validation=scalar_batches["validation"],
            scalar_normalizer=scalar_normalizer,
            hybrid_train=hybrid_batches["train"],
            hybrid_validation=hybrid_batches["validation"],
            hybrid_normalizer=hybrid_normalizer,
        )
    return AdvancedFinalMatrData(
        source_split=source_split,
        scalar_train=scalar_batches["train"],
        scalar_validation=scalar_batches["validation"],
        scalar_normalizer=scalar_normalizer,
        hybrid_train=hybrid_batches["train"],
        hybrid_validation=hybrid_batches["validation"],
        hybrid_normalizer=hybrid_normalizer,
        scalar_calibration=scalar_batches["calibration"],
        scalar_test=scalar_batches["test"],
        hybrid_calibration=hybrid_batches["calibration"],
        hybrid_test=hybrid_batches["test"],
    )


def _build_scalar_batch(
    entries: tuple[ScalarSequence, ...], normalizer: EarlyCycleNormalizer
) -> AdvancedCycleLifeBatch:
    if not entries:
        raise ValueError("every requested scalar partition requires supervised cells")
    normalized = tuple(normalizer.transform(sequence) for sequence, _label in entries)
    return AdvancedCycleLifeBatch(
        early_batch=stack_early_cycle_sequences(normalized),
        raw_labels=torch.tensor([label for _sequence, label in entries], dtype=torch.float32),
    )


def _build_hybrid_batch(
    entries: tuple[HybridSequence, ...], normalizer: EarlyCycleNormalizer
) -> AdvancedTrajectoryBatch:
    if not entries:
        raise ValueError("every requested Hybrid partition requires eligible cells")
    normalized = tuple(normalizer.transform(sequence) for sequence, _trajectory in entries)
    early = stack_early_cycle_sequences(normalized)
    cutoff = int(early.cycle_mask.shape[1] - 1)
    prediction_cycles = torch.arange(cutoff + 1, 501, dtype=torch.int64)
    history = torch.full(early.cycle_mask.shape, float("nan"), dtype=torch.float32)
    history_mask = early.cycle_mask.clone()
    history_mask[:, 0] = False
    target = torch.empty((len(entries), 500 - cutoff), dtype=torch.float32)
    initial = torch.empty(len(entries), dtype=torch.float32)
    for row, (_sequence, trajectory) in enumerate(entries):
        if set(trajectory) != set(range(1, 501)):
            raise ValueError("Hybrid supervision must contain every real cycle 1 through 500")
        initial[row] = trajectory[cutoff]
        for cycle in range(1, cutoff + 1):
            if history_mask[row, cycle]:
                history[row, cycle] = trajectory[cycle]
        target[row] = torch.tensor(
            [trajectory[cycle] for cycle in range(cutoff + 1, 501)],
            dtype=torch.float32,
        )
    return AdvancedTrajectoryBatch(
        inputs=HybridPatchV2Inputs(
            early_batch=early,
            initial_soh=initial,
            prediction_cycles=prediction_cycles,
        ),
        targets=HybridPatchV2Targets(
            history_soh=history,
            history_mask=history_mask,
            target_soh=target,
            target_mask=torch.ones_like(target, dtype=torch.bool),
        ),
    )


def _validate_component_contract(
    component: MatrBatchArtifactReference,
    conversion: MatrBatchConversionReport,
    split: SplitManifest,
    eligibility: MatrTrajectoryEligibilityAudit,
    supervision: MatrSupervisionArtifact,
) -> None:
    conversion_by_cell = {cell.cell_id: cell for cell in conversion.cells}
    conversion_cells = set(conversion_by_cell)
    eligible = set(eligibility.eligible_cell_ids)
    excluded_by_cell = {item.cell_id: item for item in eligibility.excluded}
    excluded = set(excluded_by_cell)
    supervision_by_cell = {cell.cell_id: cell for cell in supervision.cells}
    if (
        conversion.batch_index != component.batch_index
        or conversion.raw_sha256 != component.raw_sha256
        or conversion.cell_count != component.cell_count
        or conversion.cell_count != len(conversion.cells)
        or split.dataset_id != "MATR"
        or set(split.all_cells) != conversion_cells
        or eligibility.batch_index != component.batch_index
        or eligibility.horizon_cycle != 500
        or eligible | excluded != conversion_cells
        or eligible & excluded
        or len(eligible) != component.hybrid_eligible_count
        or supervision.raw_sha256 != component.raw_sha256
        or supervision.source_report_sha256 != sha256_canonical(conversion.model_dump(mode="json"))
        or supervision.horizon_cycle != 500
        or {cell.cell_id for cell in supervision.cells} != eligible
        or sum(not cell.official_life_right_censored for cell in conversion.cells)
        != component.scalar_label_count
        or any(
            not _supervision_evidence_matches_conversion(
                supervision_by_cell[cell_id], conversion_by_cell[cell_id]
            )
            for cell_id in eligible
        )
        or any(
            supervision_by_cell[cell_id].observed_cycle_count
            <= eligibility.horizon_cycle
            for cell_id in eligible
        )
        or any(
            not 0
            < excluded_by_cell[cell_id].observed_cycle_count
            <= eligibility.horizon_cycle
            for cell_id in excluded
        )
    ):
        raise ValueError("MATR component artifacts are inconsistent")


def _supervision_evidence_matches_conversion(
    supervision: MatrSupervisionCellEvidence,
    conversion: MatrCellConversionEvidence,
) -> bool:
    return bool(
        supervision.raw_cell_id == conversion.raw_cell_id
        and supervision.official_life_label == conversion.official_life_label
        and supervision.official_life_right_censored
        == conversion.official_life_right_censored
        and supervision.reference_capacity_ah == conversion.reference_capacity_ah
    )


def _is_scalar_target(evidence: MatrCellConversionEvidence, cutoff_cycle: int) -> bool:
    return (
        not evidence.official_life_right_censored
        and evidence.official_life_label is not None
        and evidence.official_life_label > cutoff_cycle
    )


def _verified_registered_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = _inside(root, relative)
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered MATR evidence must be a regular file")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("registered MATR evidence SHA-256 mismatch")
    return path


def _inside(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative, "registered MATR path")
    path = (root / Path(*safe.parts)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("registered MATR path escapes project_root")
    return path


def _load_early_sequence(
    *,
    processed_root: Path,
    evidence: MatrCellConversionEvidence,
    raw_sha256: str,
    config: MultichannelCycleConfig,
    data_version: str,
) -> EarlyCycleSequence:
    """Load only cutoff-bounded rows before invoking the label-free builder."""

    root = processed_root.resolve(strict=True)
    relative = _safe_relative(evidence.manifest_relative_path, "processed manifest")
    manifest_path = (root / Path(*relative.parts)).resolve(strict=True)
    if not manifest_path.is_relative_to(root) or manifest_path.is_symlink():
        raise ValueError("processed manifest must remain inside processed_root")
    manifest = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())
    if (
        manifest.cell_id != evidence.cell_id
        or manifest.parquet_sha256 != evidence.parquet_sha256
        or manifest.metadata_sha256 != evidence.metadata_sha256
        or manifest.row_count != evidence.row_count
    ):
        raise ValueError("processed cell manifest differs from conversion evidence")
    verified = verify_cell_artifacts(root, manifest)
    if (
        verified.metadata.source_sha256 != raw_sha256
        or verified.metadata.official_life_label != evidence.official_life_label
        or verified.metadata.reference_capacity_ah != evidence.reference_capacity_ah
    ):
        raise ValueError("processed cell metadata differs from registered MATR evidence")
    table = pq.read_table(
        verified.parquet_path,
        filters=[("cycle_index", "<=", config.cutoff_cycle)],
    )
    records = tuple(CycleRecord.model_validate(row) for row in table.to_pylist())
    if not records or any(record.cycle_index > config.cutoff_cycle for record in records):
        raise ValueError("early MATR input must contain only cutoff-bounded rows")
    return build_early_cycle_sequence(records, config=config, data_version=data_version)


def _load_supervision_rows(
    *,
    supervision_path: Path,
    supervision: MatrSupervisionArtifact,
    allowed_cell_ids: tuple[str, ...],
    allowed_evidence: dict[str, MatrCellConversionEvidence],
) -> dict[str, dict[int, float]]:
    """Read and validate only the explicitly allowed partition rows."""

    if not allowed_cell_ids or len(set(allowed_cell_ids)) != len(allowed_cell_ids):
        raise ValueError("allowed supervision cells must be non-empty and unique")
    approved = {cell.cell_id for cell in supervision.cells}
    if not set(allowed_cell_ids) <= approved:
        raise ValueError("allowed supervision cells are absent from the artifact")
    if set(allowed_evidence) != set(allowed_cell_ids):
        raise ValueError("allowed supervision evidence must exactly match selected cells")
    supervision_evidence = {cell.cell_id: cell for cell in supervision.cells}
    if any(
        not _supervision_evidence_matches_conversion(
            supervision_evidence[cell_id], allowed_evidence[cell_id]
        )
        for cell_id in allowed_cell_ids
    ):
        raise ValueError("selected supervision evidence differs from conversion evidence")
    if any(
        supervision_evidence[cell_id].observed_cycle_count
        <= supervision.horizon_cycle
        for cell_id in allowed_cell_ids
    ):
        raise ValueError("selected supervision evidence does not cover the full horizon")
    if supervision_path.is_symlink() or not supervision_path.is_file():
        raise ValueError("supervision Parquet must be a regular file")
    if _sha256_file(supervision_path) != supervision.parquet_sha256:
        raise ValueError("supervision Parquet SHA-256 mismatch")
    table = pq.read_table(
        supervision_path,
        filters=[("cell_id", "in", list(allowed_cell_ids))],
    )
    expected_columns = {
        "dataset_id",
        "cell_id",
        "cycle_index",
        "discharge_capacity_ah",
        "reference_capacity_ah",
        "soh",
    }
    if set(table.schema.names) != expected_columns:
        raise ValueError("MATR supervision Parquet schema is invalid")
    rows: dict[str, dict[int, float]] = {}
    for row in table.to_pylist():
        cell_id = str(row["cell_id"])
        if row["dataset_id"] != "MATR" or cell_id not in allowed_cell_ids:
            raise ValueError("MATR supervision contains an invalid selected row")
        cycle = int(row["cycle_index"])
        try:
            discharge_capacity = float(row["discharge_capacity_ah"])
            reference_capacity = float(row["reference_capacity_ah"])
            soh = float(row["soh"])
        except (TypeError, ValueError) as exc:
            raise ValueError("MATR supervision contains an invalid selected row") from exc
        expected_reference = allowed_evidence[cell_id].reference_capacity_ah
        if (
            not 1 <= cycle <= supervision.horizon_cycle
            or not math.isfinite(discharge_capacity)
            or discharge_capacity < 0
            or not math.isfinite(reference_capacity)
            or reference_capacity <= 0
            or not math.isclose(
                reference_capacity,
                expected_reference,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isfinite(soh)
            or soh < 0
            or not math.isclose(
                soh,
                discharge_capacity / reference_capacity,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("MATR supervision contains an invalid selected row")
        by_cycle = rows.setdefault(cell_id, {})
        if cycle in by_cycle:
            raise ValueError("MATR supervision contains a duplicate selected cell-cycle")
        by_cycle[cycle] = soh
    expected_cycles = set(range(1, supervision.horizon_cycle + 1))
    if set(rows) != set(allowed_cell_ids) or any(
        set(values) != expected_cycles for values in rows.values()
    ):
        raise ValueError("selected MATR supervision must cover every real cycle")
    return {cell_id: rows[cell_id] for cell_id in allowed_cell_ids}


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
