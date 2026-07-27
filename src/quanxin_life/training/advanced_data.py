"""Leakage-safe MATR data assembly for advanced model selection and final runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
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
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.early_cycle_sequence import (
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)
from quanxin_life.features.multichannel_cycle import (
    MultichannelCycleConfig,
)
from quanxin_life.features.verified_matr_sequence import (
    load_verified_matr_early_sequence,
)
from quanxin_life.models.cyclepatch import stack_early_cycle_sequences
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Inputs,
    HybridPatchV2Targets,
)
from quanxin_life.training.advanced_cache import (
    EarlyCycleSequenceCache,
)
from quanxin_life.training.advanced_tasks import (
    AdvancedCycleLifeBatch,
    AdvancedTrajectoryBatch,
)


@dataclass(frozen=True, order=True)
class MaskedCycleAuditEntry:
    """One explicitly excluded real cycle and the scientific exclusion reason."""

    cell_id: str
    cycle_index: int
    reason: str

    def __post_init__(self) -> None:
        if not self.cell_id.strip() or not self.reason.strip():
            raise ValueError("masked cycle audit identifiers must be non-empty")
        if self.cycle_index < 0:
            raise ValueError("masked cycle audit cycle_index must be non-negative")


@dataclass(frozen=True)
class MaskedCycleAudit:
    """Canonical, hash-bound record of all cycles excluded during data assembly."""

    entries: tuple[MaskedCycleAuditEntry, ...] = ()
    masked_cycle_count: int = field(init=False)
    audit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        entries = tuple(sorted(self.entries))
        if len(set(entries)) != len(entries):
            raise ValueError("masked cycle audit entries must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "masked_cycle_count",
            len({(entry.cell_id, entry.cycle_index) for entry in entries}),
        )
        object.__setattr__(
            self,
            "audit_sha256",
            sha256_canonical(
                [
                    {
                        "cell_id": entry.cell_id,
                        "cycle_index": entry.cycle_index,
                        "reason": entry.reason,
                    }
                    for entry in entries
                ]
            ),
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
    masked_cycle_audit: MaskedCycleAudit


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
    sequence_cache = EarlyCycleSequenceCache(
        root / "data" / "cache" / "advanced_sequences"
    )
    scalar_by_cell: dict[str, ScalarSequence] = {}
    hybrid_by_cell: dict[str, HybridSequence] = {}
    masked_cycle_entries: list[MaskedCycleAuditEntry] = []
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
        sequences: dict[str, EarlyCycleSequence] = {}
        for cell_id in sequence_cells:
            sequences[cell_id] = _load_early_sequence(
                processed_root=processed_root,
                evidence=evidence[cell_id],
                reference_capacity_ah=evidence[cell_id].reference_capacity_ah,
                raw_sha256=component.raw_sha256,
                config=config,
                data_version=manifest.data_version,
                audit_entries=masked_cycle_entries,
                cache=sequence_cache,
            )
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
        masked_cycle_entries=tuple(masked_cycle_entries),
        final=len(partitions) == 4,
    )


def _assemble_advanced_matr_data(
    *,
    source_split: SplitManifest,
    scalar_sequences: dict[str, tuple[ScalarSequence, ...]],
    hybrid_sequences: dict[str, tuple[HybridSequence, ...]],
    final: bool,
    masked_cycle_entries: tuple[MaskedCycleAuditEntry, ...] = (),
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
    audit_entries = list(masked_cycle_entries)
    hybrid_batches = {
        partition: _build_hybrid_batch(
            entries,
            hybrid_normalizer,
            audit_entries=audit_entries,
        )
        for partition, entries in hybrid_sequences.items()
    }
    masked_cycle_audit = MaskedCycleAudit(entries=tuple(audit_entries))
    if not final:
        return AdvancedSelectionMatrData(
            source_split=source_split,
            scalar_train=scalar_batches["train"],
            scalar_validation=scalar_batches["validation"],
            scalar_normalizer=scalar_normalizer,
            hybrid_train=hybrid_batches["train"],
            hybrid_validation=hybrid_batches["validation"],
            hybrid_normalizer=hybrid_normalizer,
            masked_cycle_audit=masked_cycle_audit,
        )
    return AdvancedFinalMatrData(
        source_split=source_split,
        scalar_train=scalar_batches["train"],
        scalar_validation=scalar_batches["validation"],
        scalar_normalizer=scalar_normalizer,
        hybrid_train=hybrid_batches["train"],
        hybrid_validation=hybrid_batches["validation"],
        hybrid_normalizer=hybrid_normalizer,
        masked_cycle_audit=masked_cycle_audit,
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
    entries: tuple[HybridSequence, ...],
    normalizer: EarlyCycleNormalizer,
    *,
    audit_entries: list[MaskedCycleAuditEntry] | None = None,
) -> AdvancedTrajectoryBatch:
    if not entries:
        raise ValueError("every requested Hybrid partition requires eligible cells")
    normalized = tuple(normalizer.transform(sequence) for sequence, _trajectory in entries)
    early = stack_early_cycle_sequences(normalized)
    cutoff = int(early.cycle_mask.shape[1] - 1)
    prediction_cycles = torch.arange(cutoff + 1, 501, dtype=torch.int64)
    history = torch.full(early.cycle_mask.shape, float("nan"), dtype=torch.float32)
    history_mask = torch.zeros_like(early.cycle_mask)
    target = torch.full(
        (len(entries), 500 - cutoff), float("nan"), dtype=torch.float32
    )
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    initial = torch.empty(len(entries), dtype=torch.float32)
    for row, (_sequence, trajectory) in enumerate(entries):
        if set(trajectory) != set(range(1, 501)):
            raise ValueError("Hybrid supervision must contain every real cycle 1 through 500")
        initial_value = float(trajectory[cutoff])
        for cycle in range(1, cutoff + 1):
            soh = float(trajectory[cycle])
            if not bool(early.cycle_mask[row, cycle]):
                continue
            if _is_valid_soh(soh):
                history[row, cycle] = soh
                history_mask[row, cycle] = True
            elif audit_entries is not None:
                audit_entries.append(
                    MaskedCycleAuditEntry(
                        cell_id=_sequence.cell_id,
                        cycle_index=cycle,
                        reason="history_soh_out_of_range",
                    )
                )
        for index, cycle in enumerate(range(cutoff + 1, 501)):
            soh = float(trajectory[cycle])
            if _is_valid_soh(soh):
                target[row, index] = soh
                target_mask[row, index] = True
            elif audit_entries is not None:
                audit_entries.append(
                    MaskedCycleAuditEntry(
                        cell_id=_sequence.cell_id,
                        cycle_index=cycle,
                        reason="target_soh_out_of_range",
                    )
                )
        has_history = bool(history_mask[row].any().item())
        has_target = bool(target_mask[row].any().item())
        if not has_history and not has_target:
            raise ValueError(
                f"Hybrid partition cell {_sequence.cell_id} has no valid "
                "history/target supervision"
            )
        if not _is_valid_soh(initial_value):
            raise ValueError(
                f"Hybrid initial SOH is invalid for cell {_sequence.cell_id} "
                f"at cycle {cutoff}"
            )
        initial[row] = initial_value
        if not has_target:
            raise ValueError(
                f"Hybrid partition cell {_sequence.cell_id} has no valid target supervision"
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
            target_mask=target_mask,
        ),
    )


def _is_valid_soh(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value <= 1.5


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
    reference_capacity_ah: float,
    raw_sha256: str,
    config: MultichannelCycleConfig,
    data_version: str,
    audit_entries: list[MaskedCycleAuditEntry] | None = None,
    cache: EarlyCycleSequenceCache | None = None,
) -> EarlyCycleSequence:
    """Load only cutoff-bounded rows before invoking the label-free builder."""

    loaded = load_verified_matr_early_sequence(
        processed_root=processed_root,
        evidence=evidence,
        reference_capacity_ah=reference_capacity_ah,
        raw_sha256=raw_sha256,
        config=config,
        data_version=data_version,
        cache=cache,
    )
    if audit_entries is not None:
        audit_entries.extend(
            MaskedCycleAuditEntry(
                cell_id=evidence.cell_id,
                cycle_index=cycle,
                reason="early_capacity_ratio_above_1_5",
            )
            for cycle in loaded.masked_cycle_indices
        )
    return loaded.sequence


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
