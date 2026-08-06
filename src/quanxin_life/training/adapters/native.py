"""Safe native-model access to frozen tensor model views."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from quanxin_life.core import (
    LastBatchPolicy,
    PredictionTarget,
    TrainingReadableSplit,
    sha256_canonical,
)
from quanxin_life.data.model_views.builder import verify_model_view
from quanxin_life.data.model_views.schemas import ModelViewConfig, ModelViewManifest
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.early_cycle_sequence import EarlyCycleNormalizer
from quanxin_life.models.cyclepatch import EarlyCycleBatch
from quanxin_life.models.hybridpatch_v2 import HybridPatchV2Inputs, HybridPatchV2Targets
from quanxin_life.training.advanced_data import (
    AdvancedFinalMatrData,
    AdvancedSelectionMatrData,
    MaskedCycleAudit,
)
from quanxin_life.training.advanced_tasks import (
    AdvancedCycleLifeBatch,
    AdvancedTrajectoryBatch,
)
from quanxin_life.training.batching import BatchPlan
from quanxin_life.training.matr_data import MatrCurveCohorts, MatrHybridCohorts
from quanxin_life.training.tasks import CycleLifeCurveBatch, HybridTrajectoryBatch


@dataclass(frozen=True)
class NativeTensorBatch:
    cell_ids: tuple[str, ...]
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class NativeMatrViewRoots:
    scalar_root: Path
    trajectory_root: Path
    model_view_sha256: str


class NativeTensorPartition(Dataset[dict[str, torch.Tensor]]):
    """One cell-aligned partition loaded only from a verified safetensors View."""

    def __init__(
        self,
        *,
        split: TrainingReadableSplit,
        cell_ids: tuple[str, ...],
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        if not cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("native tensor partition cell IDs must be non-empty and unique")
        if not tensors:
            raise ValueError("native tensor partition requires tensors")
        if any(tensor.ndim == 0 or tensor.shape[0] != len(cell_ids) for tensor in tensors.values()):
            raise ValueError("native tensor partition tensors must align with cell IDs")
        self.split = split
        self.cell_ids = cell_ids
        self._tensors = dict(tensors)

    def __len__(self) -> int:
        return len(self.cell_ids)

    @overload
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[dict[str, torch.Tensor], ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        return {name: tensor[index] for name, tensor in self._tensors.items()}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return (self[index] for index in range(len(self)))

    def loader(self, plan: BatchPlan) -> DataLoader[NativeTensorBatch]:
        if plan.sample_count != len(self):
            raise ValueError("native tensor batch plan sample_count does not match partition")
        global_micro_batch = plan.micro_batch_size * plan.visible_gpu_count
        return DataLoader(
            range(len(self)),  # type: ignore[arg-type]  # collate changes sample type
            batch_size=global_micro_batch,
            shuffle=False,
            drop_last=plan.last_batch_policy is LastBatchPolicy.DROP,
            collate_fn=self._collate_indices,
        )

    def _collate_indices(self, indices: list[int]) -> NativeTensorBatch:
        return NativeTensorBatch(
            cell_ids=tuple(self.cell_ids[index] for index in indices),
            tensors={name: tensor[indices] for name, tensor in self._tensors.items()},
        )


class NativeTensorModelView:
    """Verified v2 View with safe, cell-indexed partition access."""

    def __init__(self, root: Path) -> None:
        directory = Path(root).resolve(strict=True)
        manifest = verify_model_view(directory)
        if manifest.schema_version != "model-view-manifest-v2":
            raise ValueError("native tensor training requires a v2 model view")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("view_id") != manifest.view_id:
            raise ValueError("native tensor metadata and manifest view IDs differ")
        partitions = metadata.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            raise ValueError("native tensor metadata requires partitions")
        tensors = load_file(directory / "tensors.safetensors", device="cpu")
        self.root = directory
        self.manifest: ModelViewManifest = manifest
        self.metadata: dict[str, object] = metadata
        self._normalization_path = directory / "normalization.json"
        self.output_sha256 = (directory / "COMMITTED").read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", self.output_sha256) is None:
            raise ValueError("native tensor model view COMMITTED marker is invalid")
        self._partitions: dict[TrainingReadableSplit, NativeTensorPartition] = {}
        observed_cells: set[str] = set()
        for payload in partitions:
            if not isinstance(payload, dict):
                raise ValueError("native tensor partition metadata must be an object")
            split = TrainingReadableSplit(str(payload.get("partition", "")))
            raw_cell_ids = payload.get("cell_ids")
            raw_shapes = payload.get("tensor_shapes")
            if not isinstance(raw_cell_ids, list) or not isinstance(raw_shapes, dict):
                raise ValueError("native tensor partition metadata is incomplete")
            cell_ids = tuple(str(value) for value in raw_cell_ids)
            if observed_cells & set(cell_ids):
                raise ValueError("native tensor model view leaks cells across partitions")
            if split in self._partitions:
                raise ValueError("native tensor model view repeats a partition")
            prefix = f"{split.value}."
            selected = {
                name.removeprefix(prefix): tensor
                for name, tensor in tensors.items()
                if name.startswith(prefix)
            }
            expected_names = {
                str(name).removeprefix(prefix)
                for name in raw_shapes
                if str(name).startswith(prefix)
            }
            if set(selected) != expected_names:
                raise ValueError("native tensor metadata and safetensors keys differ")
            self._partitions[split] = NativeTensorPartition(
                split=split,
                cell_ids=cell_ids,
                tensors=selected,
            )
            observed_cells.update(cell_ids)

    @property
    def normalization(self) -> EarlyCycleNormalizer:
        return _load_normalization(self._normalization_path)

    def partition(
        self,
        split: TrainingReadableSplit | str,
    ) -> NativeTensorPartition:
        normalized = TrainingReadableSplit(split)
        try:
            return self._partitions[normalized]
        except KeyError as exc:
            raise KeyError(f"model view partition is unavailable: {normalized.value}") from exc


def resolve_native_matr_view_roots(
    project_root: Path,
    *,
    cutoff_cycle: int,
) -> NativeMatrViewRoots:
    root = Path(project_root).resolve(strict=True)
    config_root = root / "configs" / "model_views"
    paths = (
        *sorted(config_root.glob("*.json")),
        *sorted((config_root / "matr_cutoffs").glob("*.json")),
    )
    configs = tuple(
        ModelViewConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    )
    scalar = _require_cutoff_view_config(
        configs,
        cutoff_cycle=cutoff_cycle,
        view_prefix="early_life_sequence",
        target_semantics="matr_official_cycle_life",
    )
    trajectory = _require_cutoff_view_config(
        configs,
        cutoff_cycle=cutoff_cycle,
        view_prefix="soh_trajectory",
        target_semantics="matr_observed_soh_to_cycle_500",
    )
    scalar_root = root / "data" / "model_views" / scalar.view_id / scalar.view_version
    trajectory_root = (
        root / "data" / "model_views" / trajectory.view_id / trajectory.view_version
    )
    scalar_manifest = verify_model_view(scalar_root)
    trajectory_manifest = verify_model_view(trajectory_root)
    for config, manifest in (
        (scalar, scalar_manifest),
        (trajectory, trajectory_manifest),
    ):
        if (
            manifest.view_id != config.view_id
            or manifest.view_version != config.view_version
            or manifest.cutoff_cycle != cutoff_cycle
            or manifest.target_semantics != config.target_semantics
        ):
            raise ValueError("native MATR View manifest differs from its cutoff config")
    return NativeMatrViewRoots(
        scalar_root=scalar_root,
        trajectory_root=trajectory_root,
        model_view_sha256=sha256_canonical(
            {
                scalar.view_id: _committed_sha256(scalar_root),
                trajectory.view_id: _committed_sha256(trajectory_root),
            }
        ),
    )


def _require_cutoff_view_config(
    configs: tuple[ModelViewConfig, ...],
    *,
    cutoff_cycle: int,
    view_prefix: str,
    target_semantics: str,
) -> ModelViewConfig:
    matches = tuple(
        config
        for config in configs
        if config.cutoff_cycle == cutoff_cycle
        and config.view_id in {view_prefix, f"{view_prefix}_c{cutoff_cycle}"}
        and config.target_semantics == target_semantics
    )
    if len(matches) != 1:
        raise ValueError(
            f"exactly one frozen {view_prefix} View is required for cutoff {cutoff_cycle}"
        )
    return matches[0]


def _committed_sha256(root: Path) -> str:
    value = (root / "COMMITTED").read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("native MATR View COMMITTED marker is invalid")
    return value


def load_native_advanced_matr_selection_data(
    *,
    scalar_view_root: Path,
    trajectory_view_root: Path,
    combined_split: SplitManifest,
    cutoff_cycle: int,
) -> AdvancedSelectionMatrData:
    """Materialize train/validation batches exclusively from frozen tensor Views."""

    result = _load_native_advanced_matr_data(
        scalar_view_root=scalar_view_root,
        trajectory_view_root=trajectory_view_root,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
        partitions=(TrainingReadableSplit.TRAIN, TrainingReadableSplit.VALIDATION),
    )
    if isinstance(result, AdvancedFinalMatrData):
        raise RuntimeError("selection loading exposed held-out partitions")
    return result


def load_native_advanced_matr_final_data(
    *,
    scalar_view_root: Path,
    trajectory_view_root: Path,
    combined_split: SplitManifest,
    cutoff_cycle: int,
) -> AdvancedFinalMatrData:
    """Materialize all frozen partitions for one-time final evaluation."""

    result = _load_native_advanced_matr_data(
        scalar_view_root=scalar_view_root,
        trajectory_view_root=trajectory_view_root,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
        partitions=tuple(TrainingReadableSplit),
    )
    if not isinstance(result, AdvancedFinalMatrData):
        raise RuntimeError("final loading did not expose held-out partitions")
    return result


def load_native_matr_legacy_cohorts(
    *,
    scalar_view_root: Path,
    trajectory_view_root: Path,
    combined_split: SplitManifest,
    cutoff_cycle: int,
) -> tuple[MatrCurveCohorts, MatrHybridCohorts]:
    """Reconstruct legacy baseline tensors from the same frozen native Views."""

    data = load_native_advanced_matr_final_data(
        scalar_view_root=scalar_view_root,
        trajectory_view_root=trajectory_view_root,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
    )
    curves = {
        split: _legacy_curve_batch(
            getattr(data, f"scalar_{split}"),
            normalizer=data.scalar_normalizer,
        )
        for split in ("train", "validation", "calibration", "test")
    }
    trajectory_batches = tuple(
        getattr(data, f"hybrid_{split}")
        for split in ("train", "validation", "calibration", "test")
    )
    prediction_cycles = _legacy_common_prediction_cycles(trajectory_batches)
    hybrid = {
        split: _legacy_hybrid_batch(
            getattr(data, f"hybrid_{split}"),
            prediction_cycles=prediction_cycles,
        )
        for split in ("train", "validation", "calibration", "test")
    }
    return (
        MatrCurveCohorts(
            train=curves["train"],
            validation=curves["validation"],
            calibration=curves["calibration"],
            test=curves["test"],
        ),
        MatrHybridCohorts(
            train=hybrid["train"],
            validation=hybrid["validation"],
            calibration=hybrid["calibration"],
            test=hybrid["test"],
        ),
    )


def _legacy_curve_batch(
    batch: AdvancedCycleLifeBatch,
    *,
    normalizer: EarlyCycleNormalizer,
) -> CycleLifeCurveBatch:
    early = batch.early_batch
    discharge_capacity = early.values[:, :, 1, :, 2].clone()
    discharge_mask = early.sample_mask[:, :, 1, :]
    complete_cycles = discharge_mask.all(dim=2)
    if not torch.equal(complete_cycles, early.cycle_mask):
        raise ValueError("legacy curve reconstruction requires complete observed curves")
    mean = early.values.new_tensor(normalizer.variable_means[2])
    raw_std = early.values.new_tensor(normalizer.variable_stds[2])
    scale = torch.where(raw_std > 0, raw_std, torch.ones_like(raw_std))
    raw_capacity = discharge_capacity * scale + mean
    return CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=batch.cell_ids,
        curve_values=raw_capacity,
        observed_mask=early.cycle_mask,
        observed_cycles=batch.raw_labels,
        cutoff_cycle=batch.cutoff_cycle,
    )


def _legacy_common_prediction_cycles(
    batches: tuple[AdvancedTrajectoryBatch, ...],
) -> tuple[int, ...]:
    source = batches[0].inputs.prediction_cycles
    common = torch.ones_like(source, dtype=torch.bool)
    for batch in batches:
        if not torch.equal(source, batch.inputs.prediction_cycles):
            raise ValueError("legacy Hybrid prediction axes must match")
        common &= batch.targets.target_mask.all(dim=0)
    cycles = tuple(int(value) for value in source[common].tolist())
    if len(cycles) < 3 or cycles[-1] != 500:
        raise ValueError("legacy Hybrid requires common real supervision through cycle 500")
    return cycles


def _legacy_hybrid_batch(
    batch: AdvancedTrajectoryBatch,
    *,
    prediction_cycles: tuple[int, ...],
) -> HybridTrajectoryBatch:
    source_indices = {
        int(cycle): index
        for index, cycle in enumerate(batch.inputs.prediction_cycles.tolist())
    }
    indices = torch.tensor(
        [source_indices[cycle] for cycle in prediction_cycles],
        dtype=torch.int64,
        device=batch.targets.target_soh.device,
    )
    if not bool(batch.targets.target_mask.index_select(1, indices).all().item()):
        raise ValueError("legacy Hybrid cohort contains missing target supervision")
    early = batch.inputs.early_batch
    expanded_mask = early.sample_mask.unsqueeze(-1)
    safe = torch.where(expanded_mask, early.values, torch.zeros_like(early.values))
    counts = expanded_mask.to(early.values.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
    return HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=batch.cell_ids,
        features=safe.sum(dim=(1, 2, 3)) / counts,
        initial_soh=batch.inputs.initial_soh,
        target_soh=batch.targets.target_soh.index_select(1, indices),
        prediction_cycles=prediction_cycles,
        cutoff_cycle=int(early.cycle_mask.shape[1] - 1),
    )


def _load_native_advanced_matr_data(
    *,
    scalar_view_root: Path,
    trajectory_view_root: Path,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    partitions: tuple[TrainingReadableSplit, ...],
) -> AdvancedSelectionMatrData | AdvancedFinalMatrData:
    scalar_view = NativeTensorModelView(scalar_view_root)
    trajectory_view = NativeTensorModelView(trajectory_view_root)
    _validate_matr_view_pair(scalar_view, trajectory_view, cutoff_cycle=cutoff_cycle)
    scalar_batches = {
        split: _cycle_life_batch(scalar_view, split, combined_split)
        for split in partitions
    }
    trajectory_batches = {
        split: _trajectory_batch(trajectory_view, split, combined_split)
        for split in partitions
    }
    if len(partitions) == 2:
        return AdvancedSelectionMatrData(
            source_split=combined_split,
            scalar_train=scalar_batches[TrainingReadableSplit.TRAIN],
            scalar_validation=scalar_batches[TrainingReadableSplit.VALIDATION],
            scalar_normalizer=scalar_view.normalization,
            hybrid_train=trajectory_batches[TrainingReadableSplit.TRAIN],
            hybrid_validation=trajectory_batches[TrainingReadableSplit.VALIDATION],
            hybrid_normalizer=trajectory_view.normalization,
            masked_cycle_audit=MaskedCycleAudit(),
        )
    return AdvancedFinalMatrData(
        source_split=combined_split,
        scalar_train=scalar_batches[TrainingReadableSplit.TRAIN],
        scalar_validation=scalar_batches[TrainingReadableSplit.VALIDATION],
        scalar_normalizer=scalar_view.normalization,
        hybrid_train=trajectory_batches[TrainingReadableSplit.TRAIN],
        hybrid_validation=trajectory_batches[TrainingReadableSplit.VALIDATION],
        hybrid_normalizer=trajectory_view.normalization,
        masked_cycle_audit=MaskedCycleAudit(),
        scalar_calibration=scalar_batches[TrainingReadableSplit.CALIBRATION],
        scalar_test=scalar_batches[TrainingReadableSplit.TEST],
        hybrid_calibration=trajectory_batches[TrainingReadableSplit.CALIBRATION],
        hybrid_test=trajectory_batches[TrainingReadableSplit.TEST],
    )


def _validate_matr_view_pair(
    scalar_view: NativeTensorModelView,
    trajectory_view: NativeTensorModelView,
    *,
    cutoff_cycle: int,
) -> None:
    if scalar_view.manifest.view_id not in {
        "early_life_sequence",
        f"early_life_sequence_c{cutoff_cycle}",
    }:
        raise ValueError("scalar native View must be early_life_sequence")
    if trajectory_view.manifest.view_id not in {
        "soh_trajectory",
        f"soh_trajectory_c{cutoff_cycle}",
    }:
        raise ValueError("trajectory native View must be soh_trajectory")
    if scalar_view.manifest.target_semantics != "matr_official_cycle_life":
        raise ValueError("scalar native View target semantics are incompatible")
    if trajectory_view.manifest.target_semantics != "matr_observed_soh_to_cycle_500":
        raise ValueError("trajectory native View target semantics are incompatible")
    contexts = {
        (view.manifest.cutoff_cycle, view.normalization.cutoff_cycle)
        for view in (scalar_view, trajectory_view)
    }
    if contexts != {(cutoff_cycle, cutoff_cycle)}:
        raise ValueError("requested cutoff does not match frozen model View context")


def _cycle_life_batch(
    view: NativeTensorModelView,
    split: TrainingReadableSplit,
    combined_split: SplitManifest,
) -> AdvancedCycleLifeBatch:
    partition = view.partition(split)
    _validate_partition_cells(partition, combined_split)
    return AdvancedCycleLifeBatch(
        early_batch=_early_batch(view, partition),
        raw_labels=_require_tensor(partition, "target_cycle_life"),
    )


def _trajectory_batch(
    view: NativeTensorModelView,
    split: TrainingReadableSplit,
    combined_split: SplitManifest,
) -> AdvancedTrajectoryBatch:
    partition = view.partition(split)
    _validate_partition_cells(partition, combined_split)
    prediction_rows = _require_tensor(partition, "prediction_cycles")
    if prediction_rows.ndim != 2 or not torch.equal(
        prediction_rows,
        prediction_rows[0].expand_as(prediction_rows),
    ):
        raise ValueError("trajectory prediction cycles must be identical for every cell")
    expected_cycles = torch.arange(
        view.normalization.cutoff_cycle + 1,
        501,
        dtype=torch.int64,
        device=prediction_rows.device,
    )
    if not torch.equal(prediction_rows[0], expected_cycles):
        raise ValueError("trajectory supervision must terminate exactly at cycle 500")
    early = _early_batch(view, partition)
    return AdvancedTrajectoryBatch(
        inputs=HybridPatchV2Inputs(
            early_batch=early,
            initial_soh=_require_tensor(partition, "initial_soh"),
            prediction_cycles=prediction_rows[0],
        ),
        targets=HybridPatchV2Targets(
            history_soh=_require_tensor(partition, "history_soh"),
            history_mask=_require_tensor(partition, "history_mask"),
            target_soh=_require_tensor(partition, "target_soh"),
            target_mask=_require_tensor(partition, "target_mask"),
        ),
    )


def _early_batch(
    view: NativeTensorModelView,
    partition: NativeTensorPartition,
) -> EarlyCycleBatch:
    normalizer = view.normalization
    return EarlyCycleBatch(
        dataset_id=normalizer.dataset_id,
        data_version=normalizer.data_version,
        feature_version=normalizer.feature_version,
        normalization_statistics_sha256=normalizer.statistics_sha256,
        cell_ids=partition.cell_ids,
        condition_names=normalizer.condition_names,
        values=_require_tensor(partition, "values"),
        cycle_indices=_require_tensor(partition, "cycle_indices"),
        cycle_mask=_require_tensor(partition, "cycle_mask"),
        sample_mask=_require_tensor(partition, "sample_mask"),
        condition_values=_require_tensor(partition, "condition_values"),
        condition_mask=_require_tensor(partition, "condition_mask"),
    )


def _validate_partition_cells(
    partition: NativeTensorPartition,
    combined_split: SplitManifest,
) -> None:
    expected = set(getattr(combined_split, partition.split.value))
    if not set(partition.cell_ids) <= expected:
        raise ValueError("native View partition differs from the frozen cell split")


def _require_tensor(
    partition: NativeTensorPartition,
    name: str,
) -> torch.Tensor:
    try:
        return partition._tensors[name]
    except KeyError as exc:
        raise ValueError(f"native View tensor is missing: {name}") from exc


def _load_normalization(path: Path) -> EarlyCycleNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.pop("fitted_split", None) != "train":
        raise ValueError("native View normalization must be fitted on train")
    tuple_fields = {
        "cycle_indices",
        "condition_names",
        "variable_means",
        "variable_stds",
        "condition_means",
        "condition_stds",
    }
    for name in tuple_fields:
        value = payload.get(name)
        if not isinstance(value, list):
            raise ValueError(f"native View normalization field is invalid: {name}")
        payload[name] = tuple(value)
    try:
        return EarlyCycleNormalizer(**payload)
    except TypeError as exc:
        raise ValueError("native View normalization fields are invalid") from exc


__all__ = [
    "NativeMatrViewRoots",
    "NativeTensorBatch",
    "NativeTensorModelView",
    "NativeTensorPartition",
    "load_native_advanced_matr_final_data",
    "load_native_advanced_matr_selection_data",
    "load_native_matr_legacy_cohorts",
    "resolve_native_matr_view_roots",
]
