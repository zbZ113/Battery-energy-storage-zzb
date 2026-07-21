"""Governed TrainingTask adapters for the advanced MATR model families.

The adapters keep future supervision outside feature construction, bind scalar
normalisation and BatLiNet references to training cells, and expose the shared
``TrainingTask`` epoch lifecycle without consulting calibration or test data.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifePairBatch,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
    CyclePatchBatLiNet,
)
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchLifeRegressor,
    EarlyCycleBatch,
)
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Inputs,
    HybridPatchV2Predictor,
    HybridPatchV2Targets,
    compute_hybridpatch_v2_loss,
)
from quanxin_life.training.engine import EpochMetrics, Scheduler


@dataclass(frozen=True, eq=False)
class AdvancedCycleLifeBatch:
    """Label-bearing MATR batch whose feature tensor remains label-free."""

    early_batch: EarlyCycleBatch
    raw_labels: Tensor

    def __post_init__(self) -> None:
        if self.early_batch.dataset_id != "MATR":
            raise ValueError("advanced cycle-life labels are restricted to MATR")
        if not isinstance(self.raw_labels, Tensor) or not self.raw_labels.is_floating_point():
            raise ValueError("raw_labels must be a floating tensor")
        labels = self.raw_labels.detach().clone()
        object.__setattr__(self, "raw_labels", labels)
        if labels.shape != (len(self.early_batch.cell_ids),):
            raise ValueError("raw_labels must align with early_batch cells")
        if labels.device != self.early_batch.values.device:
            raise ValueError("raw_labels device must match early_batch")
        if labels.dtype != self.early_batch.values.dtype:
            raise ValueError("raw_labels dtype must match early_batch values")
        if not bool(torch.isfinite(labels).all().item()):
            raise ValueError("raw_labels must be finite")
        if bool((labels <= self.cutoff_cycle).any().item()):
            raise ValueError("raw_labels must be greater than cutoff cycle")

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.early_batch.cell_ids

    @property
    def cutoff_cycle(self) -> int:
        return int(self.early_batch.cycle_mask.shape[1] - 1)


@dataclass(frozen=True, eq=False)
class AdvancedTrajectoryBatch:
    """Aligned inference inputs and loss-only real SOH supervision."""

    inputs: HybridPatchV2Inputs
    targets: HybridPatchV2Targets

    def __post_init__(self) -> None:
        early = self.inputs.early_batch
        batch_size, cycle_count = early.cycle_mask.shape
        horizon = int(self.inputs.prediction_cycles.numel())
        if self.targets.history_soh.shape != (batch_size, cycle_count):
            raise ValueError("history targets must align with early cycle observations")
        if self.targets.target_soh.shape != (batch_size, horizon):
            raise ValueError("trajectory targets must align with prediction cycles")
        tensors = (
            self.inputs.initial_soh,
            self.targets.history_soh,
            self.targets.target_soh,
        )
        if any(tensor.dtype != early.values.dtype for tensor in tensors):
            raise ValueError("trajectory tensors must share the early batch dtype")
        if any(tensor.device != early.values.device for tensor in tensors):
            raise ValueError("trajectory tensors must share the early batch device")

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.inputs.early_batch.cell_ids


@dataclass(frozen=True)
class _CycleLifeDeviceBatches:
    train_early: EarlyCycleBatch
    validation_early: EarlyCycleBatch
    train_standardized: Tensor
    validation_raw: Tensor
    reference_indices: Tensor | None = None
    reference_labels: Tensor | None = None


@dataclass(frozen=True)
class _TrajectoryDeviceBatches:
    train: AdvancedTrajectoryBatch
    validation: AdvancedTrajectoryBatch


class CyclePatchDirectTrainingTask:
    """Direct CyclePatch regression in train-standardised cycle-life units."""

    def __init__(
        self,
        *,
        train_batch: AdvancedCycleLifeBatch,
        validation_batch: AdvancedCycleLifeBatch,
        split_manifest: SplitManifest,
        config: CyclePatchConfig,
        learning_rate: float,
        seed: int,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_optimizer_inputs(learning_rate, weight_decay)
        supervised_split = _derive_supervised_split(
            train_batch, validation_batch, split_manifest
        )
        _seed_everything(seed)
        labels = _label_mapping(train_batch)
        self.target_scaler = CycleLifeTargetScaler.fit(
            labels,
            training_cell_ids=train_batch.cell_ids,
            split_manifest=supervised_split,
            cutoff_cycle=train_batch.cutoff_cycle,
            dataset_id=train_batch.early_batch.dataset_id,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )
        self.source_split_manifest = split_manifest
        self.source_split_sha256 = _split_manifest_sha256(split_manifest)
        self.supervised_split_manifest = supervised_split
        self.supervised_split_sha256 = _split_manifest_sha256(supervised_split)
        self._train_batch = train_batch
        self._validation_batch = validation_batch
        self._train_standardized = self.target_scaler.transform(train_batch.raw_labels)
        self._model = CyclePatchLifeRegressor(
            config,
            condition_count=len(train_batch.early_batch.condition_names),
        ).to(dtype=train_batch.early_batch.values.dtype)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._scheduler: Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, factor=0.5, patience=3, min_lr=1e-6
        )
        self._device_cache: dict[str, _CycleLifeDeviceBatches] = {}

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def _batches_for_device(self, device: torch.device) -> _CycleLifeDeviceBatches:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = _CycleLifeDeviceBatches(
                train_early=_move_early_batch(self._train_batch.early_batch, device),
                validation_early=_move_early_batch(
                    self._validation_batch.early_batch, device
                ),
                train_standardized=self._train_standardized.to(device=device),
                validation_raw=self._validation_batch.raw_labels.to(device=device),
            )
            self._device_cache[key] = cached
        return cached

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        batches = self._batches_for_device(device)
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        prediction = self._model(batches.train_early)
        loss = functional.smooth_l1_loss(
            prediction, batches.train_standardized, beta=1.0
        )
        _require_finite_loss(loss)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        return EpochMetrics(loss=_as_float(loss))

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        return self.evaluate(self._validation_batch, device=device)

    def evaluate(
        self, batch: AdvancedCycleLifeBatch, *, device: torch.device
    ) -> EpochMetrics:
        _validate_evaluation_early_batch(self._train_batch.early_batch, batch.early_batch)
        self._model.eval()
        with torch.no_grad():
            standardized = self._model(_move_early_batch(batch.early_batch, device))
            raw_prediction = self.target_scaler.inverse_transform(standardized)
            metrics = _raw_cycle_metrics(raw_prediction, batch.raw_labels.to(device=device))
        return EpochMetrics(loss=metrics["mae"], metrics=metrics)


class CyclePatchBatLiNetTrainingTask:
    """Inter-cell CyclePatch training with a frozen train-only reference set."""

    def __init__(
        self,
        *,
        train_batch: AdvancedCycleLifeBatch,
        validation_batch: AdvancedCycleLifeBatch,
        split_manifest: SplitManifest,
        target_scaler: CycleLifeTargetScaler,
        reference_library: CycleLifeReferenceLibrary,
        config: BatLiNetConfig,
        learning_rate: float,
        seed: int,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_optimizer_inputs(learning_rate, weight_decay)
        supervised_split = _derive_supervised_split(
            train_batch, validation_batch, split_manifest
        )
        reference_cells = set(reference_library.cell_ids)
        if not reference_cells <= set(train_batch.cell_ids):
            raise ValueError("reference cells must be a subset of training cells")
        if reference_cells & set(validation_batch.cell_ids):
            raise ValueError("reference and validation cells must be disjoint")
        if reference_library.reference_count != config.reference_count:
            raise ValueError("reference library count must match BatLiNet config")
        if reference_library.seed != seed:
            raise ValueError("reference library seed must match training task seed")
        if target_scaler.dataset_id != train_batch.early_batch.dataset_id:
            raise ValueError("target scaler dataset must match training data")
        if target_scaler.cutoff_cycle != train_batch.cutoff_cycle:
            raise ValueError("target scaler cutoff must match training data")
        if reference_library.scaler_context_sha256 != target_scaler.context_sha256:
            raise ValueError("reference library and target scaler context do not match")
        if (
            reference_library.training_cell_ids_sha256
            != target_scaler.training_cell_ids_sha256
            or reference_library.training_labels_sha256
            != target_scaler.training_labels_sha256
        ):
            raise ValueError("reference library and target scaler hashes do not match")
        expected_scaler = CycleLifeTargetScaler.fit(
            _label_mapping(train_batch),
            training_cell_ids=train_batch.cell_ids,
            split_manifest=supervised_split,
            cutoff_cycle=train_batch.cutoff_cycle,
            dataset_id=train_batch.early_batch.dataset_id,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )
        if expected_scaler != target_scaler:
            raise ValueError("target scaler is not bound to the supplied training labels")
        index_by_cell = {
            cell_id: index for index, cell_id in enumerate(train_batch.cell_ids)
        }
        reference_indices = torch.tensor(
            [index_by_cell[cell_id] for cell_id in reference_library.cell_ids],
            dtype=torch.int64,
        )
        reference_labels = torch.tensor(
            reference_library.standardized_labels,
            dtype=train_batch.raw_labels.dtype,
        )
        expected_reference_labels = target_scaler.transform(train_batch.raw_labels)[
            reference_indices
        ]
        if not torch.allclose(reference_labels, expected_reference_labels):
            raise ValueError("reference labels do not match training labels and scaler")

        _seed_everything(seed)
        self.source_split_manifest = split_manifest
        self.source_split_sha256 = _split_manifest_sha256(split_manifest)
        self.supervised_split_manifest = supervised_split
        self.supervised_split_sha256 = _split_manifest_sha256(supervised_split)
        self.target_scaler = target_scaler
        self.reference_library = reference_library
        self._train_batch = train_batch
        self._validation_batch = validation_batch
        self._train_standardized = target_scaler.transform(train_batch.raw_labels)
        self._reference_indices = reference_indices
        self._reference_labels = reference_labels
        self._model = CyclePatchBatLiNet(
            config,
            condition_count=len(train_batch.early_batch.condition_names),
        ).to(dtype=train_batch.early_batch.values.dtype)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._scheduler: Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, factor=0.5, patience=3, min_lr=1e-6
        )
        self._device_cache: dict[str, _CycleLifeDeviceBatches] = {}

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def _batches_for_device(self, device: torch.device) -> _CycleLifeDeviceBatches:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = _CycleLifeDeviceBatches(
                train_early=_move_early_batch(self._train_batch.early_batch, device),
                validation_early=_move_early_batch(
                    self._validation_batch.early_batch, device
                ),
                train_standardized=self._train_standardized.to(device=device),
                validation_raw=self._validation_batch.raw_labels.to(device=device),
                reference_indices=self._reference_indices.to(device=device),
                reference_labels=self._reference_labels.to(device=device),
            )
            self._device_cache[key] = cached
        return cached

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        batches = self._batches_for_device(device)
        assert batches.reference_indices is not None
        assert batches.reference_labels is not None
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        train_embeddings = self._model.encode(batches.train_early)
        reference_embeddings = train_embeddings.index_select(
            0, batches.reference_indices
        )
        loss = self._model.loss(
            CycleLifePairBatch(
                target_embeddings=train_embeddings,
                target_labels=batches.train_standardized,
                reference_embeddings=reference_embeddings,
                reference_labels=batches.reference_labels,
            )
        )
        _require_finite_loss(loss)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        return EpochMetrics(loss=_as_float(loss))

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        return self.evaluate(self._validation_batch, device=device)

    def evaluate(
        self, batch: AdvancedCycleLifeBatch, *, device: torch.device
    ) -> EpochMetrics:
        _validate_evaluation_early_batch(self._train_batch.early_batch, batch.early_batch)
        batches = self._batches_for_device(device)
        assert batches.reference_indices is not None
        assert batches.reference_labels is not None
        self._model.eval()
        with torch.no_grad():
            train_embeddings = self._model.encode(batches.train_early)
            reference_embeddings = train_embeddings.index_select(
                0, batches.reference_indices
            )
            validation_embeddings = self._model.encode(
                _move_early_batch(batch.early_batch, device)
            )
            standardized = self._model.fuse_standardized(
                validation_embeddings,
                reference_embeddings,
                batches.reference_labels,
            )
            raw_prediction = self.target_scaler.inverse_transform(standardized)
            metrics = _raw_cycle_metrics(raw_prediction, batch.raw_labels.to(device=device))
        return EpochMetrics(loss=metrics["mae"], metrics=metrics)


class HybridPatchV2TrainingTask:
    """Real-through-cycle-500 trajectory training without an independent RUL head."""

    def __init__(
        self,
        *,
        train_batch: AdvancedTrajectoryBatch,
        validation_batch: AdvancedTrajectoryBatch,
        split_manifest: SplitManifest,
        config: HybridPatchV2Config,
        learning_rate: float,
        seed: int,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_optimizer_inputs(learning_rate, weight_decay)
        supervised_split = _derive_trajectory_supervised_split(
            train_batch, validation_batch, split_manifest
        )
        _seed_everything(seed)
        self.source_split_manifest = split_manifest
        self.source_split_sha256 = _split_manifest_sha256(split_manifest)
        self.supervised_split_manifest = supervised_split
        self.supervised_split_sha256 = _split_manifest_sha256(supervised_split)
        self._train_batch = train_batch
        self._validation_batch = validation_batch
        self._config = config
        self._model = HybridPatchV2Predictor(
            config,
            condition_count=len(train_batch.inputs.early_batch.condition_names),
        ).to(dtype=train_batch.inputs.early_batch.values.dtype)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._scheduler: Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, factor=0.5, patience=3, min_lr=1e-6
        )
        self._device_cache: dict[str, _TrajectoryDeviceBatches] = {}

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def _batches_for_device(self, device: torch.device) -> _TrajectoryDeviceBatches:
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = _TrajectoryDeviceBatches(
                train=_move_trajectory_batch(self._train_batch, device),
                validation=_move_trajectory_batch(self._validation_batch, device),
            )
            self._device_cache[key] = cached
        return cached

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        batch = self._batches_for_device(device).train
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        output = self._model(batch.inputs)
        loss = compute_hybridpatch_v2_loss(
            output, batch.targets, batch.inputs, self._config
        )
        _require_finite_loss(loss.total)
        loss.total.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        return EpochMetrics(
            loss=_as_float(loss.total),
            metrics={
                "trajectory": _as_float(loss.trajectory),
                "history": _as_float(loss.history),
                "smooth": _as_float(loss.smooth),
                "order": _as_float(loss.order),
                "residual": _as_float(loss.residual),
            },
        )

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        return self.evaluate(self._validation_batch, device=device)

    def evaluate(
        self, batch: AdvancedTrajectoryBatch, *, device: torch.device
    ) -> EpochMetrics:
        _validate_evaluation_early_batch(
            self._train_batch.inputs.early_batch, batch.inputs.early_batch
        )
        batch = _move_trajectory_batch(batch, device)
        self._model.eval()
        with torch.no_grad():
            output = self._model(batch.inputs)
            loss = compute_hybridpatch_v2_loss(
                output, batch.targets, batch.inputs, self._config
            )
            metrics = _trajectory_metrics(
                output.predicted_soh,
                batch.targets.target_soh,
                batch.targets.target_mask,
            )
        return EpochMetrics(loss=_as_float(loss.total), metrics=metrics)


def _validate_optimizer_inputs(learning_rate: float, weight_decay: float) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")


def _derive_supervised_split(
    train: AdvancedCycleLifeBatch,
    validation: AdvancedCycleLifeBatch,
    source_split: SplitManifest,
) -> SplitManifest:
    if _early_schema(train.early_batch) != _early_schema(validation.early_batch):
        raise ValueError("training and validation early batch schemas must match")
    return _derive_partition_supervised_split(
        train_cells=train.cell_ids,
        validation_cells=validation.cell_ids,
        source_split=source_split,
        dataset_id=train.early_batch.dataset_id,
    )


def _derive_trajectory_supervised_split(
    train: AdvancedTrajectoryBatch,
    validation: AdvancedTrajectoryBatch,
    source_split: SplitManifest,
) -> SplitManifest:
    _validate_trajectory_batches(train, validation)
    return _derive_partition_supervised_split(
        train_cells=train.cell_ids,
        validation_cells=validation.cell_ids,
        source_split=source_split,
        dataset_id=train.inputs.early_batch.dataset_id,
    )


def _derive_partition_supervised_split(
    *,
    train_cells: tuple[str, ...],
    validation_cells: tuple[str, ...],
    source_split: SplitManifest,
    dataset_id: str,
) -> SplitManifest:
    if set(train_cells) & set(validation_cells):
        raise ValueError("training and validation cells must be disjoint")
    if source_split.dataset_id != dataset_id:
        raise ValueError("SplitManifest dataset must match training data")
    _require_ordered_partition_subset(
        train_cells,
        source_split.train,
        partition_name="train",
    )
    _require_ordered_partition_subset(
        validation_cells,
        source_split.validation,
        partition_name="validation",
    )
    return SplitManifest(
        dataset_id=source_split.dataset_id,
        seed=source_split.seed,
        train=train_cells,
        validation=validation_cells,
        calibration=(),
        test=(),
    )


def _require_ordered_partition_subset(
    supervised_cells: tuple[str, ...],
    source_cells: tuple[str, ...],
    *,
    partition_name: str,
) -> None:
    if not supervised_cells:
        raise ValueError(
            f"supervised cells must be a non-empty ordered subset of source {partition_name}"
        )
    source_index = {cell_id: index for index, cell_id in enumerate(source_cells)}
    if any(cell_id not in source_index for cell_id in supervised_cells):
        raise ValueError(
            f"supervised cells must be an ordered subset of source {partition_name}"
        )
    positions = tuple(source_index[cell_id] for cell_id in supervised_cells)
    if any(current <= previous for previous, current in pairwise(positions)):
        raise ValueError(
            f"supervised cells must preserve source {partition_name} relative order"
        )


def _split_manifest_sha256(split_manifest: SplitManifest) -> str:
    return sha256_canonical(split_manifest.model_dump(mode="json"))


def _validate_trajectory_batches(
    train: AdvancedTrajectoryBatch,
    validation: AdvancedTrajectoryBatch,
) -> None:
    if set(train.cell_ids) & set(validation.cell_ids):
        raise ValueError("training and validation cells must be disjoint")
    if _early_schema(train.inputs.early_batch) != _early_schema(
        validation.inputs.early_batch
    ):
        raise ValueError("training and validation early batch schemas must match")
    train_cycles = train.inputs.prediction_cycles.detach().cpu()
    validation_cycles = validation.inputs.prediction_cycles.detach().cpu()
    if not torch.equal(train_cycles, validation_cycles):
        raise ValueError("training and validation prediction cycles must match")


def _validate_evaluation_early_batch(
    training: EarlyCycleBatch, evaluation: EarlyCycleBatch
) -> None:
    if set(training.cell_ids) & set(evaluation.cell_ids):
        raise ValueError("evaluation cells must be disjoint from training cells")
    if _early_schema(training) != _early_schema(evaluation):
        raise ValueError("evaluation early batch schema must match training data")


def _early_schema(batch: EarlyCycleBatch) -> tuple[object, ...]:
    return (
        batch.dataset_id,
        batch.data_version,
        batch.feature_version,
        batch.normalization_statistics_sha256,
        batch.condition_names,
        batch.values.shape[1:],
        batch.values.dtype,
    )


def _label_mapping(batch: AdvancedCycleLifeBatch) -> dict[str, float]:
    labels = batch.raw_labels.detach().cpu().tolist()
    return {
        cell_id: float(label)
        for cell_id, label in zip(batch.cell_ids, labels, strict=True)
    }


def _move_early_batch(batch: EarlyCycleBatch, device: torch.device) -> EarlyCycleBatch:
    return EarlyCycleBatch(
        dataset_id=batch.dataset_id,
        data_version=batch.data_version,
        feature_version=batch.feature_version,
        normalization_statistics_sha256=batch.normalization_statistics_sha256,
        cell_ids=batch.cell_ids,
        condition_names=batch.condition_names,
        values=batch.values.to(device=device),
        cycle_indices=batch.cycle_indices.to(device=device),
        cycle_mask=batch.cycle_mask.to(device=device),
        sample_mask=batch.sample_mask.to(device=device),
        condition_values=batch.condition_values.to(device=device),
        condition_mask=batch.condition_mask.to(device=device),
    )


def _move_trajectory_batch(
    batch: AdvancedTrajectoryBatch, device: torch.device
) -> AdvancedTrajectoryBatch:
    inputs = HybridPatchV2Inputs(
        early_batch=_move_early_batch(batch.inputs.early_batch, device),
        initial_soh=batch.inputs.initial_soh.to(device=device),
        prediction_cycles=batch.inputs.prediction_cycles.to(device=device),
    )
    targets = HybridPatchV2Targets(
        history_soh=batch.targets.history_soh.to(device=device),
        history_mask=batch.targets.history_mask.to(device=device),
        target_soh=batch.targets.target_soh.to(device=device),
        target_mask=batch.targets.target_mask.to(device=device),
    )
    return AdvancedTrajectoryBatch(inputs=inputs, targets=targets)


def _raw_cycle_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("cycle-life prediction and target must align as vectors")
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("cycle-life validation produced non-finite predictions")
    error = prediction - target
    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error.square()))
    mape = torch.mean(torch.abs(error) / torch.abs(target).clamp_min(1e-12)) * 100.0
    centered = target - torch.mean(target)
    denominator = torch.sum(centered.square())
    if bool((denominator == 0).item()):
        r2 = torch.zeros((), dtype=target.dtype, device=target.device)
    else:
        r2 = 1.0 - torch.sum(error.square()) / denominator
    return {
        "mae": _as_float(mae),
        "rmse": _as_float(rmse),
        "mape": _as_float(mape),
        "r2": _as_float(r2),
    }


def _trajectory_metrics(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> dict[str, float]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("trajectory prediction, target and mask must align")
    weights = mask.to(dtype=prediction.dtype)
    count = weights.sum().clamp_min(1.0)
    safe_target = torch.where(mask, target, prediction)
    error = prediction - safe_target
    mae = (torch.abs(error) * weights).sum() / count
    rmse = torch.sqrt((error.square() * weights).sum() / count)
    adjacent = mask[:, 1:] & mask[:, :-1]
    adjacent_weights = adjacent.to(dtype=prediction.dtype)
    violations = (prediction[:, 1:] > prediction[:, :-1]).to(
        dtype=prediction.dtype
    )
    violation_rate = (violations * adjacent_weights).sum() / adjacent_weights.sum().clamp_min(
        1.0
    )
    return {
        "mae": _as_float(mae),
        "rmse": _as_float(rmse),
        "monotonic_violation_rate": _as_float(violation_rate),
    }


def _seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _require_finite_loss(loss: Tensor) -> None:
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError("advanced training produced a non-finite scalar loss")


def _as_float(value: Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("advanced training metric must be finite")
    return result
