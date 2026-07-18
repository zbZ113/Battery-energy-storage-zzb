"""Numerical training tasks consumed by the shared resumable Trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from quanxin_life.core import PredictionTarget
from quanxin_life.models.cpmlp import _CPMLPNetwork
from quanxin_life.models.hybrid_degradation import (
    _HybridNetwork,
    normalise_prediction_cycle_positions,
)
from quanxin_life.training.engine import EpochMetrics


@dataclass(frozen=True)
class CycleLifeCurveBatch:
    """One cell-disjoint curve cohort with explicit observed life events."""

    dataset_id: str
    target: PredictionTarget
    cell_ids: tuple[str, ...]
    curve_values: Tensor
    observed_mask: Tensor
    observed_cycles: Tensor
    cutoff_cycle: int

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.cell_ids:
            raise ValueError("curve batch requires a dataset and at least one cell")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("curve batch cell_ids must be unique")
        if self.cutoff_cycle < 0:
            raise ValueError("curve batch cutoff_cycle must be non-negative")
        if self.curve_values.ndim != 3:
            raise ValueError("curve_values must have shape [cell, cycle, voltage]")
        if self.observed_mask.shape != self.curve_values.shape[:2]:
            raise ValueError("observed_mask must align with curve_values")
        if self.observed_mask.dtype is not torch.bool:
            raise ValueError("observed_mask must use torch.bool")
        if self.observed_cycles.shape != (len(self.cell_ids),):
            raise ValueError("observed cycle labels must align with cell_ids")
        if self.curve_values.shape[0] != len(self.cell_ids):
            raise ValueError("curve_values must align with cell_ids")
        if torch.any(self.observed_mask.sum(dim=1) == 0):
            raise ValueError("every curve batch cell requires an observed curve")
        observed = self.curve_values[self.observed_mask]
        absent = self.curve_values[~self.observed_mask]
        if not torch.isfinite(observed).all():
            raise ValueError("observed curve values must be finite")
        if absent.numel() and not torch.isnan(absent).all():
            raise ValueError("unobserved curve values must remain NaN")
        if not torch.isfinite(self.observed_cycles).all() or torch.any(
            self.observed_cycles <= self.cutoff_cycle
        ):
            raise ValueError("observed cycle labels must be finite and after the cutoff")


@dataclass(frozen=True)
class HybridTrajectoryBatch:
    """One split cohort of real, cycle-indexed SOH supervision through cycle 500."""

    dataset_id: str
    cell_ids: tuple[str, ...]
    features: Tensor
    initial_soh: Tensor
    target_soh: Tensor
    prediction_cycles: tuple[int, ...]
    cutoff_cycle: int

    def __post_init__(self) -> None:
        cell_count = len(self.cell_ids)
        if not self.dataset_id or cell_count == 0:
            raise ValueError("trajectory batch requires a dataset and at least one cell")
        if len(set(self.cell_ids)) != cell_count:
            raise ValueError("trajectory batch cell_ids must be unique")
        if self.features.ndim != 2 or self.features.shape[0] != cell_count:
            raise ValueError("trajectory features must have shape [cell, feature]")
        if self.initial_soh.shape != (cell_count,):
            raise ValueError("initial_soh must align with cell_ids")
        if self.target_soh.shape != (cell_count, len(self.prediction_cycles)):
            raise ValueError("target_soh must align with cells and prediction cycles")
        if len(self.prediction_cycles) < 3 or self.prediction_cycles[-1] != 500:
            raise ValueError("Hybrid supervision must end at real cycle 500")
        if any(
            current <= previous
            for previous, current in zip(
                self.prediction_cycles, self.prediction_cycles[1:], strict=False
            )
        ) or any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction cycles must increase strictly after the cutoff")
        for tensor in (self.features, self.initial_soh, self.target_soh):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("trajectory supervision tensors must be finite floats")
        if torch.any(self.initial_soh <= 0) or torch.any(self.target_soh <= 0):
            raise ValueError("trajectory SOH values must be positive")


class CPMLPTrainingTask:
    """Full-batch CPMLP optimization for the explicit MATR official target."""

    def __init__(
        self,
        *,
        train_batch: CycleLifeCurveBatch,
        validation_batch: CycleLifeCurveBatch,
        curve_hidden_dim: int,
        aggregation_hidden_dim: int,
        learning_rate: float,
        seed: int = 20260712,
    ) -> None:
        if (
            train_batch.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
            or validation_batch.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        ):
            raise ValueError("CPMLP A100 task requires the MATR official cycle-life target")
        _validate_curve_cohorts(train_batch, validation_batch)
        _seed_everything(seed)
        self.train_batch = train_batch
        self.validation_batch = validation_batch
        self.model = _CPMLPNetwork(
            voltage_points=train_batch.curve_values.shape[2],
            cutoff_cycle=train_batch.cutoff_cycle,
            curve_hidden_dim=curve_hidden_dim,
            aggregation_hidden_dim=aggregation_hidden_dim,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predicted = self.model(
            self.train_batch.curve_values.to(device),
            self.train_batch.observed_mask.to(device),
        )
        target = self.train_batch.observed_cycles.to(device)
        loss = functional.mse_loss(predicted, target)
        if not torch.isfinite(loss):
            raise RuntimeError("CPMLP training produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return EpochMetrics(loss=float(loss.detach().cpu()), metrics={})

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(
                self.validation_batch.curve_values.to(device),
                self.validation_batch.observed_mask.to(device),
            )
        target = self.validation_batch.observed_cycles.to(device)
        return _cycle_metrics(predicted, target)


class HybridTrajectoryTrainingTask:
    """Full-batch optimization on genuine observed SOH trajectories through cycle 500."""

    def __init__(
        self,
        *,
        train_batch: HybridTrajectoryBatch,
        validation_batch: HybridTrajectoryBatch,
        hidden_dim: int,
        learning_rate: float,
        seed: int = 20260712,
    ) -> None:
        _validate_trajectory_cohorts(train_batch, validation_batch)
        _seed_everything(seed)
        self.train_batch = train_batch
        self.validation_batch = validation_batch
        self.model = _HybridNetwork(
            input_dim=train_batch.features.shape[1],
            horizon=len(train_batch.prediction_cycles),
            hidden_dim=hidden_dim,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        self._cycle_positions = torch.tensor(
            normalise_prediction_cycle_positions(
                prediction_cycles=train_batch.prediction_cycles,
                cutoff_cycle=train_batch.cutoff_cycle,
            ),
            dtype=torch.float32,
        )

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predicted = self._predict(self.train_batch, device=device)
        target = self.train_batch.target_soh.to(device)
        fit_loss = functional.mse_loss(predicted, target)
        smooth_loss = (
            predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2]
        ).square().mean()
        loss = fit_loss + 0.01 * smooth_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Hybrid training produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return EpochMetrics(loss=float(loss.detach().cpu()), metrics={})

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del epoch
        return self.evaluate(self.validation_batch, device=device)

    def predict(
        self,
        batch: HybridTrajectoryBatch,
        *,
        device: torch.device,
    ) -> Tensor:
        _validate_trajectory_cohorts(self.train_batch, batch)
        self.model.eval()
        with torch.no_grad():
            return self._predict(batch, device=device)

    def evaluate(
        self,
        batch: HybridTrajectoryBatch,
        *,
        device: torch.device,
    ) -> EpochMetrics:
        predicted = self.predict(batch, device=device)
        target = batch.target_soh.to(device)
        errors = predicted - target
        mae = float(errors.abs().mean().cpu())
        rmse = float(errors.square().mean().sqrt().cpu())
        violations = float((predicted[:, 1:] > predicted[:, :-1]).float().mean().cpu())
        return EpochMetrics(
            loss=mae,
            metrics={
                "mae": mae,
                "monotonic_violation_rate": violations,
                "rmse": rmse,
            },
        )

    def _predict(self, batch: HybridTrajectoryBatch, *, device: torch.device) -> Tensor:
        return cast(
            Tensor,
            self.model(
                batch.features.to(device),
                batch.initial_soh.to(device),
                self._cycle_positions.to(device),
            ),
        )


def _validate_curve_cohorts(
    train: CycleLifeCurveBatch, validation: CycleLifeCurveBatch
) -> None:
    if set(train.cell_ids) & set(validation.cell_ids):
        raise ValueError("training and validation curve cohorts must be cell-disjoint")
    if (
        train.dataset_id != validation.dataset_id
        or train.target is not validation.target
        or train.cutoff_cycle != validation.cutoff_cycle
        or train.curve_values.shape[1:] != validation.curve_values.shape[1:]
    ):
        raise ValueError("training and validation curve contracts must match")


def _validate_trajectory_cohorts(
    train: HybridTrajectoryBatch, validation: HybridTrajectoryBatch
) -> None:
    if set(train.cell_ids) & set(validation.cell_ids):
        raise ValueError("training and validation trajectory cohorts must be cell-disjoint")
    if (
        train.dataset_id != validation.dataset_id
        or train.cutoff_cycle != validation.cutoff_cycle
        or train.prediction_cycles != validation.prediction_cycles
        or train.features.shape[1] != validation.features.shape[1]
    ):
        raise ValueError("training and validation trajectory contracts must match")


def _cycle_metrics(predicted: Tensor, target: Tensor) -> EpochMetrics:
    errors = predicted - target
    mae = float(errors.abs().mean().cpu())
    rmse = float(errors.square().mean().sqrt().cpu())
    mape = float((errors.abs() / target).mean().mul(100).cpu())
    centered = target - target.mean()
    total = centered.square().sum()
    metrics = {"mae": mae, "mape": mape, "rmse": rmse}
    if float(total.cpu()) > 0:
        residual = errors.square().sum()
        metrics["r2"] = float((1 - residual / total).cpu())
    return EpochMetrics(loss=mae, metrics=dict(sorted(metrics.items())))


def _seed_everything(seed: int) -> None:
    if not isinstance(seed, int):
        raise ValueError("training seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
