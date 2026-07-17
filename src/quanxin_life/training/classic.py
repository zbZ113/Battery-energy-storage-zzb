"""Target-aware classic and XGBoost baselines for the A100 experiment matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from quanxin_life.core import PredictionTarget
from quanxin_life.training.tasks import CycleLifeCurveBatch

_FEATURE_NAMES = (
    "curve_coverage",
    "first_capacity_ah",
    "last_capacity_ah",
    "capacity_change_ah",
    "delta_q_mean_ah",
    "delta_q_variance_ah2",
    "delta_q_min_ah",
    "delta_q_max_ah",
    "delta_q_l2_ah",
)


@dataclass(frozen=True)
class CurveTabularBatch:
    cell_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray
    cutoff_cycle: int


@dataclass(frozen=True)
class DummyCycleLifeModel:
    predicted_cycle: float

    def predict(self, batch: CycleLifeCurveBatch) -> np.ndarray:
        _require_official_target(batch)
        return np.full(len(batch.cell_ids), self.predicted_cycle, dtype=np.float64)


@dataclass(frozen=True)
class VarianceCycleLifeModel:
    intercept: float
    coefficient: float

    def predict(self, batch: CycleLifeCurveBatch) -> np.ndarray:
        tabular = curve_batch_to_tabular(batch)
        variance_index = tabular.feature_names.index("delta_q_variance_ah2")
        log_variance = np.log(np.maximum(tabular.values[:, variance_index], 1e-12))
        predicted = np.exp(self.intercept + self.coefficient * log_variance)
        return np.asarray(
            np.maximum(predicted, float(batch.cutoff_cycle)), dtype=np.float64
        )


@dataclass
class XGBoostCycleLifeResult:
    booster: xgb.Booster
    feature_names: tuple[str, ...]
    best_iteration: int
    evaluation_history: dict[str, dict[str, list[float]]]

    def predict(self, batch: CycleLifeCurveBatch) -> np.ndarray:
        tabular = curve_batch_to_tabular(batch)
        if tabular.feature_names != self.feature_names:
            raise ValueError("XGBoost feature schema does not match the fitted model")
        matrix = xgb.DMatrix(tabular.values, feature_names=list(self.feature_names))
        prediction = np.asarray(
            self.booster.predict(matrix, iteration_range=(0, self.best_iteration + 1)),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("XGBoost produced non-finite cycle-life predictions")
        return np.maximum(prediction, float(batch.cutoff_cycle))

    def save_model(self, destination: Path) -> None:
        if destination.suffix.lower() != ".ubj":
            raise ValueError("formal XGBoost artifacts must use UBJ")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(destination)


def curve_batch_to_tabular(batch: CycleLifeCurveBatch) -> CurveTabularBatch:
    _require_official_target(batch)
    rows: list[list[float]] = []
    values = batch.curve_values.detach().cpu()
    masks = batch.observed_mask.detach().cpu()
    for cell_values, cell_mask in zip(values, masks, strict=True):
        indices = torch.nonzero(cell_mask, as_tuple=False).flatten()
        if indices.numel() < 2:
            raise ValueError("classic curve features require at least two observed cycles")
        first = cell_values[int(indices[0])].numpy().astype(np.float64, copy=False)
        last = cell_values[int(indices[-1])].numpy().astype(np.float64, copy=False)
        delta = last - first
        row = [
            float(cell_mask.float().mean()),
            float(np.max(first)),
            float(np.max(last)),
            float(np.max(last) - np.max(first)),
            float(np.mean(delta)),
            float(np.var(delta)),
            float(np.min(delta)),
            float(np.max(delta)),
            float(np.linalg.norm(delta)),
        ]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("classic curve features must be finite")
        rows.append(row)
    return CurveTabularBatch(
        cell_ids=batch.cell_ids,
        feature_names=_FEATURE_NAMES,
        values=np.asarray(rows, dtype=np.float32),
        labels=batch.observed_cycles.detach().cpu().numpy().astype(np.float32, copy=False),
        cutoff_cycle=batch.cutoff_cycle,
    )


def fit_dummy_cycle_life(batch: CycleLifeCurveBatch) -> DummyCycleLifeModel:
    _require_official_target(batch)
    value = float(batch.observed_cycles.float().mean())
    if not math.isfinite(value):
        raise ValueError("Dummy training labels must be finite")
    return DummyCycleLifeModel(predicted_cycle=value)


def fit_variance_cycle_life(batch: CycleLifeCurveBatch) -> VarianceCycleLifeModel:
    tabular = curve_batch_to_tabular(batch)
    variance_index = tabular.feature_names.index("delta_q_variance_ah2")
    x_values = np.log(np.maximum(tabular.values[:, variance_index], 1e-12))
    y_values = np.log(tabular.labels)
    design = np.column_stack((np.ones_like(x_values), x_values))
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Variance baseline fitting produced non-finite coefficients")
    return VarianceCycleLifeModel(
        intercept=float(coefficients[0]),
        coefficient=float(coefficients[1]),
    )


def train_xgboost_cycle_life(
    *,
    train_batch: CycleLifeCurveBatch,
    validation_batch: CycleLifeCurveBatch,
    max_rounds: int,
    early_stopping_rounds: int,
    seed: int,
    device: str,
    checkpoint_directory: Path,
) -> XGBoostCycleLifeResult:
    if set(train_batch.cell_ids) & set(validation_batch.cell_ids):
        raise ValueError("XGBoost training and validation cohorts must be cell-disjoint")
    if max_rounds <= 0 or early_stopping_rounds <= 0:
        raise ValueError("XGBoost rounds and early stopping must be positive")
    train = curve_batch_to_tabular(train_batch)
    validation = curve_batch_to_tabular(validation_batch)
    if train.feature_names != validation.feature_names:
        raise ValueError("XGBoost training and validation feature schemas must match")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    dtrain = xgb.DMatrix(
        train.values,
        label=train.labels,
        feature_names=list(train.feature_names),
    )
    dvalidation = xgb.DMatrix(
        validation.values,
        label=validation.labels,
        feature_names=list(validation.feature_names),
    )
    history: dict[str, dict[str, list[float]]] = {}
    checkpoint_interval = min(50, max_rounds)
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "device": device,
            "seed": seed,
            "nthread": 1,
            "max_depth": 3,
            "eta": 0.05,
        },
        dtrain,
        num_boost_round=max_rounds,
        evals=((dtrain, "train"), (dvalidation, "validation")),
        early_stopping_rounds=early_stopping_rounds,
        evals_result=history,
        verbose_eval=1,
        callbacks=(
            xgb.callback.TrainingCheckPoint(
                directory=checkpoint_directory,
                name="round",
                as_pickle=False,
                interval=checkpoint_interval,
            ),
        ),
    )
    best_iteration = int(getattr(booster, "best_iteration", max_rounds - 1))
    return XGBoostCycleLifeResult(
        booster=booster,
        feature_names=train.feature_names,
        best_iteration=best_iteration,
        evaluation_history=history,
    )


def _require_official_target(batch: CycleLifeCurveBatch) -> None:
    if batch.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
        raise ValueError("classic MATR training requires the official cycle-life target")
