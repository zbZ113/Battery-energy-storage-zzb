"""Target-aware classic and XGBoost baselines for the A100 experiment matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb

from quanxin_life.core import PredictionTarget, sha256_canonical
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
    resumed_from_round: int | None = None

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
    checkpoint_interval: int | None = None,
    log_directory: Path | None = None,
    _interrupt_after_round: int | None = None,
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
    context_sha256 = _xgboost_context_sha256(
        train=train,
        validation=validation,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        device=device,
    )
    recovered = _load_xgboost_checkpoint(
        checkpoint_directory,
        expected_context_sha256=context_sha256,
    )
    if recovered is None:
        initial_booster = None
        completed_rounds = 0
        best_iteration = -1
        best_score: float | None = None
        no_improvement = 0
        history: dict[str, dict[str, list[float]]] = {}
    else:
        (
            initial_booster,
            completed_rounds,
            best_iteration,
            best_score,
            no_improvement,
            history,
        ) = recovered
    interval = checkpoint_interval or min(50, max_rounds)
    if interval <= 0:
        raise ValueError("XGBoost checkpoint interval must be positive")
    callback = _XGBoostRecoveryCallback(
        checkpoint_directory=checkpoint_directory,
        context_sha256=context_sha256,
        starting_round=completed_rounds,
        checkpoint_interval=interval,
        early_stopping_rounds=early_stopping_rounds,
        best_iteration=best_iteration,
        best_score=best_score,
        no_improvement=no_improvement,
        history=history,
        log_directory=log_directory or checkpoint_directory.parent,
        interrupt_after_round=_interrupt_after_round,
    )
    remaining_rounds = max_rounds - completed_rounds
    if remaining_rounds <= 0:
        assert initial_booster is not None
        return XGBoostCycleLifeResult(
            booster=initial_booster,
            feature_names=train.feature_names,
            best_iteration=best_iteration,
            evaluation_history=history,
            resumed_from_round=completed_rounds,
        )
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
        num_boost_round=remaining_rounds,
        evals=((dtrain, "train"), (dvalidation, "validation")),
        verbose_eval=False,
        callbacks=(callback,),
        xgb_model=initial_booster,
    )
    return XGBoostCycleLifeResult(
        booster=booster,
        feature_names=train.feature_names,
        best_iteration=callback.best_iteration,
        evaluation_history=callback.history,
        resumed_from_round=completed_rounds or None,
    )


class _XGBoostRecoveryCallback(xgb.callback.TrainingCallback):
    def __init__(
        self,
        *,
        checkpoint_directory: Path,
        context_sha256: str,
        starting_round: int,
        checkpoint_interval: int,
        early_stopping_rounds: int,
        best_iteration: int,
        best_score: float | None,
        no_improvement: int,
        history: dict[str, dict[str, list[float]]],
        log_directory: Path,
        interrupt_after_round: int | None,
    ) -> None:
        self.checkpoint_directory = checkpoint_directory
        self.context_sha256 = context_sha256
        self.starting_round = starting_round
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iteration = best_iteration
        self.best_score = best_score
        self.no_improvement = no_improvement
        self.history = history
        self.log_directory = log_directory
        self.interrupt_after_round = interrupt_after_round
        self.completed_rounds = starting_round
        self.last_saved_round = starting_round

    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
    ) -> bool:
        global_round = self.starting_round + epoch + 1
        values = _append_xgboost_history(self.history, evals_log)
        validation_mae = values["validation"]["mae"]
        if self.best_score is None or validation_mae < self.best_score:
            self.best_score = validation_mae
            self.best_iteration = global_round - 1
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        self.completed_rounds = global_round
        model.set_attr(
            best_iteration=str(self.best_iteration),
            best_score=str(self.best_score),
        )
        should_stop = self.no_improvement >= self.early_stopping_rounds
        if global_round % self.checkpoint_interval == 0 or should_stop:
            self._save(model)
        print(
            f"xgboost_round={global_round} train_mae={values['train']['mae']:.6g} "
            f"validation_mae={validation_mae:.6g} "
            f"best_iteration={self.best_iteration} "
            f"early_stop_counter={self.no_improvement}"
        )
        if (
            self.interrupt_after_round is not None
            and global_round >= self.interrupt_after_round
        ):
            if self.last_saved_round != global_round:
                self._save(model)
            raise RuntimeError("synthetic XGBoost interruption")
        return should_stop

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if self.completed_rounds > self.last_saved_round:
            self._save(model)
        return model

    def _save(self, model: xgb.Booster) -> None:
        _save_xgboost_checkpoint(
            self.checkpoint_directory,
            context_sha256=self.context_sha256,
            completed_rounds=self.completed_rounds,
            best_iteration=self.best_iteration,
            best_score=self.best_score,
            no_improvement=self.no_improvement,
            history=self.history,
            booster=model,
        )
        _write_xgboost_logs(self.log_directory, self.history)
        self.last_saved_round = self.completed_rounds


def _append_xgboost_history(
    history: dict[str, dict[str, list[float]]],
    evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
) -> dict[str, dict[str, float]]:
    current: dict[str, dict[str, float]] = {}
    for dataset, metrics in evals_log.items():
        current[dataset] = {}
        for metric_name, metric_values in metrics.items():
            latest = metric_values[-1]
            value = float(latest[0] if isinstance(latest, tuple) else latest)
            if not math.isfinite(value):
                raise ValueError("XGBoost evaluation metrics must be finite")
            history.setdefault(dataset, {}).setdefault(metric_name, []).append(value)
            current[dataset][metric_name] = value
    if "validation" not in current or "mae" not in current["validation"]:
        raise ValueError("XGBoost recovery requires validation MAE")
    return current


def _xgboost_context_sha256(
    *,
    train: CurveTabularBatch,
    validation: CurveTabularBatch,
    max_rounds: int,
    early_stopping_rounds: int,
    seed: int,
    device: str,
) -> str:
    return sha256_canonical(
        {
            "schema_version": "xgboost-recovery-context-v1",
            "train_cell_ids": train.cell_ids,
            "validation_cell_ids": validation.cell_ids,
            "feature_names": train.feature_names,
            "train_values_sha256": hashlib.sha256(train.values.tobytes()).hexdigest(),
            "train_labels_sha256": hashlib.sha256(train.labels.tobytes()).hexdigest(),
            "validation_values_sha256": hashlib.sha256(
                validation.values.tobytes()
            ).hexdigest(),
            "validation_labels_sha256": hashlib.sha256(
                validation.labels.tobytes()
            ).hexdigest(),
            "max_rounds": max_rounds,
            "early_stopping_rounds": early_stopping_rounds,
            "seed": seed,
            "device": device,
        }
    )


def _save_xgboost_checkpoint(
    checkpoint_directory: Path,
    *,
    context_sha256: str,
    completed_rounds: int,
    best_iteration: int,
    best_score: float | None,
    no_improvement: int,
    history: dict[str, dict[str, list[float]]],
    booster: xgb.Booster,
) -> None:
    name = f"round-{completed_rounds:06d}"
    target = checkpoint_directory / name
    temporary = checkpoint_directory / f".{name}.{os.getpid()}.tmp"
    if target.exists() or temporary.exists():
        raise ValueError("XGBoost checkpoint round path already exists")
    temporary.mkdir()
    try:
        model_path = temporary / "model.ubj"
        state_path = temporary / "state.json"
        booster.save_model(model_path)
        state = {
            "schema_version": "xgboost-recovery-state-v1",
            "context_sha256": context_sha256,
            "completed_rounds": completed_rounds,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "no_improvement": no_improvement,
            "evaluation_history": history,
        }
        _write_json(state_path, state)
        manifest_payload = {
            "schema_version": "xgboost-recovery-manifest-v1",
            "context_sha256": context_sha256,
            "files": [
                _file_row(model_path),
                _file_row(state_path),
            ],
        }
        _write_json(
            temporary / "manifest.json",
            {
                **manifest_payload,
                "manifest_sha256": sha256_canonical(manifest_payload),
            },
        )
        temporary.replace(target)
        _write_json_atomic(checkpoint_directory / "last.json", {"checkpoint": name})
        _prune_xgboost_checkpoints(checkpoint_directory, keep_recent=3)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_xgboost_checkpoint(
    checkpoint_directory: Path,
    *,
    expected_context_sha256: str,
) -> tuple[
    xgb.Booster,
    int,
    int,
    float | None,
    int,
    dict[str, dict[str, list[float]]],
] | None:
    pointer = checkpoint_directory / "last.json"
    if not pointer.exists():
        return None
    pointer_payload = _read_json(pointer)
    if set(pointer_payload) != {"checkpoint"}:
        raise ValueError("XGBoost checkpoint pointer schema is invalid")
    name = pointer_payload["checkpoint"]
    if not isinstance(name, str) or not name.startswith("round-"):
        raise ValueError("XGBoost checkpoint pointer is invalid")
    root = checkpoint_directory / name
    if root.is_symlink() or not root.is_dir():
        raise ValueError("XGBoost checkpoint directory is invalid")
    manifest = _read_json(root / "manifest.json")
    claimed_manifest_sha256 = manifest.pop("manifest_sha256", None)
    if claimed_manifest_sha256 != sha256_canonical(manifest):
        raise ValueError("XGBoost checkpoint manifest SHA-256 mismatch")
    if manifest.get("context_sha256") != expected_context_sha256:
        raise ValueError("XGBoost checkpoint context does not match this run")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("XGBoost checkpoint file inventory is invalid")
    expected_names = {"manifest.json"}
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("XGBoost checkpoint file row is invalid")
        relative = row["relative_path"]
        if relative not in {"model.ubj", "state.json"}:
            raise ValueError("XGBoost checkpoint contains an unapproved file")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("XGBoost checkpoint file is missing")
        if path.stat().st_size != row["size_bytes"]:
            raise ValueError("XGBoost checkpoint file size mismatch")
        if _sha256_file(path) != row["sha256"]:
            raise ValueError("XGBoost checkpoint file SHA-256 mismatch")
        expected_names.add(relative)
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("XGBoost checkpoint contains an unexpected file")
    state = _read_json(root / "state.json")
    if state.get("context_sha256") != expected_context_sha256:
        raise ValueError("XGBoost checkpoint state context mismatch")
    completed_rounds = _required_nonnegative_int(state, "completed_rounds")
    best_iteration = _required_nonnegative_int(state, "best_iteration")
    no_improvement = _required_nonnegative_int(state, "no_improvement")
    best_score_value = state.get("best_score")
    if best_score_value is not None and not isinstance(best_score_value, (int, float)):
        raise ValueError("XGBoost checkpoint best score is invalid")
    history_value = state.get("evaluation_history")
    if not isinstance(history_value, dict):
        raise ValueError("XGBoost checkpoint evaluation history is invalid")
    history = _validate_xgboost_history(history_value, completed_rounds)
    booster = xgb.Booster()
    booster.load_model(root / "model.ubj")
    if booster.num_boosted_rounds() != completed_rounds:
        raise ValueError("XGBoost checkpoint round count does not match its state")
    return (
        booster,
        completed_rounds,
        best_iteration,
        None if best_score_value is None else float(best_score_value),
        no_improvement,
        history,
    )


def _validate_xgboost_history(
    value: dict[str, Any], completed_rounds: int
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for dataset, metrics in value.items():
        if not isinstance(dataset, str) or not isinstance(metrics, dict):
            raise ValueError("XGBoost checkpoint history is invalid")
        result[dataset] = {}
        for name, values in metrics.items():
            if not isinstance(name, str) or not isinstance(values, list):
                raise ValueError("XGBoost checkpoint metric history is invalid")
            normalized = [float(item) for item in values]
            if len(normalized) != completed_rounds or not all(
                math.isfinite(item) for item in normalized
            ):
                raise ValueError("XGBoost checkpoint metric history length is invalid")
            result[dataset][name] = normalized
    return result


def _required_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"XGBoost checkpoint {key} is invalid")
    return value


def _file_row(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("XGBoost checkpoint JSON must be a regular file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    payload = json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("XGBoost checkpoint JSON must contain an object")
    return payload


def _prune_xgboost_checkpoints(checkpoint_directory: Path, *, keep_recent: int) -> None:
    root = checkpoint_directory.resolve(strict=True)
    directories = sorted(
        path
        for path in checkpoint_directory.glob("round-*")
        if path.is_dir() and not path.is_symlink()
    )
    for path in directories[:-keep_recent]:
        resolved = path.resolve(strict=True)
        if resolved.parent != root:
            raise ValueError("XGBoost checkpoint pruning target escapes its root")
        shutil.rmtree(resolved)


def _write_xgboost_logs(
    log_directory: Path,
    history: dict[str, dict[str, list[float]]],
) -> None:
    train_mae = history.get("train", {}).get("mae", [])
    validation_mae = history.get("validation", {}).get("mae", [])
    if len(train_mae) != len(validation_mae):
        raise ValueError("XGBoost train and validation histories must align")
    log_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        (index, train_value, validation_value)
        for index, (train_value, validation_value) in enumerate(
            zip(train_mae, validation_mae, strict=True),
            start=1,
        )
    ]
    jsonl = "".join(
        json.dumps(
            {
                "round": round_index,
                "train_mae": train_value,
                "validation_mae": validation_value,
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
        for round_index, train_value, validation_value in rows
    )
    _write_text_atomic(log_directory / "training_log.jsonl", jsonl)
    _write_csv_atomic(
        log_directory / "metrics_epoch.csv",
        ("round", "train_mae"),
        [(round_index, train_value) for round_index, train_value, _validation in rows],
    )
    _write_csv_atomic(
        log_directory / "metrics_validation.csv",
        ("round", "validation_mae"),
        [
            (round_index, validation_value)
            for round_index, _train, validation_value in rows
        ],
    )


def _write_text_atomic(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[tuple[int, float]],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    temporary.replace(path)


def _require_official_target(batch: CycleLifeCurveBatch) -> None:
    if batch.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
        raise ValueError("classic MATR training requires the official cycle-life target")
