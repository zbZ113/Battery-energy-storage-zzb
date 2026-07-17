"""Shared epoch lifecycle with validation, early stopping and safe recovery."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import torch
from pydantic import ConfigDict, Field

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.training.checkpoint import (
    CheckpointContext,
    TrainingCheckpointManifest,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from quanxin_life.training.config import ModelTrainingConfig


class EpochMetrics(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: float = Field(ge=0, allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)


class TrainingRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    EARLY_STOPPED = "EARLY_STOPPED"
    INTERRUPTED = "INTERRUPTED"
    SKIPPED_COMPLETED = "SKIPPED_COMPLETED"


class TrainingRunResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "training-run-result-v1"
    run_id: str = Field(min_length=1)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TrainingRunStatus
    last_epoch: int = Field(ge=0)
    best_epoch: int | None = Field(default=None, ge=0)
    best_metric: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    resumed_from_epoch: int | None = Field(default=None, ge=0)
    finished_at: datetime


Scheduler = (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
)


class TrainingTask(Protocol):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Scheduler | None

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics: ...

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics: ...


class TrainingEngine:
    """Run one model/cutoff/seed task without ever consulting the test split."""

    def __init__(
        self,
        *,
        task: TrainingTask,
        context: CheckpointContext,
        config: ModelTrainingConfig,
        run_directory: Path,
        device: torch.device,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if config.name != context.model_name:
            raise ValueError("training config model does not match checkpoint context")
        self.task = task
        self.context = context
        self.config = config
        self.run_directory = run_directory
        self.device = device
        self.stop_requested = stop_requested or (lambda: False)

    def run(self) -> TrainingRunResult:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        checkpoints = self.run_directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        self.task.model.to(self.device)
        completed = self._load_terminal_result()
        if completed is not None:
            pointer_name = "best.json" if (checkpoints / "best.json").is_file() else "last.json"
            self._restore_pointer(checkpoints, pointer_name)
            return completed.model_copy(update={"status": TrainingRunStatus.SKIPPED_COMPLETED})

        start_epoch, progress = self._resume(checkpoints)
        resumed_from = progress.epoch if progress.epoch > 0 else None
        logger = _RunLogger(self.run_directory, self.context)
        signal_flag = _SignalFlag()
        signal_flag.install()
        status = TrainingRunStatus.COMPLETED
        try:
            for epoch in range(start_epoch, self.config.max_epochs + 1):
                epoch_started = time.perf_counter()
                train = self.task.train_epoch(epoch, device=self.device)
                _validate_metrics(train)
                validation: EpochMetrics | None = None
                improved = False
                best_epoch: int | None = progress.best_epoch
                best_metric: float | None = progress.best_metric
                no_improvement = progress.validations_without_improvement
                if epoch % self.config.validation_interval == 0:
                    validation = self.task.validate(epoch, device=self.device)
                    _validate_metrics(validation)
                    metric = validation.metrics.get("mae", validation.loss)
                    if progress.best_metric is None or metric < progress.best_metric:
                        improved = True
                        best_epoch = epoch
                        best_metric = metric
                        no_improvement = 0
                    else:
                        best_epoch = progress.best_epoch
                        best_metric = progress.best_metric
                        no_improvement = progress.validations_without_improvement + 1
                    if self.task.scheduler is not None:
                        if isinstance(
                            self.task.scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.task.scheduler.step(metric)
                        else:
                            self.task.scheduler.step()

                progress = TrainingProgress(
                    epoch=epoch,
                    global_step=progress.global_step + 1,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    validations_without_improvement=no_improvement,
                )
                checkpoint_name = self._save_checkpoint(checkpoints, progress)
                _write_pointer(checkpoints / "last.json", checkpoint_name)
                if improved:
                    _write_pointer(checkpoints / "best.json", checkpoint_name)
                elapsed = time.perf_counter() - epoch_started
                logger.record(
                    epoch,
                    train=train,
                    validation=validation,
                    elapsed=elapsed,
                    learning_rate=float(self.task.optimizer.param_groups[0]["lr"]),
                )

                if (
                    validation is not None
                    and no_improvement >= self.config.early_stopping_patience
                ):
                    status = TrainingRunStatus.EARLY_STOPPED
                    break
                if signal_flag.requested or self.stop_requested():
                    status = TrainingRunStatus.INTERRUPTED
                    break
        finally:
            signal_flag.restore()

        result = TrainingRunResult(
            run_id=self.context.run_id,
            context_sha256=self._context_sha256(),
            status=status,
            last_epoch=progress.epoch,
            best_epoch=progress.best_epoch,
            best_metric=progress.best_metric,
            resumed_from_epoch=resumed_from,
            finished_at=datetime.now(UTC),
        )
        _write_json_atomic(
            self.run_directory / "run_status.json",
            result.model_dump(mode="json"),
        )
        if status in {TrainingRunStatus.COMPLETED, TrainingRunStatus.EARLY_STOPPED}:
            pointer_name = "best.json" if (checkpoints / "best.json").is_file() else "last.json"
            self._restore_pointer(checkpoints, pointer_name)
        return result

    def _load_terminal_result(self) -> TrainingRunResult | None:
        path = self.run_directory / "run_status.json"
        if not path.is_file() or path.is_symlink():
            return None
        result = TrainingRunResult.model_validate(_read_json(path))
        if (
            result.run_id != self.context.run_id
            or result.context_sha256 != self._context_sha256()
        ):
            raise ValueError("completed training result does not match the requested run")
        if result.status in {
            TrainingRunStatus.COMPLETED,
            TrainingRunStatus.EARLY_STOPPED,
        }:
            return result
        return None

    def _context_sha256(self) -> str:
        return sha256_canonical(self.context.model_dump(mode="json"))

    def _resume(self, checkpoints: Path) -> tuple[int, TrainingProgress]:
        pointer = checkpoints / "last.json"
        if not pointer.exists():
            return 1, TrainingProgress(epoch=0, global_step=0)
        progress = self._restore_pointer(checkpoints, "last.json")
        return progress.epoch + 1, progress

    def _restore_pointer(self, checkpoints: Path, pointer_name: str) -> TrainingProgress:
        checkpoint_name = _read_pointer(checkpoints / pointer_name)
        checkpoint_root = checkpoints / checkpoint_name
        manifest = TrainingCheckpointManifest.model_validate(
            _read_json(checkpoint_root / "manifest.json")
        )
        return load_training_checkpoint(
            checkpoint_root,
            manifest,
            expected_context=self.context,
            model=self.task.model,
            optimizer=self.task.optimizer,
            scheduler=self.task.scheduler,
        )

    def _save_checkpoint(self, checkpoints: Path, progress: TrainingProgress) -> str:
        name = f"epoch-{progress.epoch:06d}"
        target = checkpoints / name
        temporary = checkpoints / f".{name}.{os.getpid()}.tmp"
        if target.exists() or temporary.exists():
            raise ValueError("checkpoint epoch path already exists")
        temporary.mkdir()
        try:
            save_training_checkpoint(
                temporary,
                context=self.context,
                progress=progress,
                model=self.task.model,
                optimizer=self.task.optimizer,
                scheduler=self.task.scheduler,
            )
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                for path in temporary.iterdir():
                    path.unlink(missing_ok=True)
                temporary.rmdir()
            raise
        return name


class _RunLogger:
    def __init__(self, run_directory: Path, context: CheckpointContext) -> None:
        self.run_directory = run_directory
        self.context = context

    def record(
        self,
        epoch: int,
        *,
        train: EpochMetrics,
        validation: EpochMetrics | None,
        elapsed: float,
        learning_rate: float,
    ) -> None:
        base: dict[str, object] = {
            "dataset": self.context.dataset_id,
            "model": self.context.model_name,
            "cutoff": self.context.cutoff_cycle,
            "seed": self.context.seed,
            "epoch": epoch,
            "train_loss": train.loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
        }
        event = dict(base)
        if validation is not None:
            event["validation_loss"] = validation.loss
            event.update(validation.metrics)
        with (self.run_directory / "training_log.jsonl").open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            handle.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        _append_csv(self.run_directory / "metrics_epoch.csv", base)
        if validation is not None:
            validation_row = {
                **base,
                "validation_loss": validation.loss,
                **validation.metrics,
            }
            _append_csv(self.run_directory / "metrics_validation.csv", validation_row)
        validation_text = (
            ""
            if validation is None
            else f" validation_loss={validation.loss:.6g}"
        )
        print(
            f"dataset={self.context.dataset_id} model={self.context.model_name} "
            f"cutoff={self.context.cutoff_cycle} seed={self.context.seed} "
            f"epoch={epoch} train_loss={train.loss:.6g}{validation_text} "
            f"learning_rate={learning_rate:.6g} elapsed_seconds={elapsed:.3f}"
        )


class _SignalFlag:
    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[signal.Signals, Any] = {}

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for name in ("SIGINT", "SIGTERM"):
            kind = getattr(signal, name, None)
            if kind is None:
                continue
            self._previous[kind] = signal.getsignal(kind)
            signal.signal(kind, self._handle)

    def restore(self) -> None:
        for kind, handler in self._previous.items():
            signal.signal(kind, handler)

    def _handle(self, _signum: int, _frame: object) -> None:
        self.requested = True


def _validate_metrics(metrics: EpochMetrics) -> None:
    if not math.isfinite(metrics.loss) or any(
        not math.isfinite(value) for value in metrics.metrics.values()
    ):
        raise ValueError("training metrics must be finite")


def _append_csv(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_pointer(path: Path, checkpoint_name: str) -> None:
    _write_json_atomic(path, {"checkpoint": checkpoint_name})


def _read_pointer(path: Path) -> str:
    payload = _read_json(path)
    if set(payload) != {"checkpoint"}:
        raise ValueError("checkpoint pointer has an invalid schema")
    value = payload["checkpoint"]
    if not isinstance(value, str) or not value.startswith("epoch-"):
        raise ValueError("checkpoint pointer is invalid")
    return value


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("training state JSON must be a regular file")
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
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
        raise ValueError("training state JSON must contain an object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
