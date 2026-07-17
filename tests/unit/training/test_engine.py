from __future__ import annotations

from pathlib import Path

import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.checkpoint import CheckpointContext
from quanxin_life.training.config import ModelTrainingConfig
from quanxin_life.training.engine import (
    EpochMetrics,
    TrainingEngine,
    TrainingRunStatus,
)


class ToyTrainingTask:
    def __init__(self, validation_values: dict[int, float]) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
        self.validation_values = validation_values
        self.trained_epochs: list[int] = []
        self.validated_epochs: list[int] = []

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        self.trained_epochs.append(epoch)
        self.model.to(device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model(torch.ones((2, 1), device=device)) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return EpochMetrics(loss=float(loss.detach().cpu()), metrics={})

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        self.validated_epochs.append(epoch)
        value = self.validation_values[epoch]
        return EpochMetrics(loss=value, metrics={"mae": value})


def _context() -> CheckpointContext:
    return CheckpointContext(
        run_id="matr-cpmlp-c20-s20260712",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="cpmlp",
        cutoff_cycle=20,
        seed=20260712,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="matr-v1",
        split_version="split-v1",
        feature_version="curve-v1",
        source_commit="c" * 40,
    )


def test_engine_validates_early_stops_and_writes_consistent_evidence(
    tmp_path: Path,
) -> None:
    task = ToyTrainingTask({2: 1.0, 4: 1.1, 6: 1.2})
    engine = TrainingEngine(
        task=task,
        context=_context(),
        config=ModelTrainingConfig(
            name="cpmlp",
            max_epochs=10,
            validation_interval=2,
            early_stopping_patience=2,
        ),
        run_directory=tmp_path,
        device=torch.device("cpu"),
    )

    result = engine.run()

    assert result.status is TrainingRunStatus.EARLY_STOPPED
    assert result.last_epoch == 6
    assert result.best_epoch == 2
    assert task.validated_epochs == [2, 4, 6]
    assert (tmp_path / "training_log.jsonl").is_file()
    assert (tmp_path / "metrics_epoch.csv").is_file()
    assert (tmp_path / "metrics_validation.csv").is_file()
    assert (tmp_path / "checkpoints" / "last.json").is_file()
    assert (tmp_path / "checkpoints" / "best.json").is_file()
    assert (tmp_path / "run_status.json").is_file()


def test_engine_resumes_last_checkpoint_and_completed_run_is_skipped(
    tmp_path: Path,
) -> None:
    def stop_after_two() -> bool:
        return (tmp_path / "stop").exists()
    first_task = ToyTrainingTask({2: 1.0, 4: 0.9})
    original_train = first_task.train_epoch

    def train_and_request_stop(epoch: int, *, device: torch.device) -> EpochMetrics:
        metrics = original_train(epoch, device=device)
        if epoch == 2:
            (tmp_path / "stop").touch()
        return metrics

    first_task.train_epoch = train_and_request_stop  # type: ignore[method-assign]
    config = ModelTrainingConfig(
        name="cpmlp",
        max_epochs=4,
        validation_interval=2,
        early_stopping_patience=3,
    )
    first = TrainingEngine(
        task=first_task,
        context=_context(),
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
        stop_requested=stop_after_two,
    ).run()
    assert first.status is TrainingRunStatus.INTERRUPTED
    assert first.last_epoch == 2

    (tmp_path / "stop").unlink()
    resumed_task = ToyTrainingTask({2: 1.0, 4: 0.9})
    resumed = TrainingEngine(
        task=resumed_task,
        context=_context(),
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
    ).run()
    assert resumed.status is TrainingRunStatus.COMPLETED
    assert resumed.resumed_from_epoch == 2
    assert resumed_task.trained_epochs == [3, 4]

    skipped_task = ToyTrainingTask({2: 1.0, 4: 0.9})
    skipped = TrainingEngine(
        task=skipped_task,
        context=_context(),
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
    ).run()
    assert skipped.status is TrainingRunStatus.SKIPPED_COMPLETED
    assert skipped_task.trained_epochs == []
    for expected, actual in zip(
        resumed_task.model.parameters(), skipped_task.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_completed_run_cannot_be_reused_with_another_context(tmp_path: Path) -> None:
    config = ModelTrainingConfig(
        name="cpmlp",
        max_epochs=2,
        validation_interval=2,
        early_stopping_patience=3,
    )
    TrainingEngine(
        task=ToyTrainingTask({2: 1.0}),
        context=_context(),
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
    ).run()

    changed_context = _context().model_copy(update={"input_bundle_sha256": "d" * 64})
    try:
        TrainingEngine(
            task=ToyTrainingTask({2: 1.0}),
            context=changed_context,
            config=config,
            run_directory=tmp_path,
            device=torch.device("cpu"),
        ).run()
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a completed run with another context must be rejected")
