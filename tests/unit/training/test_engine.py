from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.checkpoint import (
    AdvancedCheckpointContext,
    CheckpointContext,
    model_architecture_sha256,
)
from quanxin_life.training.config import ModelTrainingConfig
from quanxin_life.training.engine import (
    EpochMetrics,
    TrainingEngine,
    TrainingRunStatus,
)


class ToyTrainingTask:
    def __init__(
        self,
        validation_values: dict[int, float],
        *,
        stochastic: bool = False,
    ) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
        self.validation_values = validation_values
        self.stochastic = stochastic
        self.trained_epochs: list[int] = []
        self.validated_epochs: list[int] = []

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        self.trained_epochs.append(epoch)
        self.model.to(device)
        self.optimizer.zero_grad(set_to_none=True)
        features = (
            torch.randn((2, 1), device=device)
            if self.stochastic
            else torch.ones((2, 1), device=device)
        )
        loss = (self.model(features) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return EpochMetrics(loss=float(loss.detach().cpu()), metrics={})

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        self.validated_epochs.append(epoch)
        value = self.validation_values[epoch]
        return EpochMetrics(loss=value, metrics={"mae": value})


def _assert_nested_state_equal(expected: object, actual: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected) == set(actual)
        for key, value in expected.items():
            _assert_nested_state_equal(value, actual[key])
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual


def _stable_validation_rows(run_directory: Path) -> tuple[dict[str, str], ...]:
    with (run_directory / "metrics_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    return tuple(
        {key: value for key, value in row.items() if key != "elapsed_seconds"}
        for row in rows
    )


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


def _advanced_context(task: ToyTrainingTask) -> AdvancedCheckpointContext:
    candidate_config_sha256 = "d" * 64
    return AdvancedCheckpointContext(
        run_id="matr-cyclepatch-direct-c20-s20260712",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="cyclepatch_direct",
        cutoff_cycle=20,
        seed=20260712,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="matr-v1",
        split_version="split-v1",
        feature_version="curve-v2",
        source_commit="c" * 40,
        run_mode="select",
        stage="selection_candidate",
        candidate_config_sha256=candidate_config_sha256,
        model_architecture_sha256=model_architecture_sha256(
            task.model,
            candidate_config_sha256,
        ),
        normalization_sha256="e" * 64,
    )


def test_advanced_engine_pauses_at_30_and_resumes_to_90(tmp_path: Path) -> None:
    config = ModelTrainingConfig(
        name="cyclepatch_direct",
        max_epochs=90,
        validation_interval=30,
        early_stopping_patience=10,
    )
    first_task = ToyTrainingTask({30: 1.0, 60: 0.9, 90: 0.8})
    context = _advanced_context(first_task)

    paused = TrainingEngine(
        task=first_task,
        context=context,
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
        keep_recent_checkpoints=100,
    ).run(epoch_limit=30)

    assert paused.status is TrainingRunStatus.PAUSED_STAGE
    assert paused.last_epoch == 30
    assert first_task.trained_epochs == list(range(1, 31))
    assert json.loads(
        (tmp_path / "run_status.json").read_text(encoding="utf-8")
    )["status"] == "PAUSED_STAGE"
    pointer = json.loads(
        (tmp_path / "checkpoints" / "last.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            tmp_path
            / "checkpoints"
            / pointer["checkpoint"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "safe-training-checkpoint-v2"
    assert manifest["context"]["candidate_config_sha256"] == "d" * 64
    assert manifest["context"]["normalization_sha256"] == "e" * 64
    assert manifest["context"]["model_architecture_sha256"] == (
        context.model_architecture_sha256
    )

    resumed_task = ToyTrainingTask({30: 1.0, 60: 0.9, 90: 0.8})
    resumed = TrainingEngine(
        task=resumed_task,
        context=context,
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
        keep_recent_checkpoints=100,
    ).run(epoch_limit=90)

    assert resumed.status is TrainingRunStatus.COMPLETED
    assert resumed.resumed_from_epoch == 30
    assert resumed.last_epoch == 90
    assert resumed_task.trained_epochs == list(range(31, 91))


@pytest.mark.parametrize("epoch_limit", [0, 3, True, 1.5, "2"])
def test_engine_rejects_epoch_limit_outside_configured_range(
    tmp_path: Path,
    epoch_limit: object,
) -> None:
    task = ToyTrainingTask({2: 1.0})
    engine = TrainingEngine(
        task=task,
        context=_context(),
        config=ModelTrainingConfig(
            name="cpmlp",
            max_epochs=2,
            validation_interval=2,
            early_stopping_patience=3,
        ),
        run_directory=tmp_path,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="epoch_limit"):
        engine.run(epoch_limit=epoch_limit)  # type: ignore[arg-type]


def test_advanced_resume_matches_continuous_training_state(tmp_path: Path) -> None:
    config = ModelTrainingConfig(
        name="cyclepatch_direct",
        max_epochs=90,
        validation_interval=30,
        early_stopping_patience=10,
    )
    validation_values = {30: 1.0, 60: 0.9, 90: 0.8}

    torch.manual_seed(20260719)
    continuous_task = ToyTrainingTask(validation_values, stochastic=True)
    continuous_context = _advanced_context(continuous_task).model_copy(
        update={"run_id": "continuous-cyclepatch-direct"}
    )
    continuous = TrainingEngine(
        task=continuous_task,
        context=continuous_context,
        config=config,
        run_directory=tmp_path / "continuous",
        device=torch.device("cpu"),
        keep_recent_checkpoints=100,
    ).run()

    torch.manual_seed(20260719)
    staged_task = ToyTrainingTask(validation_values, stochastic=True)
    staged_context = _advanced_context(staged_task).model_copy(
        update={"run_id": "staged-cyclepatch-direct"}
    )
    paused = TrainingEngine(
        task=staged_task,
        context=staged_context,
        config=config,
        run_directory=tmp_path / "staged",
        device=torch.device("cpu"),
        keep_recent_checkpoints=100,
    ).run(epoch_limit=30)
    resumed_task = ToyTrainingTask(validation_values, stochastic=True)
    resumed = TrainingEngine(
        task=resumed_task,
        context=staged_context,
        config=config,
        run_directory=tmp_path / "staged",
        device=torch.device("cpu"),
        keep_recent_checkpoints=100,
    ).run(epoch_limit=90)

    assert paused.status is TrainingRunStatus.PAUSED_STAGE
    assert continuous.status is TrainingRunStatus.COMPLETED
    assert resumed.status is TrainingRunStatus.COMPLETED
    assert resumed.resumed_from_epoch == 30
    assert resumed.best_epoch == continuous.best_epoch
    assert resumed.best_metric == continuous.best_metric
    for name, expected in continuous_task.model.state_dict().items():
        assert torch.equal(expected, resumed_task.model.state_dict()[name])
    _assert_nested_state_equal(
        continuous_task.optimizer.state_dict(),
        resumed_task.optimizer.state_dict(),
    )
    _assert_nested_state_equal(
        continuous_task.scheduler.state_dict(),
        resumed_task.scheduler.state_dict(),
    )
    assert _stable_validation_rows(tmp_path / "continuous") == (
        _stable_validation_rows(tmp_path / "staged")
    )


def test_advanced_engine_rejects_checkpoint_with_v1_schema(tmp_path: Path) -> None:
    config = ModelTrainingConfig(
        name="cyclepatch_direct",
        max_epochs=2,
        validation_interval=1,
        early_stopping_patience=3,
    )
    first_task = ToyTrainingTask({1: 1.0, 2: 0.9})
    context = _advanced_context(first_task)
    TrainingEngine(
        task=first_task,
        context=context,
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
    ).run(epoch_limit=1)
    pointer = json.loads(
        (tmp_path / "checkpoints" / "last.json").read_text(encoding="utf-8")
    )
    manifest_path = (
        tmp_path / "checkpoints" / pointer["checkpoint"] / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "safe-training-checkpoint-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed_task = ToyTrainingTask({1: 1.0, 2: 0.9})
    with pytest.raises(ValueError):
        TrainingEngine(
            task=resumed_task,
            context=context,
            config=config,
            run_directory=tmp_path,
            device=torch.device("cpu"),
        ).run(epoch_limit=2)


def test_paused_advanced_run_rejects_changed_context(tmp_path: Path) -> None:
    config = ModelTrainingConfig(
        name="cyclepatch_direct",
        max_epochs=2,
        validation_interval=1,
        early_stopping_patience=3,
    )
    first_task = ToyTrainingTask({1: 1.0, 2: 0.9})
    context = _advanced_context(first_task)
    TrainingEngine(
        task=first_task,
        context=context,
        config=config,
        run_directory=tmp_path,
        device=torch.device("cpu"),
    ).run(epoch_limit=1)
    changed_candidate = "f" * 64
    resumed_task = ToyTrainingTask({1: 1.0, 2: 0.9})
    changed_context = context.model_copy(
        update={
            "candidate_config_sha256": changed_candidate,
            "model_architecture_sha256": model_architecture_sha256(
                resumed_task.model,
                changed_candidate,
            ),
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        TrainingEngine(
            task=resumed_task,
            context=changed_context,
            config=config,
            run_directory=tmp_path,
            device=torch.device("cpu"),
        ).run(epoch_limit=2)


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
    assert result.training_time_seconds > 0
    assert result.peak_gpu_memory_bytes == 0
    assert task.validated_epochs == [2, 4, 6]
    assert (tmp_path / "training_log.jsonl").is_file()
    assert (tmp_path / "metrics_epoch.csv").is_file()
    assert (tmp_path / "metrics_validation.csv").is_file()
    assert (tmp_path / "checkpoints" / "last.json").is_file()
    assert (tmp_path / "checkpoints" / "best.json").is_file()
    assert (tmp_path / "run_status.json").is_file()
    pointer = json.loads(
        (tmp_path / "checkpoints" / "last.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            tmp_path
            / "checkpoints"
            / pointer["checkpoint"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "safe-training-checkpoint-v1"
    retained = {
        path.name
        for path in (tmp_path / "checkpoints").glob("epoch-*")
        if path.is_dir()
    }
    assert retained == {
        "epoch-000002",
        "epoch-000004",
        "epoch-000005",
        "epoch-000006",
    }
    events = [
        json.loads(line)
        for line in (tmp_path / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["best_epoch"] == 2
    assert events[-1]["early_stop_counter"] == 2
    assert events[-1]["gpu_memory_allocated_bytes"] == 0
    assert events[-1]["gpu_memory_reserved_bytes"] == 0


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
    assert resumed.training_time_seconds >= first.training_time_seconds
    assert resumed.peak_gpu_memory_bytes == 0
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
