from __future__ import annotations

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.tasks import (
    CPMLPTrainingTask,
    CycleLifeCurveBatch,
    HybridTrajectoryBatch,
    HybridTrajectoryTrainingTask,
)


def _curve_batch(cell_ids: tuple[str, ...], *, scale: float) -> CycleLifeCurveBatch:
    values = torch.tensor(
        [
            [
                [0.8 * scale, 0.4 * scale, 0.1 * scale],
                [0.7 * scale, 0.3 * scale, 0.05 * scale],
            ]
            for _ in cell_ids
        ],
        dtype=torch.float32,
    )
    return CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=cell_ids,
        curve_values=values,
        observed_mask=torch.ones((len(cell_ids), 2), dtype=torch.bool),
        observed_cycles=torch.tensor(
            [100.0 + index * 10 for index in range(len(cell_ids))],
            dtype=torch.float32,
        ),
        cutoff_cycle=20,
    )


def test_cpmlp_task_trains_and_validates_official_cycle_life_on_cpu() -> None:
    task = CPMLPTrainingTask(
        train_batch=_curve_batch(("train-a", "train-b"), scale=1.0),
        validation_batch=_curve_batch(("validation-a", "validation-b"), scale=0.9),
        curve_hidden_dim=4,
        aggregation_hidden_dim=4,
        learning_rate=0.001,
    )

    trained = task.train_epoch(1, device=torch.device("cpu"))
    validated = task.validate(1, device=torch.device("cpu"))

    assert trained.loss >= 0
    assert set(validated.metrics) == {"mae", "mape", "r2", "rmse"}
    assert validated.metrics["mae"] >= 0


def test_cpmlp_task_rejects_cell_overlap_or_wrong_target() -> None:
    train = _curve_batch(("shared", "train-b"), scale=1.0)
    validation = _curve_batch(("shared", "validation-b"), scale=0.9)
    with pytest.raises(ValueError, match="cell-disjoint"):
        CPMLPTrainingTask(
            train_batch=train,
            validation_batch=validation,
            curve_hidden_dim=4,
            aggregation_hidden_dim=4,
            learning_rate=0.001,
        )

    wrong_target = train.__class__(
        **{
            **train.__dict__,
            "target": PredictionTarget.UNIFIED_EOL80_CYCLE,
        }
    )
    with pytest.raises(ValueError, match="official cycle-life"):
        CPMLPTrainingTask(
            train_batch=wrong_target,
            validation_batch=_curve_batch(("validation-a",), scale=0.9),
            curve_hidden_dim=4,
            aggregation_hidden_dim=4,
            learning_rate=0.001,
        )


def test_cycle_life_batch_rejects_event_at_the_observation_cutoff() -> None:
    with pytest.raises(ValueError, match="after the cutoff"):
        CycleLifeCurveBatch(
            dataset_id="MATR",
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            cell_ids=("already-observed",),
            curve_values=torch.tensor([[[0.8], [0.7]]], dtype=torch.float32),
            observed_mask=torch.ones((1, 2), dtype=torch.bool),
            observed_cycles=torch.tensor([20.0], dtype=torch.float32),
            cutoff_cycle=20,
        )


def _trajectory_batch(cell_ids: tuple[str, ...]) -> HybridTrajectoryBatch:
    return HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=cell_ids,
        features=torch.tensor([[1.0, 0.01, 25.0] for _ in cell_ids]),
        initial_soh=torch.tensor([0.98 for _ in cell_ids]),
        target_soh=torch.tensor(
            [[0.97, 0.95, 0.93, 0.91] for _ in cell_ids], dtype=torch.float32
        ),
        prediction_cycles=(50, 100, 200, 500),
        cutoff_cycle=20,
    )


def test_hybrid_task_uses_real_trajectory_validation_metrics() -> None:
    task = HybridTrajectoryTrainingTask(
        train_batch=_trajectory_batch(("train-a", "train-b")),
        validation_batch=_trajectory_batch(("validation-a",)),
        hidden_dim=4,
        learning_rate=0.001,
    )

    trained = task.train_epoch(1, device=torch.device("cpu"))
    validated = task.validate(1, device=torch.device("cpu"))
    test_batch = _trajectory_batch(("test-a", "test-b"))
    tested = task.evaluate(test_batch, device=torch.device("cpu"))
    predicted = task.predict(test_batch, device=torch.device("cpu"))

    assert trained.loss >= 0
    assert set(validated.metrics) == {"mae", "monotonic_violation_rate", "rmse"}
    assert validated.metrics["monotonic_violation_rate"] == 0.0
    assert tested.metrics["mae"] >= 0
    assert predicted.shape == test_batch.target_soh.shape
