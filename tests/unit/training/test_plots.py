from pathlib import Path

import numpy as np
import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.plots import (
    write_cycle_life_evaluation_plot,
    write_hybrid_trajectory_plot,
)
from quanxin_life.training.tasks import CycleLifeCurveBatch, HybridTrajectoryBatch

pytest.importorskip("matplotlib")


def test_training_plots_are_written_from_explicit_model_outputs(tmp_path: Path) -> None:
    cycle_batch = CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=("test-a", "test-b"),
        curve_values=torch.ones((2, 2, 3)),
        observed_mask=torch.ones((2, 2), dtype=torch.bool),
        observed_cycles=torch.tensor([100.0, 140.0]),
        cutoff_cycle=20,
    )
    cycle_path = tmp_path / "cycle.png"
    write_cycle_life_evaluation_plot(
        cycle_path,
        batch=cycle_batch,
        predicted=np.asarray([110.0, 130.0]),
        interval_radius_cycle=15.0,
    )

    trajectory_batch = HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=("test-a", "test-b"),
        features=torch.ones((2, 3)),
        initial_soh=torch.tensor([0.98, 0.98]),
        target_soh=torch.tensor([[0.97, 0.95, 0.94], [0.96, 0.94, 0.93]]),
        prediction_cycles=(100, 300, 500),
        cutoff_cycle=20,
    )
    trajectory_path = tmp_path / "trajectory.png"
    write_hybrid_trajectory_plot(
        trajectory_path,
        batch=trajectory_batch,
        predicted=np.asarray(
            [[0.965, 0.945, 0.935], [0.955, 0.935, 0.925]]
        ),
    )

    assert cycle_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert trajectory_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
