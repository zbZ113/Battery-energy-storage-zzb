from pathlib import Path

import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.classic import (
    curve_batch_to_tabular,
    fit_dummy_cycle_life,
    fit_variance_cycle_life,
    train_xgboost_cycle_life,
)
from quanxin_life.training.tasks import CycleLifeCurveBatch


def _batch(cell_ids: tuple[str, ...], labels: tuple[float, ...]) -> CycleLifeCurveBatch:
    rows = []
    for index, _cell_id in enumerate(cell_ids):
        scale = 1.0 - index * 0.05
        rows.append(
            [
                [1.0, 0.5, 0.0],
                [0.9 * scale, 0.45 * scale, 0.0],
            ]
        )
    return CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=cell_ids,
        curve_values=torch.tensor(rows, dtype=torch.float32),
        observed_mask=torch.ones((len(cell_ids), 2), dtype=torch.bool),
        observed_cycles=torch.tensor(labels, dtype=torch.float32),
        cutoff_cycle=20,
    )


def test_classic_baselines_and_xgboost_use_explicit_official_targets(tmp_path: Path) -> None:
    train = _batch(("a", "b", "c"), (100.0, 120.0, 140.0))
    validation = _batch(("d", "e"), (110.0, 130.0))
    test = _batch(("f", "g"), (115.0, 135.0))
    tabular = curve_batch_to_tabular(train)

    assert tabular.feature_names[0] == "curve_coverage"
    assert tabular.values.shape[0] == 3
    assert fit_dummy_cycle_life(train).predict(test).shape == (2,)
    assert fit_variance_cycle_life(train).predict(test).shape == (2,)

    result = train_xgboost_cycle_life(
        train_batch=train,
        validation_batch=validation,
        max_rounds=5,
        early_stopping_rounds=2,
        seed=20260712,
        device="cpu",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    prediction = result.predict(test)
    artifact = tmp_path / "model.ubj"
    result.save_model(artifact)

    assert prediction.shape == (2,)
    assert result.best_iteration < 5
    assert artifact.is_file()
