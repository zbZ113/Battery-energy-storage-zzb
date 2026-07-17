import json
from pathlib import Path

import numpy as np
import pytest
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


def test_xgboost_resumes_hash_bound_ubj_checkpoint(tmp_path: Path) -> None:
    train = _batch(("a", "b", "c"), (100.0, 120.0, 140.0))
    validation = _batch(("d", "e"), (110.0, 130.0))
    test = _batch(("f", "g"), (115.0, 135.0))
    checkpoint_directory = tmp_path / "resumed"

    with pytest.raises(RuntimeError, match="synthetic XGBoost interruption"):
        train_xgboost_cycle_life(
            train_batch=train,
            validation_batch=validation,
            max_rounds=8,
            early_stopping_rounds=20,
            seed=20260712,
            device="cpu",
            checkpoint_directory=checkpoint_directory,
            checkpoint_interval=2,
            _interrupt_after_round=4,
        )

    resumed = train_xgboost_cycle_life(
        train_batch=train,
        validation_batch=validation,
        max_rounds=8,
        early_stopping_rounds=20,
        seed=20260712,
        device="cpu",
        checkpoint_directory=checkpoint_directory,
        checkpoint_interval=2,
    )
    continuous = train_xgboost_cycle_life(
        train_batch=train,
        validation_batch=validation,
        max_rounds=8,
        early_stopping_rounds=20,
        seed=20260712,
        device="cpu",
        checkpoint_directory=tmp_path / "continuous",
        checkpoint_interval=2,
    )

    assert resumed.resumed_from_round == 4
    assert resumed.training_time_seconds > 0
    assert np.allclose(resumed.predict(test), continuous.predict(test), atol=1e-6)
    assert (tmp_path / "training_log.jsonl").is_file()
    assert (tmp_path / "metrics_epoch.csv").is_file()
    assert (tmp_path / "metrics_validation.csv").is_file()
    assert len((tmp_path / "training_log.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_xgboost_rejects_tampered_resume_checkpoint(tmp_path: Path) -> None:
    train = _batch(("a", "b", "c"), (100.0, 120.0, 140.0))
    validation = _batch(("d", "e"), (110.0, 130.0))
    checkpoint_directory = tmp_path / "checkpoints"
    train_xgboost_cycle_life(
        train_batch=train,
        validation_batch=validation,
        max_rounds=5,
        early_stopping_rounds=20,
        seed=20260712,
        device="cpu",
        checkpoint_directory=checkpoint_directory,
        checkpoint_interval=1,
    )
    pointer = checkpoint_directory / "last.json"
    checkpoint_name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    state = checkpoint_directory / checkpoint_name / "state.json"
    state.write_bytes(state.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=r"size|SHA-256"):
        train_xgboost_cycle_life(
            train_batch=train,
            validation_batch=validation,
            max_rounds=5,
            early_stopping_rounds=20,
            seed=20260712,
            device="cpu",
            checkpoint_directory=checkpoint_directory,
            checkpoint_interval=1,
        )
