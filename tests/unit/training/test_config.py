from __future__ import annotations

import pytest
from pydantic import ValidationError

from quanxin_life.core import PredictionTarget
from quanxin_life.training.config import (
    CheckpointPolicy,
    LoggingPolicy,
    ModelTrainingConfig,
    TrainingSuiteConfig,
)


def test_matr_suite_binds_official_target_and_model_specific_epochs() -> None:
    config = TrainingSuiteConfig(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cutoffs=(20, 50, 100, 150),
        seeds=(20260712, 20260713),
        models=(
            ModelTrainingConfig(name="cpmlp", max_epochs=300),
            ModelTrainingConfig(name="hybrid", max_epochs=500),
        ),
        physical_gpu_index=1,
        batch_size=16,
        data_version="matr-2018-04-12-v1",
        split_version="matr-cell-split-v1",
        feature_version="early-cycle-v1",
        checkpoint=CheckpointPolicy(),
        logging=LoggingPolicy(),
    )

    assert config.physical_gpu_index == 1
    assert config.models[0].validation_interval == 5
    assert config.models[0].early_stopping_patience == 10
    assert len(config.config_sha256) == 64


def test_matr_suite_rejects_eol80_target_relabelling() -> None:
    with pytest.raises(ValidationError, match="official cycle-life"):
        TrainingSuiteConfig(
            dataset_id="MATR",
            target=PredictionTarget.UNIFIED_EOL80_CYCLE,
            cutoffs=(50,),
            seeds=(20260712,),
            models=(ModelTrainingConfig(name="cpmlp", max_epochs=2),),
            physical_gpu_index=1,
            batch_size=4,
            data_version="matr-v1",
            split_version="split-v1",
            feature_version="feature-v1",
        )


def test_training_suite_rejects_duplicate_cutoffs_models_and_seeds() -> None:
    with pytest.raises(ValidationError):
        TrainingSuiteConfig(
            dataset_id="HUST",
            target=PredictionTarget.UNIFIED_EOL80_CYCLE,
            cutoffs=(50, 50),
            seeds=(1, 1),
            models=(
                ModelTrainingConfig(name="cpmlp", max_epochs=2),
                ModelTrainingConfig(name="cpmlp", max_epochs=2),
            ),
            physical_gpu_index=1,
            batch_size=4,
            data_version="hust-v1",
            split_version="split-v1",
            feature_version="feature-v1",
        )
