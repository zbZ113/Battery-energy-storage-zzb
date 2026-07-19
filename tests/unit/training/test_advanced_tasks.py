from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
)
from quanxin_life.models.cyclepatch import CyclePatchConfig, EarlyCycleBatch
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Inputs,
    HybridPatchV2Targets,
)
from quanxin_life.training.advanced_tasks import (
    AdvancedCycleLifeBatch,
    AdvancedTrajectoryBatch,
    CyclePatchBatLiNetTrainingTask,
    CyclePatchDirectTrainingTask,
    HybridPatchV2TrainingTask,
)


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("train-a", "train-censored", "train-b"),
        validation=("validation-censored", "validation-a", "validation-b"),
        calibration=("calibration-a",),
        test=("test-a",),
    )


def _supervised_split(
    source: SplitManifest,
    *,
    train: tuple[str, ...] = ("train-a", "train-b"),
    validation: tuple[str, ...] = ("validation-a", "validation-b"),
) -> SplitManifest:
    return SplitManifest(
        dataset_id=source.dataset_id,
        seed=source.seed,
        train=train,
        validation=validation,
        calibration=(),
        test=(),
    )


def _cyclepatch_config() -> CyclePatchConfig:
    return CyclePatchConfig(d_model=12, layers=1, heads=3, dropout=0.0)


def _early_batch(cell_ids: tuple[str, ...], *, cutoff: int = 2) -> EarlyCycleBatch:
    batch_size = len(cell_ids)
    values = torch.full((batch_size, cutoff + 1, 2, 150, 3), torch.nan)
    sample_mask = torch.zeros((batch_size, cutoff + 1, 2, 150), dtype=torch.bool)
    cycle_mask = torch.zeros((batch_size, cutoff + 1), dtype=torch.bool)
    for row in range(batch_size):
        for cycle in range(1, cutoff + 1):
            base = 0.1 * (row + 1) + 0.01 * cycle
            values[row, cycle, :, :, 0] = torch.linspace(3.0, 4.1, 150)
            values[row, cycle, :, :, 1] = 1.0 + base
            values[row, cycle, :, :, 2] = torch.linspace(0.0, 1.0 + base, 150)
            sample_mask[row, cycle] = True
            cycle_mask[row, cycle] = True
    return EarlyCycleBatch(
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-v1",
        normalization_statistics_sha256="a" * 64,
        cell_ids=cell_ids,
        condition_names=("temperature_c",),
        values=values,
        cycle_indices=torch.arange(cutoff + 1, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1),
        cycle_mask=cycle_mask,
        sample_mask=sample_mask,
        condition_values=torch.full((batch_size, 1), 25.0),
        condition_mask=torch.ones((batch_size, 1), dtype=torch.bool),
    )


def _cycle_life_batches() -> tuple[AdvancedCycleLifeBatch, AdvancedCycleLifeBatch]:
    return (
        AdvancedCycleLifeBatch(
            early_batch=_early_batch(("train-a", "train-b")),
            raw_labels=torch.tensor([100.0, 200.0]),
        ),
        AdvancedCycleLifeBatch(
            early_batch=_early_batch(("validation-a", "validation-b")),
            raw_labels=torch.tensor([300.0, 400.0]),
        ),
    )


def _trajectory_batch(cell_ids: tuple[str, ...]) -> AdvancedTrajectoryBatch:
    early = _early_batch(cell_ids)
    batch_size = len(cell_ids)
    prediction_cycles = torch.tensor([3, 8, 500], dtype=torch.int64)
    initial_soh = torch.full((batch_size,), 0.98)
    history_soh = torch.full((batch_size, 3), torch.nan)
    history_soh[:, 1] = 0.99
    history_soh[:, 2] = 0.98
    target_soh = torch.tensor(
        [[0.97, 0.94, 0.80], [0.965, 0.93, 0.78]], dtype=torch.float32
    )[:batch_size]
    return AdvancedTrajectoryBatch(
        inputs=HybridPatchV2Inputs(
            early_batch=early,
            initial_soh=initial_soh,
            prediction_cycles=prediction_cycles,
        ),
        targets=HybridPatchV2Targets(
            history_soh=history_soh,
            history_mask=early.cycle_mask.clone(),
            target_soh=target_soh,
            target_mask=torch.ones_like(target_soh, dtype=torch.bool),
        ),
    )


def _parameters(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _changed(before: tuple[torch.Tensor, ...], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters(), strict=True)
    )


def _same_state(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return all(
        torch.equal(left_value, right.state_dict()[name])
        for name, left_value in left.state_dict().items()
    )


def test_advanced_cycle_life_batch_rejects_misaligned_or_invalid_labels() -> None:
    early = _early_batch(("train-a", "train-b"))
    with pytest.raises(ValueError, match="align"):
        AdvancedCycleLifeBatch(early_batch=early, raw_labels=torch.tensor([100.0]))
    with pytest.raises(ValueError, match="greater than cutoff"):
        AdvancedCycleLifeBatch(
            early_batch=early,
            raw_labels=torch.tensor([2.0, 100.0]),
        )


def test_direct_task_rejects_train_validation_overlap() -> None:
    train, _validation = _cycle_life_batches()
    overlap = AdvancedCycleLifeBatch(
        early_batch=_early_batch(("train-a", "validation-b")),
        raw_labels=torch.tensor([100.0, 400.0]),
    )
    with pytest.raises(ValueError, match="disjoint"):
        CyclePatchDirectTrainingTask(
            train_batch=train,
            validation_batch=overlap,
            split_manifest=_split(),
            config=_cyclepatch_config(),
            learning_rate=1e-3,
            seed=38,
        )


def test_direct_task_trains_validates_raw_units_and_caches_device_batches() -> None:
    train, validation = _cycle_life_batches()
    task = CyclePatchDirectTrainingTask(
        train_batch=train,
        validation_batch=validation,
        split_manifest=_split(),
        config=_cyclepatch_config(),
        learning_rate=1e-2,
        seed=38,
    )
    assert task.source_split_manifest == _split()
    assert task.source_split_sha256 == sha256_canonical(
        task.source_split_manifest.model_dump(mode="json")
    )
    assert task.supervised_split_manifest == _supervised_split(_split())
    assert task.supervised_split_sha256 == sha256_canonical(
        task.supervised_split_manifest.model_dump(mode="json")
    )
    assert task.target_scaler.training_cell_ids_sha256 == sha256_canonical(
        sorted(task.supervised_split_manifest.train)
    )
    before = _parameters(task.model)
    task.train_epoch(1, device=torch.device("cpu"))
    assert _changed(before, task.model)

    cached = task._batches_for_device(torch.device("cpu"))
    assert cached is task._batches_for_device(torch.device("cpu"))
    before_validation = _parameters(task.model)
    metrics = task.validate(1, device=torch.device("cpu"))
    assert not _changed(before_validation, task.model)
    assert metrics.metrics.keys() == {"mae", "rmse", "mape", "r2"}
    assert metrics.metrics["mae"] > 1.0


def test_direct_task_seed_controls_initial_model_state() -> None:
    train, validation = _cycle_life_batches()
    common = dict(
        train_batch=train,
        validation_batch=validation,
        split_manifest=_split(),
        config=_cyclepatch_config(),
        learning_rate=1e-3,
    )
    first = CyclePatchDirectTrainingTask(**common, seed=38)
    second = CyclePatchDirectTrainingTask(**common, seed=38)
    different = CyclePatchDirectTrainingTask(**common, seed=39)
    assert _same_state(first.model, second.model)
    assert not _same_state(first.model, different.model)


@pytest.mark.parametrize(
    ("train_cells", "validation_cells", "message"),
    [
        (("train-a", "test-a"), ("validation-a", "validation-b"), "source train"),
        (("train-b", "train-a"), ("validation-a", "validation-b"), "source train"),
        (("train-a", "train-b"), ("validation-a", "calibration-a"), "source validation"),
        (("train-a", "train-b"), ("validation-b", "validation-a"), "source validation"),
    ],
)
def test_direct_task_rejects_supervision_outside_source_partition_or_order(
    train_cells: tuple[str, ...],
    validation_cells: tuple[str, ...],
    message: str,
) -> None:
    train = AdvancedCycleLifeBatch(
        early_batch=_early_batch(train_cells),
        raw_labels=torch.tensor([100.0, 200.0]),
    )
    validation = AdvancedCycleLifeBatch(
        early_batch=_early_batch(validation_cells),
        raw_labels=torch.tensor([300.0, 400.0]),
    )

    with pytest.raises(ValueError, match=message):
        CyclePatchDirectTrainingTask(
            train_batch=train,
            validation_batch=validation,
            split_manifest=_split(),
            config=_cyclepatch_config(),
            learning_rate=1e-3,
            seed=38,
        )


def _batlinet_context(
    *, reference_seed: int = 38
) -> tuple[
    AdvancedCycleLifeBatch,
    AdvancedCycleLifeBatch,
    CycleLifeTargetScaler,
    CycleLifeReferenceLibrary,
]:
    train, validation = _cycle_life_batches()
    split = _supervised_split(_split())
    labels = dict(zip(train.cell_ids, (100.0, 200.0), strict=True))
    scaler = CycleLifeTargetScaler.fit(
        labels,
        training_cell_ids=train.cell_ids,
        split_manifest=split,
        cutoff_cycle=train.cutoff_cycle,
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    library = CycleLifeReferenceLibrary.build(
        labels,
        training_cell_ids=train.cell_ids,
        split_manifest=split,
        scaler=scaler,
        reference_count=2,
        seed=reference_seed,
    )
    return train, validation, scaler, library


def test_batlinet_task_uses_only_frozen_training_references_and_trains() -> None:
    train, validation, scaler, library = _batlinet_context()
    task = CyclePatchBatLiNetTrainingTask(
        train_batch=train,
        validation_batch=validation,
        split_manifest=_split(),
        target_scaler=scaler,
        reference_library=library,
        config=BatLiNetConfig(
            encoder=_cyclepatch_config(),
            reference_count=2,
        ),
        learning_rate=1e-2,
        seed=38,
    )
    assert task.source_split_manifest == _split()
    assert task.source_split_sha256 == sha256_canonical(
        task.source_split_manifest.model_dump(mode="json")
    )
    assert task.supervised_split_manifest == _supervised_split(_split())
    assert task.supervised_split_sha256 == sha256_canonical(
        task.supervised_split_manifest.model_dump(mode="json")
    )
    assert task.target_scaler.training_cell_ids_sha256 == sha256_canonical(
        sorted(task.supervised_split_manifest.train)
    )
    assert task.reference_library.training_cell_ids_sha256 == (
        task.target_scaler.training_cell_ids_sha256
    )
    assert set(task.reference_library.cell_ids) <= set(_split().train)
    assert set(task.reference_library.cell_ids).isdisjoint(_split().validation)
    before = _parameters(task.model)
    task.train_epoch(1, device=torch.device("cpu"))
    assert _changed(before, task.model)
    before_validation = _parameters(task.model)
    metrics = task.validate(1, device=torch.device("cpu"))
    assert not _changed(before_validation, task.model)
    assert metrics.metrics["mae"] > 1.0


def test_batlinet_task_rejects_a_heldout_reference_library() -> None:
    train, validation, scaler, _library = _batlinet_context()
    alternate_split = SplitManifest(
        dataset_id="MATR",
        train=("validation-a", "validation-b"),
        validation=("train-a", "train-b"),
        calibration=("calibration-a",),
        test=("test-a",),
    )
    heldout_labels = {"validation-a": 300.0, "validation-b": 400.0}
    heldout_scaler = CycleLifeTargetScaler.fit(
        heldout_labels,
        training_cell_ids=alternate_split.train,
        split_manifest=alternate_split,
        cutoff_cycle=train.cutoff_cycle,
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    heldout_library = CycleLifeReferenceLibrary.build(
        heldout_labels,
        training_cell_ids=alternate_split.train,
        split_manifest=alternate_split,
        scaler=heldout_scaler,
        reference_count=2,
        seed=38,
    )
    with pytest.raises(ValueError, match=r"reference.*training"):
        CyclePatchBatLiNetTrainingTask(
            train_batch=train,
            validation_batch=validation,
            split_manifest=_split(),
            target_scaler=scaler,
            reference_library=heldout_library,
            config=BatLiNetConfig(
                encoder=_cyclepatch_config(), reference_count=2
            ),
            learning_rate=1e-3,
            seed=38,
        )


def test_batlinet_task_rejects_reference_library_from_a_different_seed() -> None:
    train, validation, scaler, library = _batlinet_context(reference_seed=39)

    with pytest.raises(ValueError, match="reference library seed"):
        CyclePatchBatLiNetTrainingTask(
            train_batch=train,
            validation_batch=validation,
            split_manifest=_split(),
            target_scaler=scaler,
            reference_library=library,
            config=BatLiNetConfig(
                encoder=_cyclepatch_config(), reference_count=2
            ),
            learning_rate=1e-3,
            seed=38,
        )


def test_hybrid_task_rejects_overlap_and_mismatched_prediction_axis() -> None:
    train = _trajectory_batch(("train-a", "train-b"))
    overlap = _trajectory_batch(("train-a", "validation-b"))
    with pytest.raises(ValueError, match="disjoint"):
        HybridPatchV2TrainingTask(
            train_batch=train,
            validation_batch=overlap,
            split_manifest=_split(),
            config=HybridPatchV2Config(cyclepatch=_cyclepatch_config()),
            learning_rate=1e-3,
            seed=38,
        )

    validation = _trajectory_batch(("validation-a", "validation-b"))
    changed_axis = AdvancedTrajectoryBatch(
        inputs=replace(
            validation.inputs,
            prediction_cycles=torch.tensor([3, 9, 500], dtype=torch.int64),
        ),
        targets=validation.targets,
    )
    with pytest.raises(ValueError, match="prediction cycles"):
        HybridPatchV2TrainingTask(
            train_batch=train,
            validation_batch=changed_axis,
            split_manifest=_split(),
            config=HybridPatchV2Config(cyclepatch=_cyclepatch_config()),
            learning_rate=1e-3,
            seed=38,
        )


def test_hybrid_task_trains_and_validates_monotone_trajectory() -> None:
    task = HybridPatchV2TrainingTask(
        train_batch=_trajectory_batch(("train-a", "train-b")),
        validation_batch=_trajectory_batch(("validation-a", "validation-b")),
        split_manifest=_split(),
        config=HybridPatchV2Config(
            cyclepatch=_cyclepatch_config(),
            query_token_count=0,
            decoder_hidden_dim=8,
        ),
        learning_rate=1e-2,
        seed=38,
    )
    assert task.source_split_manifest == _split()
    assert task.source_split_sha256 == sha256_canonical(
        task.source_split_manifest.model_dump(mode="json")
    )
    assert task.supervised_split_manifest == _supervised_split(_split())
    assert task.supervised_split_sha256 == sha256_canonical(
        task.supervised_split_manifest.model_dump(mode="json")
    )
    before = _parameters(task.model)
    train_metrics = task.train_epoch(1, device=torch.device("cpu"))
    assert _changed(before, task.model)
    assert {"trajectory", "history", "smooth", "order", "residual"} <= set(
        train_metrics.metrics
    )
    before_validation = _parameters(task.model)
    validation = task.validate(1, device=torch.device("cpu"))
    assert not _changed(before_validation, task.model)
    assert validation.metrics["monotonic_violation_rate"] == 0.0
    assert validation.metrics["mae"] >= 0.0
    assert validation.metrics["rmse"] >= validation.metrics["mae"]


@pytest.mark.parametrize(
    ("train_cells", "validation_cells", "message"),
    [
        (("train-a", "calibration-a"), ("validation-a",), "source train"),
        (("train-b", "train-a"), ("validation-a",), "source train"),
        (("train-a",), ("validation-a", "test-a"), "source validation"),
        (("train-a",), ("validation-b", "validation-a"), "source validation"),
    ],
)
def test_hybrid_task_rejects_supervision_outside_source_partition_or_order(
    train_cells: tuple[str, ...],
    validation_cells: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HybridPatchV2TrainingTask(
            train_batch=_trajectory_batch(train_cells),
            validation_batch=_trajectory_batch(validation_cells),
            split_manifest=_split(),
            config=HybridPatchV2Config(cyclepatch=_cyclepatch_config()),
            learning_rate=1e-3,
            seed=38,
        )
