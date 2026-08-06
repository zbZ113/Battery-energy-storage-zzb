from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quanxin_life.core import (
    LastBatchPolicy,
    TrainingTaskType,
    sha256_canonical,
)
from quanxin_life.data.model_views.matr import (
    build_tensor_model_view,
    tensor_partition_metadata,
)
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.training.adapters import native as native_adapter
from quanxin_life.training.adapters.native import (
    NativeTensorModelView,
    load_native_advanced_matr_selection_data,
    load_native_matr_legacy_cohorts,
    resolve_native_matr_view_roots,
)
from quanxin_life.training.batching import BatchPlan


def test_native_tensor_view_loads_verified_partition_without_processed_or_raw_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data" / "model_views" / "early_life_sequence" / "v1"
    tensors = {
        "train.values": torch.tensor([[1.0], [2.0]]),
        "train.sample_mask": torch.ones((2, 1), dtype=torch.bool),
        "train.target_cycle_life": torch.tensor([100.0, 120.0]),
    }
    metadata = {
        "schema_version": "matr-tensor-view-v1",
        "view_id": "early_life_sequence",
        "partitions": [
            tensor_partition_metadata(
                partition="train",
                cell_ids=("cell-a", "cell-b"),
                tensors=tensors,
            )
        ],
    }
    normalization = {
        "fitted_split": "train",
        "training_cell_ids_sha256": sha256_canonical(("cell-a", "cell-b")),
    }
    build_tensor_model_view(
        output_root=root,
        view_id="early_life_sequence",
        view_version="v1",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="matr_official_cycle_life",
        mask_semantics=("explicit_sample_mask",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256="b" * 64,
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
        normalization_sha256=sha256_canonical(normalization),
        training_entity_ids_sha256=sha256_canonical(("cell-a", "cell-b")),
        metadata=metadata,
        normalization=normalization,
        tensors=tensors,
    )
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        normalized = str(path).replace("\\", "/")
        if "/data/raw/" in normalized or "/data/processed/" in normalized:
            raise AssertionError("native training must read only frozen model views")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    partition = NativeTensorModelView(root).partition("train")

    assert partition.cell_ids == ("cell-a", "cell-b")
    assert len(partition) == 2
    assert partition[1]["values"].item() == 2.0
    assert partition[1]["target_cycle_life"].item() == 120.0

    loader = partition.loader(
        BatchPlan(
            micro_batch_size=1,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=2,
            sample_count=2,
            last_batch_policy=LastBatchPolicy.ERROR,
            optimizer_steps_per_epoch=1,
        )
    )
    batches = tuple(loader)

    assert [batch.cell_ids for batch in batches] == [("cell-a",), ("cell-b",)]
    assert [batch.tensors["values"].shape for batch in batches] == [(1, 1), (1, 1)]


def test_native_views_materialize_advanced_selection_batches(tmp_path: Path) -> None:
    split = SplitManifest(
        dataset_id="MATR",
        seed=38,
        train=("train-a",),
        validation=("validation-a",),
        calibration=("calibration-a",),
        test=("test-a",),
    )
    scalar_root = tmp_path / "early"
    trajectory_root = tmp_path / "trajectory"
    normalization = _normalization(("train-a",))
    scalar_tensors = _scalar_tensors()
    trajectory_tensors = _trajectory_tensors()
    _build_view(
        root=scalar_root,
        view_id="early_life_sequence",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="matr_official_cycle_life",
        normalization=normalization,
        tensors=scalar_tensors,
    )
    _build_view(
        root=trajectory_root,
        view_id="soh_trajectory",
        task_type=TrainingTaskType.SOH_TRAJECTORY,
        target_semantics="matr_observed_soh_to_cycle_500",
        normalization=normalization,
        tensors=trajectory_tensors,
    )

    data = load_native_advanced_matr_selection_data(
        scalar_view_root=scalar_root,
        trajectory_view_root=trajectory_root,
        combined_split=split,
        cutoff_cycle=50,
    )

    assert data.scalar_train.cell_ids == ("train-a",)
    assert data.scalar_validation.cell_ids == ("validation-a",)
    assert data.scalar_train.raw_labels.tolist() == [200.0]
    assert data.hybrid_train.cell_ids == ("train-a",)
    assert data.hybrid_train.inputs.prediction_cycles.tolist() == list(range(51, 501))
    assert data.scalar_normalizer.statistics_sha256 == normalization["statistics_sha256"]


def test_native_views_materialize_legacy_curve_and_hybrid_cohorts(tmp_path: Path) -> None:
    split = SplitManifest(
        dataset_id="MATR",
        seed=38,
        train=("train-a",),
        validation=("validation-a",),
        calibration=("calibration-a",),
        test=("test-a",),
    )
    scalar_root = tmp_path / "early"
    trajectory_root = tmp_path / "trajectory"
    normalization = _normalization(("train-a",))
    scalar_tensors = _scalar_tensors_all_partitions()
    trajectory_tensors = _trajectory_tensors_all_partitions()
    _build_view(
        root=scalar_root,
        view_id="early_life_sequence",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="matr_official_cycle_life",
        normalization=normalization,
        tensors=scalar_tensors,
        partitions=("train", "validation", "calibration", "test"),
    )
    _build_view(
        root=trajectory_root,
        view_id="soh_trajectory",
        task_type=TrainingTaskType.SOH_TRAJECTORY,
        target_semantics="matr_observed_soh_to_cycle_500",
        normalization=normalization,
        tensors=trajectory_tensors,
        partitions=("train", "validation", "calibration", "test"),
    )

    curves, hybrid = load_native_matr_legacy_cohorts(
        scalar_view_root=scalar_root,
        trajectory_view_root=trajectory_root,
        combined_split=split,
        cutoff_cycle=50,
    )

    assert curves.train.curve_values.shape == (1, 51, 150)
    assert curves.test.cell_ids == ("test-a",)
    assert hybrid.train.prediction_cycles[-1] == 500
    assert hybrid.test.target_soh.shape == (1, 450)


def test_native_trajectory_view_must_terminate_at_cycle_500(tmp_path: Path) -> None:
    split = SplitManifest(
        dataset_id="MATR",
        seed=38,
        train=("train-a",),
        validation=("validation-a",),
        calibration=("calibration-a",),
        test=("test-a",),
    )
    scalar_root = tmp_path / "early"
    trajectory_root = tmp_path / "trajectory"
    normalization = _normalization(("train-a",))
    _build_view(
        root=scalar_root,
        view_id="early_life_sequence",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="matr_official_cycle_life",
        normalization=normalization,
        tensors=_scalar_tensors(),
    )
    trajectory_tensors = _trajectory_tensors()
    for partition in ("train", "validation"):
        trajectory_tensors[f"{partition}.prediction_cycles"] = torch.arange(
            51, 500
        ).unsqueeze(0)
        trajectory_tensors[f"{partition}.target_soh"] = torch.ones((1, 449))
        trajectory_tensors[f"{partition}.target_mask"] = torch.ones(
            (1, 449), dtype=torch.bool
        )
    _build_view(
        root=trajectory_root,
        view_id="soh_trajectory",
        task_type=TrainingTaskType.SOH_TRAJECTORY,
        target_semantics="matr_observed_soh_to_cycle_500",
        normalization=normalization,
        tensors=trajectory_tensors,
    )

    with pytest.raises(ValueError, match="cycle 500"):
        load_native_advanced_matr_selection_data(
            scalar_view_root=scalar_root,
            trajectory_view_root=trajectory_root,
            combined_split=split,
            cutoff_cycle=50,
        )


def test_native_view_roots_resolve_one_frozen_pair_per_cutoff(tmp_path: Path) -> None:
    config_root = tmp_path / "configs" / "model_views" / "matr_cutoffs"
    config_root.mkdir(parents=True)
    tensors = {
        "train.values": torch.tensor([[1.0]]),
        "train.sample_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    normalization = {
        "fitted_split": "train",
        "training_cell_ids_sha256": sha256_canonical(("train-a",)),
    }
    for view_id, task_type, target_semantics in (
        (
            "early_life_sequence_c100",
            TrainingTaskType.CYCLE_LIFE,
            "matr_official_cycle_life",
        ),
        (
            "soh_trajectory_c100",
            TrainingTaskType.SOH_TRAJECTORY,
            "matr_observed_soh_to_cycle_500",
        ),
    ):
        version = f"{view_id}-v1"
        config = {
            "schema_version": "model-view-config-v1",
            "view_id": view_id,
            "view_version": version,
            "task_type": task_type.value,
            "target_semantics": target_semantics,
            "entity_key": "cell_id",
            "feature_names": ["values"],
            "mask_semantics": ["explicit_sample_mask"],
            "source_dataset_ids": ["MATR"],
            "cutoff_cycle": 100,
        }
        (config_root / f"{view_id}.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        build_tensor_model_view(
            output_root=tmp_path / "data" / "model_views" / view_id / version,
            view_id=view_id,
            view_version=version,
            task_type=task_type,
            target_semantics=target_semantics,
            mask_semantics=("explicit_sample_mask",),
            cutoff_cycle=100,
            canonical_sha256="a" * 64,
            split_sha256="b" * 64,
            builder_code_sha256="c" * 64,
            config_sha256="d" * 64,
            normalization_sha256=sha256_canonical(normalization),
            training_entity_ids_sha256=str(
                normalization["training_cell_ids_sha256"]
            ),
            metadata={
                "schema_version": "matr-tensor-view-v1",
                "view_id": view_id,
                "partitions": [
                    tensor_partition_metadata(
                        partition="train",
                        cell_ids=("train-a",),
                        tensors=tensors,
                    )
                ],
            },
            normalization=normalization,
            tensors=tensors,
        )

    roots = resolve_native_matr_view_roots(tmp_path, cutoff_cycle=100)

    assert roots.scalar_root.name == "early_life_sequence_c100-v1"
    assert roots.trajectory_root.name == "soh_trajectory_c100-v1"
    assert len(roots.model_view_sha256) == 64


def test_cutoff_specific_native_view_ids_are_accepted() -> None:
    scalar = SimpleNamespace(
        manifest=SimpleNamespace(
            view_id="early_life_sequence_c20",
            target_semantics="matr_official_cycle_life",
            cutoff_cycle=20,
        ),
        normalization=SimpleNamespace(cutoff_cycle=20),
    )
    trajectory = SimpleNamespace(
        manifest=SimpleNamespace(
            view_id="soh_trajectory_c20",
            target_semantics="matr_observed_soh_to_cycle_500",
            cutoff_cycle=20,
        ),
        normalization=SimpleNamespace(cutoff_cycle=20),
    )

    native_adapter._validate_matr_view_pair(
        scalar,
        trajectory,
        cutoff_cycle=20,
    )


def _build_view(
    *,
    root: Path,
    view_id: str,
    task_type: TrainingTaskType,
    target_semantics: str,
    normalization: dict[str, object],
    tensors: dict[str, torch.Tensor],
    partitions: tuple[str, ...] = ("train", "validation"),
) -> None:
    partition_metadata = []
    for partition in partitions:
        partition_metadata.append(
            tensor_partition_metadata(
                partition=partition,
                cell_ids=(f"{partition}-a",),
                tensors=tensors,
            )
        )
    build_tensor_model_view(
        output_root=root,
        view_id=view_id,
        view_version="v2",
        task_type=task_type,
        target_semantics=target_semantics,
        mask_semantics=("explicit_sample_mask",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256="b" * 64,
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
        normalization_sha256=sha256_canonical(normalization),
        training_entity_ids_sha256=str(normalization["training_cell_ids_sha256"]),
        metadata={
            "schema_version": "matr-tensor-view-v1",
            "view_id": view_id,
            "partitions": partition_metadata,
        },
        normalization=normalization,
        tensors=tensors,
    )


def _normalization(training_cells: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": "MATR",
        "data_version": "matr-test-v1",
        "feature_version": "cyclepatch-multichannel-v1",
        "cutoff_cycle": 50,
        "cycle_indices": list(range(51)),
        "sample_count": 150,
        "condition_names": ["temperature_c"],
        "variable_means": [0.0, 0.0, 0.0],
        "variable_stds": [1.0, 1.0, 1.0],
        "condition_means": [0.0],
        "condition_stds": [1.0],
        "training_cell_ids_sha256": sha256_canonical(sorted(training_cells)),
    }
    payload["statistics_sha256"] = sha256_canonical(
        {
            "schema_version": "early-cycle-normalizer-v1",
            **payload,
            "phase_names": ("charge", "discharge"),
            "variable_names": ("voltage_v", "current_a", "capacity_ah"),
        }
    )
    payload["fitted_split"] = "train"
    return payload


def _early_partition_tensors(partition: str) -> dict[str, torch.Tensor]:
    values = torch.zeros((1, 51, 2, 150, 3), dtype=torch.float32)
    sample_mask = torch.ones((1, 51, 2, 150), dtype=torch.bool)
    return {
        f"{partition}.values": values,
        f"{partition}.cycle_indices": torch.arange(51).unsqueeze(0),
        f"{partition}.cycle_mask": torch.ones((1, 51), dtype=torch.bool),
        f"{partition}.sample_mask": sample_mask,
        f"{partition}.condition_values": torch.zeros((1, 1)),
        f"{partition}.condition_mask": torch.ones((1, 1), dtype=torch.bool),
    }


def _scalar_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for partition in ("train", "validation"):
        tensors.update(_early_partition_tensors(partition))
        tensors[f"{partition}.target_cycle_life"] = torch.tensor([200.0])
    return tensors


def _trajectory_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for partition in ("train", "validation"):
        tensors.update(_early_partition_tensors(partition))
        tensors[f"{partition}.initial_soh"] = torch.tensor([1.0])
        tensors[f"{partition}.history_soh"] = torch.ones((1, 51))
        tensors[f"{partition}.history_mask"] = torch.ones((1, 51), dtype=torch.bool)
        tensors[f"{partition}.target_soh"] = torch.ones((1, 450))
        tensors[f"{partition}.target_mask"] = torch.ones((1, 450), dtype=torch.bool)
        tensors[f"{partition}.prediction_cycles"] = torch.arange(51, 501).unsqueeze(0)
    return tensors


def _scalar_tensors_all_partitions() -> dict[str, torch.Tensor]:
    tensors = _scalar_tensors()
    for partition in ("calibration", "test"):
        tensors.update(_early_partition_tensors(partition))
        tensors[f"{partition}.target_cycle_life"] = torch.tensor([200.0])
    return tensors


def _trajectory_tensors_all_partitions() -> dict[str, torch.Tensor]:
    tensors = _trajectory_tensors()
    for partition in ("calibration", "test"):
        tensors.update(_early_partition_tensors(partition))
        tensors[f"{partition}.initial_soh"] = torch.tensor([1.0])
        tensors[f"{partition}.history_soh"] = torch.ones((1, 51))
        tensors[f"{partition}.history_mask"] = torch.ones((1, 51), dtype=torch.bool)
        tensors[f"{partition}.target_soh"] = torch.ones((1, 450))
        tensors[f"{partition}.target_mask"] = torch.ones((1, 450), dtype=torch.bool)
        tensors[f"{partition}.prediction_cycles"] = torch.arange(51, 501).unsqueeze(0)
    return tensors
