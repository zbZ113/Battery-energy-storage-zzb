"""Frozen safetensors model views built from verified MATR processed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
from safetensors.torch import save_file

from quanxin_life.core import DatasetBuildStatus, TrainingTaskType, sha256_canonical
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest
from quanxin_life.data.model_views.builder import verify_model_view
from quanxin_life.data.model_views.schemas import (
    ModelViewArtifact,
    ModelViewBuildResult,
    ModelViewManifest,
)
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.training.advanced_data import (
    AdvancedFinalMatrData,
    load_advanced_matr_final_data,
)

_BUILDER_VERSION = "matr-tensor-model-view-v1"
_PARTITIONS = ("train", "validation", "calibration", "test")


def load_verified_matr_view_data(
    *,
    project_root: Path,
    manifest: MatrThreeBatchManifest,
    combined_split: SplitManifest,
    cutoff_cycle: int,
    feature_version: str,
) -> AdvancedFinalMatrData:
    """Reuse the governed processed-only loader without accessing raw inputs."""

    return load_advanced_matr_final_data(
        project_root=project_root,
        manifest=manifest,
        combined_split=combined_split,
        cutoff_cycle=cutoff_cycle,
        feature_version=feature_version,
    )


def tensor_partition_metadata(
    *,
    partition: str,
    cell_ids: tuple[str, ...],
    tensors: dict[str, torch.Tensor],
) -> dict[str, object]:
    if partition not in _PARTITIONS:
        raise ValueError("tensor model view partition is not supported")
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("tensor model view cell IDs must be non-empty and unique")
    prefix = f"{partition}."
    selected = {key: value for key, value in tensors.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError("tensor model view partition has no tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in selected.values()):
        raise ValueError("tensor model view artifacts must contain tensors")
    if any(tensor.ndim == 0 or tensor.shape[0] != len(cell_ids) for tensor in selected.values()):
        raise ValueError("tensor model view tensors must align with partition cells")
    return {
        "partition": partition,
        "cell_ids": list(cell_ids),
        "tensor_shapes": {
            key: list(tensor.shape) for key, tensor in sorted(selected.items())
        },
        "tensor_dtypes": {
            key: str(tensor.dtype).removeprefix("torch.")
            for key, tensor in sorted(selected.items())
        },
    }


def build_tensor_model_view(
    *,
    output_root: Path,
    view_id: str,
    view_version: str,
    task_type: TrainingTaskType,
    target_semantics: str,
    mask_semantics: tuple[str, ...],
    cutoff_cycle: int,
    canonical_sha256: str,
    split_sha256: str,
    builder_code_sha256: str,
    config_sha256: str,
    normalization_sha256: str,
    training_entity_ids_sha256: str,
    metadata: dict[str, object],
    normalization: dict[str, object],
    tensors: dict[str, torch.Tensor],
) -> ModelViewBuildResult:
    if sha256_canonical(normalization) != normalization_sha256:
        raise ValueError("normalization SHA does not match train-only statistics")
    root = Path(output_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "normalization.json", normalization)
        save_file(
            {key: tensor.detach().cpu().contiguous() for key, tensor in tensors.items()},
            staging / "tensors.safetensors",
        )
        artifacts = tuple(
            ModelViewArtifact(
                relative_path=name,
                size_bytes=(staging / name).stat().st_size,
                sha256=_sha256_file(staging / name),
            )
            for name in ("metadata.json", "normalization.json", "tensors.safetensors")
        )
        raw_partitions = metadata.get("partitions")
        if not isinstance(raw_partitions, list):
            raise ValueError("tensor model view metadata requires partitions")
        row_count = 0
        for partition in raw_partitions:
            if not isinstance(partition, dict):
                raise ValueError("tensor model view partition metadata must be an object")
            cell_ids = partition.get("cell_ids")
            if not isinstance(cell_ids, list):
                raise ValueError("tensor model view partition requires cell_ids")
            row_count += len(cell_ids)
        manifest = ModelViewManifest(
            schema_version="model-view-manifest-v2",
            view_id=view_id,
            view_version=view_version,
            task_type=task_type,
            target_semantics=target_semantics,
            entity_key="cell_id",
            source_dataset_ids=("MATR",),
            mask_semantics=mask_semantics,
            cutoff_cycle=cutoff_cycle,
            canonical_sha256=canonical_sha256,
            split_sha256=split_sha256,
            builder_version=_BUILDER_VERSION,
            builder_code_sha256=builder_code_sha256,
            config_sha256=config_sha256,
            normalization_sha256=_sha256_file(staging / "normalization.json"),
            training_entity_ids_sha256=training_entity_ids_sha256,
            row_count=row_count,
            artifacts=artifacts,
        )
        _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        digest = _commit_digest(staging)
        (staging / "COMMITTED").write_text(digest + "\n", encoding="ascii")
        verify_model_view(staging)
        if root.exists():
            existing = verify_model_view(root)
            if existing != manifest or _commit_digest(root) != digest:
                raise ValueError("changed model view context requires a new model view version")
            return ModelViewBuildResult(
                status=DatasetBuildStatus.SKIPPED_VALID,
                output_sha256=digest,
                manifest=existing,
            )
        os.replace(staging, root)
        return ModelViewBuildResult(
            status=DatasetBuildStatus.BUILT,
            output_sha256=digest,
            manifest=manifest,
        )
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()


def verify_existing_tensor_model_view(
    output_root: Path,
    *,
    view_id: str,
    view_version: str,
    task_type: TrainingTaskType | str,
    target_semantics: str,
    mask_semantics: tuple[str, ...],
    cutoff_cycle: int,
    canonical_sha256: str,
    split_sha256: str,
    builder_code_sha256: str,
    config_sha256: str,
) -> ModelViewBuildResult | None:
    root = Path(output_root)
    if not root.exists():
        return None
    manifest = verify_model_view(root)
    expected_context = (
        view_id,
        view_version,
        TrainingTaskType(task_type),
        target_semantics,
        mask_semantics,
        cutoff_cycle,
        canonical_sha256,
        split_sha256,
        builder_code_sha256,
        config_sha256,
    )
    actual_context = (
        manifest.view_id,
        manifest.view_version,
        manifest.task_type,
        manifest.target_semantics,
        manifest.mask_semantics,
        manifest.cutoff_cycle,
        manifest.canonical_sha256,
        manifest.split_sha256,
        manifest.builder_code_sha256,
        manifest.config_sha256,
    )
    if actual_context != expected_context:
        raise ValueError("changed model view context requires a new model view version")
    return ModelViewBuildResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_sha256=_commit_digest(root),
        manifest=manifest,
    )


def matr_tensor_payload(
    data: AdvancedFinalMatrData,
    *,
    view_id: Literal["early_life_sequence", "soh_trajectory"],
) -> tuple[dict[str, object], dict[str, object], dict[str, torch.Tensor], str]:
    tensors: dict[str, torch.Tensor] = {}
    partitions: list[dict[str, object]] = []
    if view_id == "early_life_sequence":
        normalizer = data.scalar_normalizer
        for partition in _PARTITIONS:
            batch = getattr(data, f"scalar_{partition}")
            early = batch.early_batch
            partition_tensors = _early_tensors(early, partition)
            partition_tensors[f"{partition}.target_cycle_life"] = batch.raw_labels
            tensors.update(partition_tensors)
            partitions.append(
                tensor_partition_metadata(
                    partition=partition,
                    cell_ids=batch.cell_ids,
                    tensors=partition_tensors,
                )
            )
        target_semantics = "matr_official_cycle_life"
        evidence = "OBSERVED_OFFICIAL_CYCLE_LIFE"
    else:
        normalizer = data.hybrid_normalizer
        for partition in _PARTITIONS:
            batch = getattr(data, f"hybrid_{partition}")
            early = batch.inputs.early_batch
            partition_tensors = _early_tensors(early, partition)
            partition_tensors.update(
                {
                    f"{partition}.initial_soh": batch.inputs.initial_soh,
                    f"{partition}.history_soh": batch.targets.history_soh,
                    f"{partition}.history_mask": batch.targets.history_mask,
                    f"{partition}.target_soh": batch.targets.target_soh,
                    f"{partition}.target_mask": batch.targets.target_mask,
                }
            )
            prediction_cycles = batch.inputs.prediction_cycles
            partition_tensors[f"{partition}.prediction_cycles"] = prediction_cycles.expand(
                len(batch.cell_ids), -1
            )
            tensors.update(partition_tensors)
            partitions.append(
                tensor_partition_metadata(
                    partition=partition,
                    cell_ids=batch.cell_ids,
                    tensors=partition_tensors,
                )
            )
        target_semantics = "matr_observed_soh_to_cycle_500"
        evidence = "OBSERVED_CAPACITY_RATIO_WITH_EXPLICIT_MASKS"
    normalization: dict[str, object] = {
        **asdict(normalizer),
        "fitted_split": "train",
    }
    metadata: dict[str, object] = {
        "schema_version": "matr-tensor-view-v1",
        "view_id": view_id,
        "dataset_id": "MATR",
        "target_semantics": target_semantics,
        "evidence": evidence,
        "right_censoring": "no fabricated targets; unavailable supervision is excluded",
        "partitions": partitions,
        "tensor_keys": sorted(tensors),
    }
    _validate_partition_isolation(partitions, data.source_split)
    return metadata, normalization, tensors, normalizer.training_cell_ids_sha256


def _early_tensors(early: Any, partition: str) -> dict[str, torch.Tensor]:
    return {
        f"{partition}.values": early.values,
        f"{partition}.cycle_indices": early.cycle_indices,
        f"{partition}.cycle_mask": early.cycle_mask,
        f"{partition}.sample_mask": early.sample_mask,
        f"{partition}.condition_values": early.condition_values,
        f"{partition}.condition_mask": early.condition_mask,
    }


def _validate_partition_isolation(
    partitions: list[dict[str, object]], split: SplitManifest
) -> None:
    observed: set[str] = set()
    for partition in partitions:
        name = str(partition["partition"])
        raw_cell_ids = partition["cell_ids"]
        if not isinstance(raw_cell_ids, list):
            raise ValueError("MATR tensor view partition requires cell_ids")
        cell_ids = tuple(str(value) for value in raw_cell_ids)
        if observed & set(cell_ids):
            raise ValueError("MATR tensor view cell crosses partitions")
        if not set(cell_ids) <= set(getattr(split, name)):
            raise ValueError("MATR tensor view partition differs from frozen cell split")
        observed.update(cell_ids)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _commit_digest(root: Path) -> str:
    return sha256_canonical(
        {
            path.name: _sha256_file(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "COMMITTED"
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "build_tensor_model_view",
    "load_verified_matr_view_data",
    "matr_tensor_payload",
    "tensor_partition_metadata",
    "verify_existing_tensor_model_view",
]
