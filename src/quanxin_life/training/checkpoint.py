"""Strict safetensors checkpoints for resumable PyTorch training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from safetensors.torch import load_file, save_file

from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256


class CheckpointContext(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    dataset_id: str = Field(min_length=1)
    target: PredictionTarget
    model_name: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    seed: int
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class AdvancedCheckpointContext(CheckpointContext):
    """Hash-bound experiment identity for governed candidate training."""

    model_name: Literal[
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    ]
    run_mode: Literal["smoke", "select", "final"]
    stage: Literal[
        "smoke",
        "selection_candidate",
        "selection_recheck",
        "final",
    ]
    candidate_config_sha256: Sha256
    model_architecture_sha256: Sha256
    normalization_sha256: Sha256
    selection_manifest_sha256: Sha256 | None = None
    reference_library_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def optional_hashes_match_run_semantics(self) -> AdvancedCheckpointContext:
        is_batlinet = self.model_name == "cyclepatch_batlinet"
        if is_batlinet != (self.reference_library_sha256 is not None):
            raise ValueError("reference_library_sha256 is required only for BatLiNet")
        approved_stages = {
            "smoke": {"smoke"},
            "select": {"selection_candidate", "selection_recheck"},
            "final": {"final"},
        }
        if self.stage not in approved_stages[self.run_mode]:
            raise ValueError("stage does not match run_mode")
        if self.run_mode == "final":
            if self.selection_manifest_sha256 is None:
                raise ValueError("selection_manifest_sha256 is required for final runs")
        elif self.selection_manifest_sha256 is not None:
            raise ValueError("selection_manifest_sha256 must be absent for smoke/select runs")
        return self


class TrainingProgress(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    best_epoch: int | None = Field(default=None, ge=0)
    best_metric: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    validations_without_improvement: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def best_fields_are_consistent(self) -> TrainingProgress:
        if (self.best_epoch is None) != (self.best_metric is None):
            raise ValueError("best_epoch and best_metric must be set together")
        if self.best_epoch is not None and self.best_epoch > self.epoch:
            raise ValueError("best_epoch cannot exceed the current epoch")
        return self


class CheckpointFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    size_bytes: int = Field(gt=0)
    sha256: Sha256


class TrainingCheckpointManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["safe-training-checkpoint-v1"] = "safe-training-checkpoint-v1"
    context: CheckpointContext
    progress: TrainingProgress
    files: tuple[CheckpointFile, ...] = Field(min_length=6, max_length=6)
    created_at: datetime
    manifest_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def file_inventory_is_exact(self) -> TrainingCheckpointManifest:
        expected = {
            "model.safetensors",
            "optimizer.safetensors",
            "optimizer_state.json",
            "scheduler_state.json",
            "rng_state.safetensors",
            "progress.json",
        }
        paths = {item.relative_path for item in self.files}
        if len(paths) != len(self.files) or paths != expected:
            raise ValueError("checkpoint file inventory must contain the six approved files")
        return self


class AdvancedTrainingCheckpointManifest(ContractModel):
    """Closed-world checkpoint manifest with advanced experiment provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["safe-training-checkpoint-v2"] = "safe-training-checkpoint-v2"
    context: AdvancedCheckpointContext
    progress: TrainingProgress
    files: tuple[CheckpointFile, ...] = Field(min_length=6, max_length=6)
    created_at: datetime
    manifest_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def file_inventory_is_exact(self) -> AdvancedTrainingCheckpointManifest:
        expected = {
            "model.safetensors",
            "optimizer.safetensors",
            "optimizer_state.json",
            "scheduler_state.json",
            "rng_state.safetensors",
            "progress.json",
        }
        paths = {item.relative_path for item in self.files}
        if len(paths) != len(self.files) or paths != expected:
            raise ValueError("checkpoint file inventory must contain the six approved files")
        return self


def save_training_checkpoint(
    checkpoint_root: Path,
    *,
    context: CheckpointContext,
    progress: TrainingProgress,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    created_at: datetime | None = None,
) -> TrainingCheckpointManifest:
    """Write one new, closed-world checkpoint directory."""

    root = _validated_checkpoint_root(checkpoint_root, require_empty=True)
    model_state = _safe_tensor_mapping(model.state_dict(), prefix="model")
    save_file(model_state, str(root / "model.safetensors"))

    optimizer_tensors, optimizer_metadata = _serialize_optimizer(model, optimizer)
    save_file(optimizer_tensors, str(root / "optimizer.safetensors"))
    _write_json(root / "optimizer_state.json", optimizer_metadata)
    _write_json(
        root / "scheduler_state.json",
        {"state": None if scheduler is None else _encode_non_finite_floats(scheduler.state_dict())},
    )

    rng_tensors, rng_metadata = _capture_rng_state()
    save_file(rng_tensors, str(root / "rng_state.safetensors"))
    _write_json(
        root / "progress.json",
        {
            "context": context.model_dump(mode="json"),
            "progress": progress.model_dump(mode="json"),
            "rng": rng_metadata,
        },
    )

    files = tuple(
        CheckpointFile(
            relative_path=name,
            size_bytes=(root / name).stat().st_size,
            sha256=_sha256_file(root / name),
        )
        for name in (
            "model.safetensors",
            "optimizer.safetensors",
            "optimizer_state.json",
            "scheduler_state.json",
            "rng_state.safetensors",
            "progress.json",
        )
    )
    timestamp = created_at or datetime.now(UTC)
    manifest_payload = {
        "schema_version": "safe-training-checkpoint-v1",
        "context": context.model_dump(mode="json"),
        "progress": progress.model_dump(mode="json"),
        "files": [item.model_dump(mode="json") for item in files],
        "created_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    manifest = TrainingCheckpointManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    _write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def load_training_checkpoint(
    checkpoint_root: Path,
    manifest: TrainingCheckpointManifest,
    *,
    expected_context: CheckpointContext,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
) -> TrainingProgress:
    """Verify every byte and restore the approved training state."""

    root = _validated_checkpoint_root(checkpoint_root, require_empty=False)
    if manifest.context != expected_context:
        raise ValueError("checkpoint context does not match the requested training run")
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if sha256_canonical(payload) != manifest.manifest_sha256:
        raise ValueError("checkpoint manifest SHA-256 does not match its contents")
    disk_manifest = TrainingCheckpointManifest.model_validate(
        _read_strict_json(root / "manifest.json")
    )
    if disk_manifest != manifest:
        raise ValueError("on-disk checkpoint manifest does not match the registered manifest")
    expected_names = {item.relative_path for item in manifest.files} | {"manifest.json"}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("checkpoint directory contains an unexpected file")
    for item in manifest.files:
        path = root / item.relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint file must be a regular non-symlinked file")
        if path.stat().st_size != item.size_bytes:
            raise ValueError("checkpoint file size does not match the manifest")
        if _sha256_file(path) != item.sha256:
            raise ValueError("checkpoint file SHA-256 does not match the manifest")

    model_state = load_file(str(root / "model.safetensors"), device="cpu")
    _validate_state_against_model(model_state, model)
    model.load_state_dict(model_state, strict=True)
    optimizer_tensors = load_file(str(root / "optimizer.safetensors"), device="cpu")
    optimizer_metadata = _read_strict_json(root / "optimizer_state.json")
    _restore_optimizer(model, optimizer, optimizer_tensors, optimizer_metadata)
    scheduler_payload = _read_strict_json(root / "scheduler_state.json")
    scheduler_state = scheduler_payload.get("state")
    if scheduler_state is not None:
        if scheduler is None or not isinstance(scheduler_state, dict):
            raise ValueError("checkpoint scheduler state is incompatible with the run")
        decoded_scheduler_state = _decode_non_finite_floats(scheduler_state)
        if not isinstance(decoded_scheduler_state, dict):
            raise ValueError("checkpoint scheduler state is incompatible with the run")
        scheduler.load_state_dict(decoded_scheduler_state)
    elif scheduler is not None:
        raise ValueError("checkpoint is missing the requested scheduler state")

    progress_payload = _read_strict_json(root / "progress.json")
    if progress_payload.get("context") != expected_context.model_dump(mode="json"):
        raise ValueError("checkpoint progress context does not match the requested run")
    progress = TrainingProgress.model_validate(progress_payload.get("progress"))
    if progress != manifest.progress:
        raise ValueError("checkpoint progress does not match the manifest")
    rng_metadata = progress_payload.get("rng")
    if not isinstance(rng_metadata, dict):
        raise ValueError("checkpoint RNG metadata is invalid")
    rng_tensors = load_file(str(root / "rng_state.safetensors"), device="cpu")
    _restore_rng_state(rng_tensors, rng_metadata)
    return progress


def model_architecture_sha256(
    model: torch.nn.Module,
    candidate_config_sha256: str,
) -> str:
    """Hash architecture metadata without binding random parameter values."""

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if (
        not isinstance(candidate_config_sha256, str)
        or len(candidate_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in candidate_config_sha256)
    ):
        raise ValueError("candidate_config_sha256 must be a lowercase SHA-256 digest")
    parameter_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    state_schema = []
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("model state_dict entries must be tensors")
        state_schema.append(
            {
                "state_key": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "requires_grad": parameter_requires_grad.get(name),
            }
        )
    return sha256_canonical(
        {
            "schema_version": "model-architecture-v2",
            "candidate_config_sha256": candidate_config_sha256,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "state_dict": state_schema,
        }
    )


def save_advanced_training_checkpoint(
    checkpoint_root: Path,
    *,
    context: AdvancedCheckpointContext,
    progress: TrainingProgress,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    created_at: datetime | None = None,
) -> AdvancedTrainingCheckpointManifest:
    """Write one new v2 checkpoint without changing the v1 byte protocol."""

    _validate_advanced_model_architecture(context, model)
    root = _validated_checkpoint_root(checkpoint_root, require_empty=True)
    model_state = _safe_tensor_mapping(model.state_dict(), prefix="model")
    save_file(model_state, str(root / "model.safetensors"))

    optimizer_tensors, optimizer_metadata = _serialize_optimizer(model, optimizer)
    save_file(optimizer_tensors, str(root / "optimizer.safetensors"))
    _write_json(root / "optimizer_state.json", optimizer_metadata)
    _write_json(
        root / "scheduler_state.json",
        {"state": None if scheduler is None else _encode_non_finite_floats(scheduler.state_dict())},
    )

    rng_tensors, rng_metadata = _capture_rng_state()
    save_file(rng_tensors, str(root / "rng_state.safetensors"))
    _write_json(
        root / "progress.json",
        {
            "context": context.model_dump(mode="json"),
            "progress": progress.model_dump(mode="json"),
            "rng": rng_metadata,
        },
    )

    files = tuple(
        CheckpointFile(
            relative_path=name,
            size_bytes=(root / name).stat().st_size,
            sha256=_sha256_file(root / name),
        )
        for name in (
            "model.safetensors",
            "optimizer.safetensors",
            "optimizer_state.json",
            "scheduler_state.json",
            "rng_state.safetensors",
            "progress.json",
        )
    )
    timestamp = created_at or datetime.now(UTC)
    manifest_payload = {
        "schema_version": "safe-training-checkpoint-v2",
        "context": context.model_dump(mode="json"),
        "progress": progress.model_dump(mode="json"),
        "files": [item.model_dump(mode="json") for item in files],
        "created_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    manifest = AdvancedTrainingCheckpointManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    _write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def load_advanced_training_checkpoint(
    checkpoint_root: Path,
    manifest: AdvancedTrainingCheckpointManifest,
    *,
    expected_context: AdvancedCheckpointContext,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
) -> TrainingProgress:
    """Verify every v2 byte and restore only the exact advanced context."""

    _validate_advanced_model_architecture(expected_context, model)
    root, restored_progress = _verify_advanced_checkpoint(
        checkpoint_root,
        manifest,
        expected_context=expected_context,
    )

    model_state = load_file(str(root / "model.safetensors"), device="cpu")
    _validate_state_against_model(model_state, model)
    model.load_state_dict(model_state, strict=True)
    optimizer_tensors = load_file(str(root / "optimizer.safetensors"), device="cpu")
    optimizer_metadata = _read_strict_json(root / "optimizer_state.json")
    _restore_optimizer(model, optimizer, optimizer_tensors, optimizer_metadata)
    scheduler_payload = _read_strict_json(root / "scheduler_state.json")
    scheduler_state = scheduler_payload.get("state")
    if scheduler_state is not None:
        if scheduler is None or not isinstance(scheduler_state, dict):
            raise ValueError("checkpoint scheduler state is incompatible with the run")
        decoded_scheduler_state = _decode_non_finite_floats(scheduler_state)
        if not isinstance(decoded_scheduler_state, dict):
            raise ValueError("checkpoint scheduler state is incompatible with the run")
        scheduler.load_state_dict(decoded_scheduler_state)
    elif scheduler is not None:
        raise ValueError("checkpoint is missing the requested scheduler state")

    progress_payload = _read_strict_json(root / "progress.json")
    rng_metadata = progress_payload.get("rng")
    if not isinstance(rng_metadata, dict):
        raise ValueError("checkpoint RNG metadata is invalid")
    rng_tensors = load_file(str(root / "rng_state.safetensors"), device="cpu")
    _restore_rng_state(rng_tensors, rng_metadata)
    return restored_progress


def load_advanced_inference_checkpoint(
    checkpoint_root: Path,
    manifest: AdvancedTrainingCheckpointManifest,
    *,
    expected_context: AdvancedCheckpointContext,
    model: torch.nn.Module,
) -> TrainingProgress:
    """Verify a complete v2 checkpoint while restoring model weights only."""

    _validate_advanced_model_architecture(expected_context, model)
    root, progress = _verify_advanced_checkpoint(
        checkpoint_root,
        manifest,
        expected_context=expected_context,
    )
    model_state = load_file(str(root / "model.safetensors"), device="cpu")
    _validate_state_against_model(model_state, model)
    model.load_state_dict(model_state, strict=True)
    return progress


def _verify_advanced_checkpoint(
    checkpoint_root: Path,
    manifest: AdvancedTrainingCheckpointManifest,
    *,
    expected_context: AdvancedCheckpointContext,
) -> tuple[Path, TrainingProgress]:
    root = _validated_checkpoint_root(checkpoint_root, require_empty=False)
    if manifest.context != expected_context:
        raise ValueError("checkpoint context does not match the requested training run")
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if sha256_canonical(payload) != manifest.manifest_sha256:
        raise ValueError("checkpoint manifest SHA-256 does not match its contents")
    disk_manifest = AdvancedTrainingCheckpointManifest.model_validate(
        _read_strict_json(root / "manifest.json")
    )
    if disk_manifest != manifest:
        raise ValueError("on-disk checkpoint manifest does not match the registered manifest")
    expected_names = {item.relative_path for item in manifest.files} | {"manifest.json"}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("checkpoint directory contains an unexpected file")
    for item in manifest.files:
        path = root / item.relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint file must be a regular non-symlinked file")
        if path.stat().st_size != item.size_bytes:
            raise ValueError("checkpoint file size does not match the manifest")
        if _sha256_file(path) != item.sha256:
            raise ValueError("checkpoint file SHA-256 does not match the manifest")

    progress_payload = _read_strict_json(root / "progress.json")
    if progress_payload.get("context") != expected_context.model_dump(mode="json"):
        raise ValueError("checkpoint progress context does not match the requested run")
    progress = TrainingProgress.model_validate(progress_payload.get("progress"))
    if progress != manifest.progress:
        raise ValueError("checkpoint progress does not match the manifest")
    return root, progress


def _validate_advanced_model_architecture(
    context: AdvancedCheckpointContext,
    model: torch.nn.Module,
) -> None:
    observed = model_architecture_sha256(model, context.candidate_config_sha256)
    if observed != context.model_architecture_sha256:
        raise ValueError("model_architecture_sha256 does not match the requested model")


def _serialize_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state_dict = optimizer.state_dict()
    identifier_to_name: dict[int, str] = {}
    groups: list[dict[str, Any]] = []
    for live_group, serialized_group in zip(
        optimizer.param_groups, state_dict["param_groups"], strict=True
    ):
        parameter_names: list[str] = []
        for parameter, identifier in zip(
            live_group["params"], serialized_group["params"], strict=True
        ):
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside the model")
            identifier_to_name[int(identifier)] = name
            parameter_names.append(name)
        group = {key: value for key, value in serialized_group.items() if key != "params"}
        group["parameter_names"] = parameter_names
        groups.append(group)

    tensors: dict[str, torch.Tensor] = {}
    scalar_state: dict[str, dict[str, Any]] = {}
    for identifier, values in state_dict["state"].items():
        name = identifier_to_name.get(int(identifier))
        if name is None:
            raise ValueError("optimizer state contains an unknown parameter identifier")
        scalar_state[name] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to(device="cpu").contiguous()
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError("optimizer tensors must be finite")
                tensors[f"{name}::{key}"] = tensor
            else:
                _require_json_scalar(value, field_name=f"optimizer state {name}.{key}")
                scalar_state[name][key] = value
    if not tensors:
        tensors["__empty__"] = torch.empty(0, dtype=torch.uint8)
    return tensors, {"param_groups": groups, "scalar_state": scalar_state}


def _restore_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    current = optimizer.state_dict()
    saved_groups = metadata.get("param_groups")
    scalar_state = metadata.get("scalar_state")
    if not isinstance(saved_groups, list) or not isinstance(scalar_state, dict):
        raise ValueError("optimizer metadata is invalid")
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer parameter-group count does not match the checkpoint")

    name_to_identifier: dict[str, int] = {}
    restored_groups: list[dict[str, Any]] = []
    for live_group, current_group, saved_group in zip(
        optimizer.param_groups, current["param_groups"], saved_groups, strict=True
    ):
        if not isinstance(saved_group, dict):
            raise ValueError("optimizer parameter-group metadata is invalid")
        live_names = [names.get(id(parameter)) for parameter in live_group["params"]]
        saved_names = saved_group.get("parameter_names")
        if live_names != saved_names or any(name is None for name in live_names):
            raise ValueError("optimizer parameters do not match the checkpoint")
        identifiers = [int(value) for value in current_group["params"]]
        for name, identifier in zip(live_names, identifiers, strict=True):
            assert name is not None
            name_to_identifier[name] = identifier
        restored_group = {
            key: value for key, value in saved_group.items() if key != "parameter_names"
        }
        restored_group["params"] = identifiers
        restored_groups.append(restored_group)

    restored_state: dict[int, dict[str, Any]] = {}
    for name, identifier in name_to_identifier.items():
        values: dict[str, Any] = {}
        saved_scalars = scalar_state.get(name, {})
        if not isinstance(saved_scalars, dict):
            raise ValueError("optimizer scalar state is invalid")
        values.update(saved_scalars)
        prefix = f"{name}::"
        for key, tensor in tensors.items():
            if key.startswith(prefix):
                values[key.removeprefix(prefix)] = tensor
        if values:
            restored_state[identifier] = values
    optimizer.load_state_dict({"state": restored_state, "param_groups": restored_groups})


def _safe_tensor_mapping(state: dict[str, torch.Tensor], *, prefix: str) -> dict[str, torch.Tensor]:
    safe: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{prefix} tensors must be finite")
        safe[name] = tensor
    if not safe:
        raise ValueError(f"{prefix} tensor state must not be empty")
    return safe


def _validate_state_against_model(state: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint model keys do not match the requested architecture")
    for name, tensor in state.items():
        if tensor.dtype != expected[name].dtype or tensor.shape != expected[name].shape:
            raise ValueError("checkpoint model tensor does not match the requested architecture")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("checkpoint model tensors must be finite")


def _capture_rng_state() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors = {"torch_cpu": torch.get_rng_state().to(device="cpu")}
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"torch_cuda_{index}"] = state.to(device="cpu")
    numpy_state = cast(
        tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float],
        np.random.get_state(),
    )
    metadata = {
        "python": _json_sequence(random.getstate()),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }
    return tensors, metadata


def _restore_rng_state(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    torch_cpu = tensors.get("torch_cpu")
    if torch_cpu is None:
        raise ValueError("checkpoint is missing the CPU RNG state")
    torch.set_rng_state(torch_cpu)
    cuda_keys = sorted(key for key in tensors if key.startswith("torch_cuda_"))
    if cuda_keys:
        if not torch.cuda.is_available() or len(cuda_keys) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all([tensors[key] for key in cuda_keys])
    python_state = metadata.get("python")
    numpy_state = metadata.get("numpy")
    if not isinstance(python_state, list) or not isinstance(numpy_state, dict):
        raise ValueError("checkpoint RNG metadata is invalid")
    random.setstate(_tuple_sequence(python_state))
    try:
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint NumPy RNG metadata is invalid") from exc


def _validated_checkpoint_root(path: Path, *, require_empty: bool) -> Path:
    root = Path(path)
    if root.is_symlink():
        raise ValueError("checkpoint root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("checkpoint root must be a directory")
    if require_empty and any(resolved.iterdir()):
        raise ValueError("checkpoint root must be empty for a new checkpoint")
    return resolved


def _read_strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint JSON must be a regular non-symlinked file")
    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint JSON must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


_NON_FINITE_FLOAT_TAG = "__quanxin_non_finite_float__"


def _encode_non_finite_floats(value: Any) -> Any:
    """Represent framework infinity sentinels without emitting invalid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {_NON_FINITE_FLOAT_TAG: label}
    if isinstance(value, dict):
        return {str(key): _encode_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_non_finite_floats(item) for item in value]
    return value


def _decode_non_finite_floats(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_NON_FINITE_FLOAT_TAG}:
            label = value[_NON_FINITE_FLOAT_TAG]
            decoded = {"+inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(label)
            if decoded is None:
                raise ValueError("checkpoint contains an invalid non-finite float tag")
            return decoded
        return {key: _decode_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_non_finite_floats(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_json_scalar(value: Any, *, field_name: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{field_name} must be a finite JSON scalar")


def _json_sequence(value: tuple[Any, ...]) -> list[Any]:
    return [_json_sequence(item) if isinstance(item, tuple) else item for item in value]


def _tuple_sequence(value: list[Any]) -> tuple[Any, ...]:
    return tuple(_tuple_sequence(item) if isinstance(item, list) else item for item in value)
