"""Content-addressed, label-free cache for raw advanced early-cycle sequences."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
from pydantic import ConfigDict, Field, model_validator
from safetensors.torch import load_file, save_file

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.features.early_cycle_sequence import (
    RAW_NORMALIZATION_VERSION,
    EarlyCycleSequence,
)
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig

_TENSOR_FILE = "sequence.safetensors"
_METADATA_FILE = "metadata.json"
_MANIFEST_FILE = "manifest.json"
_OBJECT_FILES = frozenset({_TENSOR_FILE, _METADATA_FILE, _MANIFEST_FILE})
_TENSOR_KEYS = frozenset(
    {
        "values",
        "cycle_indices",
        "cycle_mask",
        "sample_mask",
        "condition_values",
        "condition_mask",
    }
)


class MultichannelCacheConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    samples_per_phase: Literal[150] = 150
    min_phase_points: int = Field(ge=3)
    time_monotonic_tolerance_s: float = Field(ge=0.0, allow_inf_nan=False)
    capacity_monotonic_tolerance_ah: float = Field(ge=0.0, allow_inf_nan=False)
    relative_capacity_jitter_tolerance: float = Field(ge=0.0, allow_inf_nan=False)
    max_phase_segments: int = Field(ge=1)

    @classmethod
    def from_config(cls, config: MultichannelCycleConfig) -> MultichannelCacheConfig:
        return cls(
            cutoff_cycle=config.cutoff_cycle,
            feature_version=config.feature_version,
            samples_per_phase=150,
            min_phase_points=config.min_phase_points,
            time_monotonic_tolerance_s=config.time_monotonic_tolerance_s,
            capacity_monotonic_tolerance_ah=config.capacity_monotonic_tolerance_ah,
            relative_capacity_jitter_tolerance=(
                config.relative_capacity_jitter_tolerance
            ),
            max_phase_segments=config.max_phase_segments,
        )


class EarlySequenceCacheContext(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_parquet_sha256: Sha256
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    multichannel_config: MultichannelCacheConfig

    @model_validator(mode="after")
    def context_matches_config(self) -> EarlySequenceCacheContext:
        if self.cutoff_cycle != self.multichannel_config.cutoff_cycle:
            raise ValueError("cache cutoff_cycle differs from multichannel config")
        if self.feature_version != self.multichannel_config.feature_version:
            raise ValueError("cache feature_version differs from multichannel config")
        return self

    @classmethod
    def from_multichannel_config(
        cls,
        *,
        source_parquet_sha256: str,
        dataset_id: str,
        cell_id: str,
        data_version: str,
        config: MultichannelCycleConfig,
    ) -> EarlySequenceCacheContext:
        return cls(
            source_parquet_sha256=source_parquet_sha256,
            dataset_id=dataset_id,
            cell_id=cell_id,
            cutoff_cycle=config.cutoff_cycle,
            feature_version=config.feature_version,
            data_version=data_version,
            multichannel_config=MultichannelCacheConfig.from_config(config),
        )

    @property
    def context_sha256(self) -> str:
        return sha256_canonical(
            {
                "schema_version": "early-sequence-cache-context-v1",
                **self.model_dump(mode="json"),
            }
        )


class EarlySequenceCacheMetadata(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["early-sequence-cache-metadata-v1"] = (
        "early-sequence-cache-metadata-v1"
    )
    source_parquet_sha256: Sha256
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    multichannel_config: MultichannelCacheConfig
    context_sha256: Sha256
    cache_key: Sha256
    sequence_input_hash: Sha256
    cycle_indices: tuple[int, ...] = Field(min_length=1)
    phase_names: tuple[str, ...] = Field(min_length=1)
    variable_names: tuple[str, ...] = Field(min_length=1)
    condition_names: tuple[str, ...]
    normalization_version: Literal["none"] = "none"
    normalization_statistics_sha256: None = None

    @model_validator(mode="after")
    def metadata_is_bound(self) -> EarlySequenceCacheMetadata:
        context = self.context
        if context.context_sha256 != self.context_sha256:
            raise ValueError("metadata context_sha256 does not match its context")
        expected_key = _cache_key(context, self.sequence_input_hash)
        if expected_key != self.cache_key:
            raise ValueError("metadata cache_key does not match context and sequence hash")
        if self.cycle_indices != tuple(range(self.cutoff_cycle + 1)):
            raise ValueError("metadata cycle_indices must equal 0..cutoff_cycle")
        return self

    @property
    def context(self) -> EarlySequenceCacheContext:
        return EarlySequenceCacheContext(
            source_parquet_sha256=self.source_parquet_sha256,
            dataset_id=self.dataset_id,
            cell_id=self.cell_id,
            cutoff_cycle=self.cutoff_cycle,
            feature_version=self.feature_version,
            data_version=self.data_version,
            multichannel_config=self.multichannel_config,
        )


class EarlySequenceCacheFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["tensors", "metadata"]
    relative_path: Literal["sequence.safetensors", "metadata.json"]
    size_bytes: int = Field(gt=0)
    sha256: Sha256


class EarlySequenceCacheManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["early-sequence-cache-manifest-v1"] = (
        "early-sequence-cache-manifest-v1"
    )
    cache_key: Sha256
    context_sha256: Sha256
    sequence_input_hash: Sha256
    files: tuple[EarlySequenceCacheFile, EarlySequenceCacheFile]
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def manifest_is_complete(self) -> EarlySequenceCacheManifest:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if sha256_canonical(payload) != self.manifest_sha256:
            raise ValueError("cache manifest_sha256 does not match its contents")
        roles = {item.role: item.relative_path for item in self.files}
        if roles != {"tensors": _TENSOR_FILE, "metadata": _METADATA_FILE}:
            raise ValueError("cache manifest must contain exact tensor and metadata files")
        return self


class EarlySequenceCachePointer(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["early-sequence-cache-pointer-v1"] = (
        "early-sequence-cache-pointer-v1"
    )
    context_sha256: Sha256
    cache_key: Sha256
    pointer_sha256: Sha256

    @model_validator(mode="after")
    def pointer_hash_is_valid(self) -> EarlySequenceCachePointer:
        payload = self.model_dump(mode="json", exclude={"pointer_sha256"})
        if sha256_canonical(payload) != self.pointer_sha256:
            raise ValueError("cache pointer_sha256 does not match its contents")
        return self


class EarlyCycleSequenceCache:
    """Store and retrieve immutable raw sequences without supervision fields."""

    def __init__(self, root: Path) -> None:
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ValueError("cache root must be a regular directory")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        self.objects = self.root / "objects"
        self.contexts = self.root / "contexts"
        self.objects.mkdir(exist_ok=True)
        self.contexts.mkdir(exist_ok=True)

    def cache_key(self, context: EarlySequenceCacheContext, sequence_input_hash: str) -> str:
        return _cache_key(context, sequence_input_hash)

    def get_or_build(
        self,
        context: EarlySequenceCacheContext,
        builder: Callable[[], EarlyCycleSequence],
    ) -> EarlyCycleSequence:
        context = EarlySequenceCacheContext.model_validate(context.model_dump(mode="python"))
        pointer_path = self._pointer_path(context)
        if pointer_path.exists() or pointer_path.is_symlink():
            return self.load(context)
        sequence = builder()
        self.store(context, sequence)
        return self.load(context)

    def store(
        self,
        context: EarlySequenceCacheContext,
        sequence: EarlyCycleSequence,
    ) -> EarlySequenceCacheManifest:
        context = EarlySequenceCacheContext.model_validate(context.model_dump(mode="python"))
        _validate_raw_sequence(context, sequence)
        cache_key = self.cache_key(context, sequence.input_hash)
        target = self.objects / cache_key
        if target.exists() or target.is_symlink():
            loaded, manifest = self._load_object(context, cache_key)
            if loaded.input_hash != sequence.input_hash:
                raise ValueError("existing cache object does not match the built sequence")
            self._write_pointer(context, cache_key)
            return manifest

        temporary = self.objects / f".{cache_key}.{os.getpid()}.{uuid4().hex[:8]}.tmp"
        temporary.mkdir()
        try:
            tensor_path = temporary / _TENSOR_FILE
            metadata_path = temporary / _METADATA_FILE
            tensors = {
                "values": sequence.values.detach().cpu().contiguous(),
                "cycle_indices": torch.tensor(sequence.cycle_indices, dtype=torch.int64),
                "cycle_mask": sequence.cycle_mask.detach().cpu().contiguous(),
                "sample_mask": sequence.sample_mask.detach().cpu().contiguous(),
                "condition_values": sequence.condition_values.detach().cpu().contiguous(),
                "condition_mask": sequence.condition_mask.detach().cpu().contiguous(),
            }
            save_file(tensors, str(tensor_path))
            metadata = _metadata(context, sequence, cache_key)
            _write_json(metadata_path, metadata.model_dump(mode="json"))
            files = (
                _file_record("tensors", tensor_path),
                _file_record("metadata", metadata_path),
            )
            manifest_payload = {
                "schema_version": "early-sequence-cache-manifest-v1",
                "cache_key": cache_key,
                "context_sha256": context.context_sha256,
                "sequence_input_hash": sequence.input_hash,
                "files": [item.model_dump(mode="json") for item in files],
            }
            manifest = EarlySequenceCacheManifest.model_validate(
                {
                    **manifest_payload,
                    "manifest_sha256": sha256_canonical(manifest_payload),
                }
            )
            _write_json(temporary / _MANIFEST_FILE, manifest.model_dump(mode="json"))
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._write_pointer(context, cache_key)
        return manifest

    def load(self, context: EarlySequenceCacheContext) -> EarlyCycleSequence:
        context = EarlySequenceCacheContext.model_validate(context.model_dump(mode="python"))
        pointer_path = self._pointer_path(context)
        if pointer_path.is_symlink() or not pointer_path.is_file():
            raise ValueError("cache pointer does not exist or is not a regular file")
        pointer = EarlySequenceCachePointer.model_validate(_read_json(pointer_path))
        if pointer.context_sha256 != context.context_sha256:
            raise ValueError("cache pointer context does not match the requested context")
        sequence, _manifest = self._load_object(context, pointer.cache_key)
        return sequence

    def _load_object(
        self,
        context: EarlySequenceCacheContext,
        cache_key: str,
    ) -> tuple[EarlyCycleSequence, EarlySequenceCacheManifest]:
        object_root = self.objects / cache_key
        if object_root.is_symlink() or not object_root.is_dir():
            raise ValueError("cache object must be a regular directory")
        resolved = object_root.resolve(strict=True)
        if not resolved.is_relative_to(self.objects.resolve(strict=True)):
            raise ValueError("cache object escaped the object root")
        entries = tuple(resolved.iterdir())
        if {entry.name for entry in entries} != _OBJECT_FILES:
            raise ValueError("cache object violates the closed-world file inventory")
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError("cache object files must be regular and non-symlinked")
        manifest = EarlySequenceCacheManifest.model_validate(
            _read_json(resolved / _MANIFEST_FILE)
        )
        if manifest.cache_key != cache_key or manifest.context_sha256 != context.context_sha256:
            raise ValueError("cache manifest does not match the requested context")
        files = {item.role: item for item in manifest.files}
        file_specs: tuple[
            tuple[
                Literal["tensors", "metadata"],
                Literal["sequence.safetensors", "metadata.json"],
            ],
            ...,
        ] = (("tensors", "sequence.safetensors"), ("metadata", "metadata.json"))
        for role, name in file_specs:
            path = resolved / name
            record = files[role]
            if path.stat().st_size != record.size_bytes:
                raise ValueError("cache file size does not match the manifest")
            if _sha256_file(path) != record.sha256:
                raise ValueError("cache file SHA-256 does not match the manifest")
        metadata = EarlySequenceCacheMetadata.model_validate(
            _read_json(resolved / _METADATA_FILE)
        )
        if metadata.context != context or metadata.cache_key != cache_key:
            raise ValueError("cache metadata does not match the requested context")
        if metadata.sequence_input_hash != manifest.sequence_input_hash:
            raise ValueError("cache metadata sequence hash differs from the manifest")
        tensors = load_file(str(resolved / _TENSOR_FILE), device="cpu")
        sequence = _sequence_from_tensors(metadata, tensors)
        if sequence.input_hash != manifest.sequence_input_hash:
            raise ValueError("cache sequence input_hash differs from the manifest")
        if self.cache_key(context, sequence.input_hash) != cache_key:
            raise ValueError("cache object key differs from its verified sequence")
        return sequence, manifest

    def _pointer_path(self, context: EarlySequenceCacheContext) -> Path:
        return self.contexts / f"{context.context_sha256}.json"

    def _write_pointer(self, context: EarlySequenceCacheContext, cache_key: str) -> None:
        payload = {
            "schema_version": "early-sequence-cache-pointer-v1",
            "context_sha256": context.context_sha256,
            "cache_key": cache_key,
        }
        pointer = EarlySequenceCachePointer.model_validate(
            {**payload, "pointer_sha256": sha256_canonical(payload)}
        )
        path = self._pointer_path(context)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
        _write_json(temporary, pointer.model_dump(mode="json"))
        os.replace(temporary, path)


def _cache_key(context: EarlySequenceCacheContext, sequence_input_hash: str) -> str:
    if len(sequence_input_hash) != 64 or any(
        char not in "0123456789abcdef" for char in sequence_input_hash
    ):
        raise ValueError("sequence_input_hash must be a lowercase SHA-256")
    return sha256_canonical(
        {
            "schema_version": "early-sequence-cache-key-v1",
            "context_sha256": context.context_sha256,
            "sequence_input_hash": sequence_input_hash,
        }
    )


def _validate_raw_sequence(
    context: EarlySequenceCacheContext,
    sequence: EarlyCycleSequence,
) -> None:
    sequence.verify_input_hash()
    if sequence.normalization_version != RAW_NORMALIZATION_VERSION:
        raise ValueError("advanced raw cache accepts only unnormalized sequences")
    if sequence.normalization_statistics_sha256 is not None:
        raise ValueError("advanced raw cache forbids normalization statistics")
    expected = (
        context.dataset_id,
        context.cell_id,
        context.cutoff_cycle,
        context.data_version,
        context.feature_version,
    )
    observed = (
        sequence.dataset_id,
        sequence.cell_id,
        sequence.cutoff_cycle,
        sequence.data_version,
        sequence.feature_version,
    )
    if observed != expected:
        raise ValueError("built sequence identity does not match the cache context")


def _metadata(
    context: EarlySequenceCacheContext,
    sequence: EarlyCycleSequence,
    cache_key: str,
) -> EarlySequenceCacheMetadata:
    return EarlySequenceCacheMetadata(
        **context.model_dump(mode="python"),
        context_sha256=context.context_sha256,
        cache_key=cache_key,
        sequence_input_hash=sequence.input_hash,
        cycle_indices=sequence.cycle_indices,
        phase_names=sequence.phase_names,
        variable_names=sequence.variable_names,
        condition_names=sequence.condition_names,
        normalization_version="none",
        normalization_statistics_sha256=None,
    )


def _sequence_from_tensors(
    metadata: EarlySequenceCacheMetadata,
    tensors: dict[str, torch.Tensor],
) -> EarlyCycleSequence:
    if set(tensors) != _TENSOR_KEYS:
        raise ValueError("cache safetensors contains unexpected or missing tensor keys")
    expected_dtypes = {
        "values": torch.float32,
        "cycle_indices": torch.int64,
        "cycle_mask": torch.bool,
        "sample_mask": torch.bool,
        "condition_values": torch.float32,
        "condition_mask": torch.bool,
    }
    for name, dtype in expected_dtypes.items():
        if tensors[name].dtype is not dtype:
            raise ValueError(f"cache tensor {name} has an invalid dtype")
    cycle_indices_tensor = tensors["cycle_indices"]
    if cycle_indices_tensor.ndim != 1:
        raise ValueError("cache cycle_indices tensor must be one-dimensional")
    cycle_indices = tuple(int(value) for value in cycle_indices_tensor.tolist())
    if cycle_indices != metadata.cycle_indices:
        raise ValueError("cache cycle_indices tensor differs from metadata")
    values = tensors["values"]
    cycle_mask = tensors["cycle_mask"]
    sample_mask = tensors["sample_mask"]
    condition_values = tensors["condition_values"]
    condition_mask = tensors["condition_mask"]
    if values.ndim != 4 or tuple(values.shape) != (
        len(cycle_indices),
        len(metadata.phase_names),
        metadata.multichannel_config.samples_per_phase,
        len(metadata.variable_names),
    ):
        raise ValueError("cache values tensor shape does not match metadata")
    if cycle_mask.shape != (len(cycle_indices),):
        raise ValueError("cache cycle_mask shape does not align with cycles")
    if sample_mask.shape != values.shape[:-1]:
        raise ValueError("cache sample_mask shape does not align with values")
    expected_conditions = (len(metadata.condition_names),)
    if condition_values.shape != expected_conditions or condition_mask.shape != expected_conditions:
        raise ValueError("cache condition tensor shape does not align with metadata")
    observed_values = values[sample_mask]
    missing_values = values[~sample_mask]
    if observed_values.numel() and not bool(torch.isfinite(observed_values).all().item()):
        raise ValueError("cache observed values must be finite")
    if missing_values.numel() and not bool(torch.isnan(missing_values).all().item()):
        raise ValueError("cache missing values must be NaN")
    observed_conditions = condition_values[condition_mask]
    missing_conditions = condition_values[~condition_mask]
    if observed_conditions.numel() and not bool(
        torch.isfinite(observed_conditions).all().item()
    ):
        raise ValueError("cache observed conditions must be finite")
    if missing_conditions.numel() and not bool(torch.isnan(missing_conditions).all().item()):
        raise ValueError("cache missing conditions must be NaN")
    sequence = EarlyCycleSequence(
        dataset_id=metadata.dataset_id,
        cell_id=metadata.cell_id,
        cutoff_cycle=metadata.cutoff_cycle,
        data_version=metadata.data_version,
        feature_version=metadata.feature_version,
        cycle_indices=cycle_indices,
        values=values,
        cycle_mask=cycle_mask,
        sample_mask=sample_mask,
        condition_names=metadata.condition_names,
        condition_values=condition_values,
        condition_mask=condition_mask,
        phase_names=metadata.phase_names,
        variable_names=metadata.variable_names,
        normalization_version=metadata.normalization_version,
        normalization_statistics_sha256=metadata.normalization_statistics_sha256,
    )
    if sequence.input_hash != metadata.sequence_input_hash:
        raise ValueError("cache sequence input_hash differs from metadata")
    return sequence


def _file_record(
    role: Literal["tensors", "metadata"],
    path: Path,
) -> EarlySequenceCacheFile:
    relative_path: Literal["sequence.safetensors", "metadata.json"] = (
        "sequence.safetensors" if role == "tensors" else "metadata.json"
    )
    return EarlySequenceCacheFile(
        role=role,
        relative_path=relative_path,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("cache JSON must be a regular non-symlinked file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cache JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("cache JSON must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
