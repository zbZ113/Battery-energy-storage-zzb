"""Strict safetensors artifacts for the governed CPMLP and Hybrid models.

No class name from disk is imported dynamically.  Architecture and feature
JSON files are parsed into an explicit allow-list before a network is built.
Only then are hash-verified, finite float32 tensors loaded with strict keys and
shapes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.models.cpmlp import (
    CPMLPLifePredictor,
    _CPMLPNetwork,
    _CurveSchema,
)
from quanxin_life.models.hybrid_degradation import (
    HybridDegradationPredictor,
    _FittedContext,
    _HybridNetwork,
)


class DeepArtifactKind(StrEnum):
    CPMLP = "cpmlp-eol80"
    HYBRID = "hybrid-soh-trajectory"


class DeepArtifactFileRole(StrEnum):
    WEIGHTS = "weights"
    ARCHITECTURE = "architecture"
    FEATURE_CONFIG = "feature-config"


class DeepArtifactFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: DeepArtifactFileRole
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain inside the artifact root")
        return path.as_posix()


class DeepModelArtifactManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["deep-model-artifact-v1"] = "deep-model-artifact-v1"
    artifact_id: str
    artifact_kind: DeepArtifactKind
    files: tuple[DeepArtifactFile, ...] = Field(min_length=3, max_length=3)
    created_at: datetime
    manifest_sha256: Sha256

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def file_roles_are_exact(self) -> DeepModelArtifactManifest:
        roles = [item.role for item in self.files]
        if len(set(roles)) != len(roles) or set(roles) != set(DeepArtifactFileRole):
            raise ValueError("deep artifact must contain exactly one file for every role")
        return self


class CPMLPArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cpmlp-architecture-v1"] = "cpmlp-architecture-v1"
    architecture: Literal["quanxin_cpmlp_eol80"] = "quanxin_cpmlp_eol80"
    curve_hidden_dim: int = Field(gt=0)
    aggregation_hidden_dim: int = Field(gt=0)


class CPMLPFeatureConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cpmlp-feature-context-v1"] = "cpmlp-feature-context-v1"
    dataset_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    cycle_indices: tuple[int, ...] = Field(min_length=1)
    voltage_grid_v: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def axes_are_valid(self) -> CPMLPFeatureConfig:
        if any(
            current <= previous
            for previous, current in zip(
                self.cycle_indices, self.cycle_indices[1:], strict=False
            )
        ):
            raise ValueError("cycle_indices must be strictly increasing")
        if any(not math.isfinite(value) for value in self.voltage_grid_v):
            raise ValueError("voltage_grid_v must be finite")
        return self


class HybridArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hybrid-architecture-v1"] = "hybrid-architecture-v1"
    architecture: Literal["quanxin_hybrid_soh_trajectory"] = (
        "quanxin_hybrid_soh_trajectory"
    )
    hidden_dim: int = Field(gt=0)


class HybridFeatureConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hybrid-feature-context-v1"] = "hybrid-feature-context-v1"
    dataset_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    condition_feature_names: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def context_is_valid(self) -> HybridFeatureConfig:
        if len(set(self.condition_feature_names)) != len(self.condition_feature_names):
            raise ValueError("condition_feature_names must be unique")
        if any(not name.strip() for name in self.condition_feature_names):
            raise ValueError("condition_feature_names must not contain blank names")
        if any(
            current <= previous
            for previous, current in zip(
                self.prediction_cycles, self.prediction_cycles[1:], strict=False
            )
        ):
            raise ValueError("prediction_cycles must be strictly increasing")
        if any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
        return self


def export_cpmlp_artifact(
    predictor: CPMLPLifePredictor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
) -> DeepModelArtifactManifest:
    if (
        predictor._network is None
        or predictor._dataset_id is None
        or predictor._curve_schema is None
    ):
        raise ValueError("CPMLP predictor must be fitted before export")
    architecture = CPMLPArchitecture(
        curve_hidden_dim=predictor.curve_hidden_dim,
        aggregation_hidden_dim=predictor.aggregation_hidden_dim,
    )
    feature_config = CPMLPFeatureConfig(
        dataset_id=predictor._dataset_id,
        model_version=predictor.model_version,
        feature_version=predictor.feature_version,
        split_version=predictor.split_version,
        data_version=predictor.data_version,
        cutoff_cycle=predictor.cutoff_cycle,
        cycle_indices=predictor._curve_schema.cycle_indices,
        voltage_grid_v=predictor._curve_schema.voltage_grid_v,
    )
    return _export_deep_artifact(
        network=predictor._network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CPMLP,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature_config.model_dump(mode="json"),
        created_at=created_at,
    )


def export_hybrid_artifact(
    predictor: HybridDegradationPredictor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
) -> DeepModelArtifactManifest:
    if predictor._network is None or predictor._context is None:
        raise ValueError("Hybrid predictor must be fitted before export")
    architecture = HybridArchitecture(hidden_dim=predictor.hidden_dim)
    feature_config = HybridFeatureConfig(
        dataset_id=predictor._context.dataset_id,
        model_version=predictor.model_version,
        feature_version=predictor.feature_version,
        split_version=predictor.split_version,
        data_version=predictor.data_version,
        cutoff_cycle=predictor.cutoff_cycle,
        prediction_cycles=predictor.prediction_cycles,
        condition_feature_names=predictor.condition_feature_names,
    )
    return _export_deep_artifact(
        network=predictor._network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.HYBRID,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature_config.model_dump(mode="json"),
        created_at=created_at,
    )


def load_cpmlp_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> CPMLPLifePredictor:
    if manifest.artifact_kind is not DeepArtifactKind.CPMLP:
        raise ValueError("artifact is not a CPMLP model")
    files = _verify_deep_artifact(artifact_root, manifest)
    architecture = CPMLPArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = CPMLPFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    network = _CPMLPNetwork(
        voltage_points=len(feature.voltage_grid_v),
        cutoff_cycle=feature.cutoff_cycle,
        curve_hidden_dim=architecture.curve_hidden_dim,
        aggregation_hidden_dim=architecture.aggregation_hidden_dim,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    predictor = CPMLPLifePredictor(
        model_version=feature.model_version,
        feature_version=feature.feature_version,
        split_version=feature.split_version,
        data_version=feature.data_version,
        cutoff_cycle=feature.cutoff_cycle,
        curve_hidden_dim=architecture.curve_hidden_dim,
        aggregation_hidden_dim=architecture.aggregation_hidden_dim,
        epochs=1,
    )
    predictor._network = network.eval()
    predictor._dataset_id = feature.dataset_id
    predictor._curve_schema = _CurveSchema(
        cycle_indices=feature.cycle_indices,
        voltage_grid_v=feature.voltage_grid_v,
    )
    return predictor


def load_hybrid_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> HybridDegradationPredictor:
    if manifest.artifact_kind is not DeepArtifactKind.HYBRID:
        raise ValueError("artifact is not a Hybrid model")
    files = _verify_deep_artifact(artifact_root, manifest)
    architecture = HybridArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = HybridFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    network = _HybridNetwork(
        input_dim=3 + len(feature.condition_feature_names),
        horizon=len(feature.prediction_cycles),
        hidden_dim=architecture.hidden_dim,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    predictor = HybridDegradationPredictor(
        model_version=feature.model_version,
        feature_version=feature.feature_version,
        split_version=feature.split_version,
        data_version=feature.data_version,
        cutoff_cycle=feature.cutoff_cycle,
        prediction_cycles=feature.prediction_cycles,
        condition_feature_names=feature.condition_feature_names,
        hidden_dim=architecture.hidden_dim,
        epochs=1,
    )
    predictor._network = network.eval()
    predictor._context = _FittedContext(
        dataset_id=feature.dataset_id,
        condition_feature_names=feature.condition_feature_names,
    )
    return predictor


def _export_deep_artifact(
    *,
    network: torch.nn.Module,
    artifact_root: Path,
    artifact_id: str,
    artifact_kind: DeepArtifactKind,
    architecture: dict[str, Any],
    feature_config: dict[str, Any],
    created_at: datetime,
) -> DeepModelArtifactManifest:
    try:
        normalized_id = str(UUID(artifact_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("artifact_id must be a UUID string") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    root = _validated_root(artifact_root)
    target = root / normalized_id
    if target.exists() or target.is_symlink():
        raise ValueError("artifact target already exists")
    temporary = root / f".{normalized_id}.{os.getpid()}.tmp"
    if temporary.exists():
        raise ValueError("temporary artifact target already exists")
    temporary.mkdir()
    try:
        weights_path = temporary / "model.safetensors"
        architecture_path = temporary / "architecture.json"
        feature_path = temporary / "feature_config.json"
        _save_safetensors(network, weights_path)
        _write_json(architecture_path, architecture)
        _write_json(feature_path, feature_config)
        files = tuple(
            DeepArtifactFile(
                role=role,
                relative_path=f"{normalized_id}/{path.name}",
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for role, path in (
                (DeepArtifactFileRole.WEIGHTS, weights_path),
                (DeepArtifactFileRole.ARCHITECTURE, architecture_path),
                (DeepArtifactFileRole.FEATURE_CONFIG, feature_path),
            )
        )
        manifest_payload = {
            "schema_version": "deep-model-artifact-v1",
            "artifact_id": normalized_id,
            "artifact_kind": artifact_kind.value,
            "files": [item.model_dump(mode="json") for item in files],
            "created_at": created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }
        manifest = DeepModelArtifactManifest.model_validate(
            {
                **manifest_payload,
                "manifest_sha256": sha256_canonical(manifest_payload),
            }
        )
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        temporary.replace(target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_deep_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> dict[DeepArtifactFileRole, Path]:
    root = _validated_root(artifact_root)
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if sha256_canonical(payload) != manifest.manifest_sha256:
        raise ValueError("manifest SHA-256 does not match its contents")
    artifact_directory = root / manifest.artifact_id
    if artifact_directory.is_symlink():
        raise ValueError("artifact directory must not be a symbolic link")
    resolved_directory = artifact_directory.resolve(strict=True)
    if not resolved_directory.is_relative_to(root) or not resolved_directory.is_dir():
        raise ValueError("artifact directory must remain inside the artifact root")
    manifest_path = resolved_directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("artifact manifest must not be a symbolic link")
    disk_manifest = DeepModelArtifactManifest.model_validate(_read_strict_json(manifest_path))
    if disk_manifest != manifest:
        raise ValueError("on-disk artifact manifest does not match the registered manifest")

    expected_names = {"manifest.json", *(Path(item.relative_path).name for item in manifest.files)}
    actual_names = {item.name for item in resolved_directory.iterdir()}
    if actual_names != expected_names:
        raise ValueError("artifact directory contains an unexpected file")

    verified: dict[DeepArtifactFileRole, Path] = {}
    for item in manifest.files:
        path = root / item.relative_path
        if path.is_symlink():
            raise ValueError("artifact file must not be a symbolic link")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_directory) or not resolved.is_file():
            raise ValueError("artifact file must remain inside its artifact directory")
        if resolved.stat().st_size != item.size_bytes:
            raise ValueError("artifact file size does not match the manifest")
        if _sha256_file(resolved) != item.sha256:
            raise ValueError("artifact file SHA-256 does not match the manifest")
        verified[item.role] = resolved
    return verified


def _save_safetensors(network: torch.nn.Module, path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for deep model export") from exc
    state: dict[str, Tensor] = {}
    for name, tensor in network.state_dict().items():
        value = tensor.detach().to(device="cpu").contiguous()
        if value.dtype is not torch.float32:
            raise ValueError("deep model weights must use float32")
        if not torch.isfinite(value).all():
            raise ValueError("deep model weights must be finite")
        state[name] = value
    save_file(state, str(path))


def _load_verified_state(network: torch.nn.Module, path: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for deep model loading") from exc
    state = load_file(str(path), device="cpu")
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("safetensors keys do not exactly match the approved architecture")
    for name, value in state.items():
        if value.dtype is not torch.float32:
            raise ValueError("safetensors weights must use float32")
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError("safetensors weight shape does not match the approved architecture")
        if not torch.isfinite(value).all():
            raise ValueError("safetensors weights must be finite")
    try:
        network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "safetensors state is incompatible with the approved architecture"
        ) from exc


def _validated_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("artifact root must exist") from exc
    if root.is_symlink() or not resolved.is_dir():
        raise ValueError("artifact root must be a regular directory")
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
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact JSON must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
