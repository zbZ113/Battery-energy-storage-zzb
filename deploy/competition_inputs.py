"""Strict operator-owned inputs for the single-node competition runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceRegistration,
)
from quanxin_life.core.schemas import ContractModel, Sha256

_MAX_CONFIG_BYTES = 1_048_576


class AdvancedAgentRuntimePolicy(ContractModel):
    """Versioned server policy used to compile trusted Agent references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-agent-policy-v1"]
    conformal_alpha: float = Field(gt=0.0, lt=1.0)

    @field_validator("conformal_alpha")
    @classmethod
    def alpha_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("conformal_alpha must be finite")
        return value


@dataclass(frozen=True, slots=True)
class LoadedAdvancedAgentPolicy:
    policy: AdvancedAgentRuntimePolicy
    file_sha256: str


class _CalibrationSourceEntry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registration_id: str = Field(min_length=1, max_length=200)
    evidence_relative_root: str = Field(min_length=1, max_length=500)
    three_batch_manifest_sha256: Sha256

    @field_validator("evidence_relative_root")
    @classmethod
    def root_is_bounded_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if (
            path.is_absolute()
            or value in {"", "."}
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        ):
            raise ValueError("evidence root must be a bounded relative path")
        return path.as_posix()


class _CalibrationSourceRegistry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-calibration-source-registry-v1"]
    sources: tuple[_CalibrationSourceEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def registration_ids_are_unique(self) -> _CalibrationSourceRegistry:
        identities = tuple(item.registration_id for item in self.sources)
        if len(identities) != len(set(identities)):
            raise ValueError("calibration source registration IDs must be unique")
        return self


def load_advanced_agent_policy(path: Path) -> LoadedAdvancedAgentPolicy:
    """Load one strict policy file and retain its byte identity."""

    payload = _read_config(path, "Agent policy")
    parsed = _strict_json(payload)
    try:
        policy = AdvancedAgentRuntimePolicy.model_validate(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Agent policy is invalid") from exc
    return LoadedAdvancedAgentPolicy(
        policy=policy,
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_calibration_source_registrations(
    path: Path,
    *,
    evidence_root: Path,
) -> tuple[AdvancedCalibrationSourceRegistration, ...]:
    """Resolve path-free registrations below one operator-owned evidence root."""

    root = _regular_directory(evidence_root, "calibration evidence root")
    payload = _read_config(path, "calibration source registry")
    parsed = _strict_json(payload)
    try:
        registry = _CalibrationSourceRegistry.model_validate(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration source registry is invalid: {exc}") from None
    return tuple(
        AdvancedCalibrationSourceRegistration(
            registration_id=entry.registration_id,
            evidence_root=_bounded_subdirectory(
                root,
                entry.evidence_relative_root,
            ),
            three_batch_manifest_sha256=entry.three_batch_manifest_sha256,
        )
        for entry in registry.sources
    )


def _read_config(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError(f"{label} is too large")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} cannot be read") from exc


def _strict_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("configuration JSON is invalid") from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from None


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir() or not path.is_absolute():
        raise ValueError(f"{label} must be an absolute regular directory")
    return path.resolve(strict=True)


def _bounded_subdirectory(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("calibration evidence path must not contain symbolic links")
    if not candidate.is_dir():
        raise ValueError("calibration evidence source directory is missing")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("calibration evidence source escaped its registered root")
    return resolved


__all__ = [
    "AdvancedAgentRuntimePolicy",
    "LoadedAdvancedAgentPolicy",
    "load_advanced_agent_policy",
    "load_calibration_source_registrations",
]
