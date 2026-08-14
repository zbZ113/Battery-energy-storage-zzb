"""Strict operator-owned inputs for the single-node competition runtime."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceRegistration,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioRegistry,
    load_reviewed_default_scenario_registry,
)

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


class _FeishuCsvRegistrationEntry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_sha256: Sha256
    registration: CanonicalCsvBatchRegistration

    @model_validator(mode="after")
    def registration_is_bound_to_payload(self) -> _FeishuCsvRegistrationEntry:
        if self.registration.metadata.source_sha256 != self.payload_sha256:
            raise ValueError("Feishu CSV registration metadata SHA-256 does not match payload")
        observed_hashes = {
            item.sha256
            for item in self.registration.provenance
            if item.source_kind.value == "OBSERVED"
        }
        if self.payload_sha256 not in observed_hashes:
            raise ValueError("Feishu CSV registration provenance SHA-256 does not match payload")
        return self


class _FeishuCsvRegistrationRegistry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["feishu-canonical-csv-registration-registry-v1"]
    registrations: tuple[_FeishuCsvRegistrationEntry, ...]

    @model_validator(mode="after")
    def payload_hashes_are_unique(self) -> _FeishuCsvRegistrationRegistry:
        identities = tuple(item.payload_sha256 for item in self.registrations)
        if len(identities) != len(set(identities)):
            raise ValueError("Feishu CSV registration payload SHA-256 values must be unique")
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


def load_feishu_csv_registrations(
    path: Path,
) -> Mapping[str, CanonicalCsvBatchRegistration]:
    """Load exact-content Feishu CSV registrations from one strict operator file."""

    payload = _read_config(path, "Feishu CSV registration registry")
    parsed = _strict_json(payload)
    try:
        registry = _FeishuCsvRegistrationRegistry.model_validate(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feishu CSV registration registry is invalid: {exc}") from None
    registrations = {
        entry.payload_sha256: CanonicalCsvBatchRegistration.model_validate(
            entry.registration.model_dump(mode="json")
        )
        for entry in registry.registrations
    }
    return MappingProxyType(registrations)


def load_feishu_default_scenario_profiles(
    path: Path,
) -> ReviewedDefaultScenarioRegistry:
    """Load the server-reviewed proactive scenario registry fail closed."""

    return load_reviewed_default_scenario_registry(path)


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
    "load_feishu_csv_registrations",
    "load_feishu_default_scenario_profiles",
]
