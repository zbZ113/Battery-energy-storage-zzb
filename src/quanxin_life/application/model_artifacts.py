"""Hash-verified registry for non-executable model artifact formats.

The registry deliberately stops at verification and safe JSON decoding.  It
does not call pickle-compatible loaders or instantiate model objects.  A
model-specific adapter may consume a :class:`VerifiedModelArtifact` only after
this module has rechecked its immutable manifest and file digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256

if TYPE_CHECKING:
    from quanxin_life.models.xgboost import XGBoostLifePredictor

_FORBIDDEN_SUFFIXES = frozenset({".joblib", ".pickle", ".pkl", ".pt", ".pth"})
_MAX_JSON_BYTES = 256 * 1024 * 1024


class ArtifactKind(StrEnum):
    """Model families currently allowed by the governed registry."""

    XGBOOST = "xgboost"
    VARIANCE = "variance"


class ArtifactFormat(StrEnum):
    """Non-executable on-disk formats admitted by the registry."""

    XGBOOST_JSON = "xgboost-json"
    XGBOOST_UBJ = "xgboost-ubj"
    VARIANCE_JSON = "variance-json"


_FORMAT_SUFFIXES = {
    ArtifactFormat.XGBOOST_JSON: ".json",
    ArtifactFormat.XGBOOST_UBJ: ".ubj",
    ArtifactFormat.VARIANCE_JSON: ".json",
}
_KIND_FORMATS = {
    ArtifactKind.XGBOOST: frozenset(
        {ArtifactFormat.XGBOOST_JSON, ArtifactFormat.XGBOOST_UBJ}
    ),
    ArtifactKind.VARIANCE: frozenset({ArtifactFormat.VARIANCE_JSON}),
}
_JSON_FORMATS = frozenset({ArtifactFormat.XGBOOST_JSON, ArtifactFormat.VARIANCE_JSON})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        value[key] = item
    return value


class ModelArtifactManifest(ContractModel):
    """Immutable identity and training context for one model artifact file."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    artifact_id: str
    artifact_kind: ArtifactKind
    artifact_format: ArtifactFormat
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_names: tuple[str, ...] = Field(min_length=1)
    created_at: datetime

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_uuid(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc
        return str(parsed)

    @field_validator(
        "model_version",
        "data_version",
        "feature_version",
        "split_version",
        "schema_version",
        "dataset_id",
    )
    @classmethod
    def version_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("version fields must not be blank")
        return normalized

    @field_validator("feature_names")
    @classmethod
    def feature_names_are_unique_and_nonblank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("feature_names must not contain blank names")
        if len(value) != len(set(value)):
            raise ValueError("feature_names must be unique")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("relative_path")
    @classmethod
    def path_is_strictly_relative(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = Path(normalized)
        if not normalized or path.is_absolute() or path.drive:
            raise ValueError("relative_path must be a relative artifact path")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must not contain traversal segments")
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes.intersection(_FORBIDDEN_SUFFIXES):
            raise ValueError("relative_path uses a forbidden serialization suffix")
        return path.as_posix()

    @model_validator(mode="after")
    def kind_matches_format_and_extension(self) -> ModelArtifactManifest:
        if self.artifact_format not in _KIND_FORMATS[self.artifact_kind]:
            raise ValueError("artifact kind does not support the declared format")
        expected_suffix = _FORMAT_SUFFIXES[self.artifact_format]
        if Path(self.relative_path).suffix.lower() != expected_suffix:
            raise ValueError("artifact format extension does not match the declared format")
        return self


@dataclass(frozen=True)
class VerifiedModelArtifact:
    """A manifest-bound path whose size and digest were just verified."""

    manifest: ModelArtifactManifest
    absolute_path: Path


class ModelArtifactRegistry:
    """Thread-safe, append-only registry with digest verification on every read."""

    def __init__(self, artifact_root: Path) -> None:
        try:
            resolved_root = artifact_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("artifact root must exist") from exc
        if not resolved_root.is_dir():
            raise ValueError("artifact root must be a directory")
        self._artifact_root = resolved_root
        self._manifests: dict[str, ModelArtifactManifest] = {}
        self._lock = RLock()

    def register(self, manifest: ModelArtifactManifest) -> VerifiedModelArtifact:
        """Verify and append one manifest; artifact identities cannot be replaced."""

        validated = ModelArtifactManifest.model_validate(manifest.model_dump(mode="json"))
        verified = self._verify(validated)
        with self._lock:
            if validated.artifact_id in self._manifests:
                raise ValueError(f"artifact_id is already registered: {validated.artifact_id}")
            self._manifests[validated.artifact_id] = validated.model_copy(deep=True)
        return verified

    def resolve(
        self,
        artifact_id: str,
        *,
        model_version: str | None = None,
        data_version: str | None = None,
        feature_version: str | None = None,
        split_version: str | None = None,
    ) -> VerifiedModelArtifact:
        """Resolve by UUID and recheck bytes plus any caller-required context."""

        normalized_id = self._validated_artifact_id(artifact_id)
        with self._lock:
            stored = self._manifests.get(normalized_id)
            manifest = stored.model_copy(deep=True) if stored is not None else None
        if manifest is None:
            raise KeyError(f"unknown artifact_id: {normalized_id}")

        for field_name, received in (
            ("model_version", model_version),
            ("data_version", data_version),
            ("feature_version", feature_version),
            ("split_version", split_version),
        ):
            if received is not None and received != getattr(manifest, field_name):
                raise ValueError(f"{field_name} does not match the registered artifact")
        return self._verify(manifest)

    def read_json_object(self, artifact_id: str) -> dict[str, Any]:
        """Decode a verified JSON data artifact without model deserialization."""

        verified = self.resolve(artifact_id)
        manifest = verified.manifest
        if manifest.artifact_format not in _JSON_FORMATS:
            raise ValueError("artifact format is not JSON")
        if manifest.size_bytes > _MAX_JSON_BYTES:
            raise ValueError("JSON artifact exceeds the safe reader size limit")
        try:
            payload = verified.absolute_path.read_bytes()
            if len(payload) != manifest.size_bytes:
                raise ValueError("artifact size changed after verification")
            if hashlib.sha256(payload).hexdigest() != manifest.sha256:
                raise ValueError("artifact SHA-256 changed after verification")
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("artifact is not valid UTF-8 JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("artifact JSON payload must be a JSON object")
        return decoded

    @staticmethod
    def _validated_artifact_id(artifact_id: str) -> str:
        try:
            return str(UUID(artifact_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc

    def _verify(self, manifest: ModelArtifactManifest) -> VerifiedModelArtifact:
        path = self._resolve_artifact_path(manifest.relative_path)
        size_bytes = path.stat().st_size
        if size_bytes != manifest.size_bytes:
            raise ValueError("artifact size does not match the manifest")
        actual_sha256 = self._sha256_file(path)
        if actual_sha256 != manifest.sha256:
            raise ValueError("artifact SHA-256 does not match the manifest")
        return VerifiedModelArtifact(
            manifest=manifest.model_copy(deep=True),
            absolute_path=path,
        )

    def _resolve_artifact_path(self, relative_path: str) -> Path:
        candidate = self._artifact_root / Path(relative_path)
        if candidate.is_symlink():
            raise ValueError("artifact path must not be a symbolic link")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError("artifact path must reference an existing file") from exc
        if not resolved.is_relative_to(self._artifact_root):
            raise ValueError("artifact path must remain inside the artifact root")
        if not resolved.is_file():
            raise ValueError("artifact path must reference a regular file")
        return resolved

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_verified_xgboost_life_predictor(
    registry: ModelArtifactRegistry,
    artifact_id: str,
) -> XGBoostLifePredictor:
    """Load XGBoost native bytes only after a fresh registry verification."""

    from quanxin_life.models.xgboost import XGBoostLifePredictor

    if not isinstance(registry, ModelArtifactRegistry):
        raise TypeError("registry must be a ModelArtifactRegistry")
    verified = registry.resolve(artifact_id)
    manifest = verified.manifest
    if manifest.artifact_kind is not ArtifactKind.XGBOOST:
        raise ValueError("artifact is not an XGBoost life predictor")
    if manifest.artifact_format not in {
        ArtifactFormat.XGBOOST_JSON,
        ArtifactFormat.XGBOOST_UBJ,
    }:
        raise ValueError("artifact does not use an XGBoost native format")
    return XGBoostLifePredictor._from_verified_native_model(
        verified.absolute_path,
        dataset_id=manifest.dataset_id,
        model_version=manifest.model_version,
        feature_version=manifest.feature_version,
        split_version=manifest.split_version,
        data_version=manifest.data_version,
        cutoff_cycle=manifest.cutoff_cycle,
        feature_names=manifest.feature_names,
        artifact_id=manifest.artifact_id,
        artifact_sha256=manifest.sha256,
    )
