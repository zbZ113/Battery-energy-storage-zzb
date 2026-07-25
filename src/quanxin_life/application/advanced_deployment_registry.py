"""Managed registration of inactive, SHA-verified Advanced deployment bundles."""

from __future__ import annotations

import json
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.application.advanced_deployment_bundles import (
    AdvancedDeploymentBundleIndex,
    load_and_verify_advanced_deployment_bundle_index,
)
from quanxin_life.application.deep_model_artifacts import DeepArtifactFileRole
from quanxin_life.application.model_artifact_catalog import (
    AdvancedModelArtifactProvenance,
    AdvancedModelRouteProvenance,
    VerifiedModelArtifactMetadata,
    VerifiedModelArtifactRegistration,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.features.early_cycle_sequence import VARIABLE_NAMES


class AdvancedDeploymentBundleRegistration(ContractModel):
    """Immutable registry identity for one inactive Advanced deployment suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-deployment-registration-v1"] = (
        "advanced-deployment-registration-v1"
    )
    registry_id: Sha256
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    final_output_sha256: Sha256
    final_config_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    training_input_bundle_sha256: Sha256
    local_reconstructed_input_bundle_sha256: Sha256
    input_bundle_hashes_match: bool
    promotion_manifest_sha256: Sha256
    selection_manifest_sha256: Sha256
    route_count: int = Field(gt=0)
    artifact_count: int = Field(gt=0)
    registered_relative_root: str = Field(min_length=1)
    registered_at: datetime
    record_sha256: Sha256

    @field_validator("registered_at")
    @classmethod
    def registered_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("registered_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("registered_relative_root")
    @classmethod
    def registered_root_is_safe(cls, value: str) -> str:
        path = Path(value.replace("\\", "/"))
        if (
            path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("registered root must remain inside the registry")
        return path.as_posix()

    @model_validator(mode="after")
    def record_sha256_matches_contents(self) -> AdvancedDeploymentBundleRegistration:
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if sha256_canonical(payload) != self.record_sha256:
            raise ValueError("registration record SHA-256 does not match its contents")
        return self


@dataclass(frozen=True, slots=True)
class RegisteredAdvancedDeploymentBundle:
    """A registry record paired with a freshly reverified managed bundle."""

    record: AdvancedDeploymentBundleRegistration
    bundle_root: Path
    index: AdvancedDeploymentBundleIndex


class AdvancedDeploymentBundleRegistry:
    """Atomically copy and reverify inactive Advanced deployment candidates."""

    def __init__(self, registry_root: Path) -> None:
        registry_root.mkdir(parents=True, exist_ok=True)
        self._root = _regular_directory(
            registry_root,
            "deployment registry root",
        )
        self._bundles = self._root / "bundles"
        self._records = self._root / "records"
        self._bundles.mkdir(exist_ok=True)
        self._records.mkdir(exist_ok=True)
        self._verify_managed_directories()

    def register(
        self,
        bundle_root: Path,
        *,
        expected_manifest_sha256: str,
        registered_at: datetime,
    ) -> AdvancedDeploymentBundleRegistration:
        """Register one exact inactive bundle, or return its identical record."""

        self._verify_managed_directories()
        source = _regular_directory(bundle_root, "deployment bundle source")
        if source.is_relative_to(self._root) or self._root.is_relative_to(source):
            raise ValueError("deployment source and registry roots must not overlap")
        index = load_and_verify_advanced_deployment_bundle_index(
            source,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        # Keep managed descendants below Windows MAX_PATH in long workspaces.
        # The full digest remains the registry identity and detects prefix collisions.
        relative_root = _registered_relative_root(index.manifest_sha256)
        candidate = _registration(
            index,
            registered_relative_root=relative_root,
            registered_at=registered_at,
        )
        record_path = self._records / f"{candidate.registry_id}.json"
        destination = self._root / relative_root
        self._verify_managed_directories()
        if record_path.exists() or record_path.is_symlink():
            existing = _load_registration(record_path)
            _assert_registration_context(existing, candidate)
            self.resolve(existing.registry_id)
            return existing
        if destination.exists() or destination.is_symlink():
            recovered = load_and_verify_advanced_deployment_bundle_index(
                destination,
                expected_manifest_sha256=index.manifest_sha256,
            )
            if recovered != index:
                raise ValueError(
                    "unregistered deployment destination differs from the source"
                )
            self._verify_managed_directories()
            _write_registration_atomic(
                record_path,
                candidate,
                staging_root=self._root,
                before_replace=self._verify_managed_directories,
            )
            return candidate

        temporary = self._root / f".bundle-{uuid4().hex[:12]}.tmp"
        try:
            shutil.copytree(source, temporary, symlinks=False)
            copied = load_and_verify_advanced_deployment_bundle_index(
                temporary,
                expected_manifest_sha256=index.manifest_sha256,
            )
            if copied != index:
                raise ValueError("deployment bundle bytes changed during registration")
            self._verify_managed_directories()
            temporary.replace(destination)
            try:
                _write_registration_atomic(
                    record_path,
                    candidate,
                    staging_root=self._root,
                    before_replace=self._verify_managed_directories,
                )
            except BaseException:
                try:
                    self._verify_managed_directories()
                except ValueError:
                    pass
                else:
                    if destination.exists() and not _is_reparse_point(destination):
                        shutil.rmtree(destination)
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return candidate

    def resolve(self, registry_id: str) -> RegisteredAdvancedDeploymentBundle:
        """Resolve one managed bundle and freshly reverify every indexed byte."""

        self._verify_managed_directories()
        normalized = _sha256(registry_id, "registry_id")
        record_path = self._records / f"{normalized}.json"
        if record_path.is_symlink() or not record_path.is_file():
            raise KeyError(f"unknown deployment registry id: {normalized}")
        record = _load_registration(record_path)
        if record.registry_id != normalized:
            raise ValueError("deployment registry record identity does not match")
        expected_relative_root = _registered_relative_root(normalized)
        if record.registered_relative_root != expected_relative_root:
            raise ValueError("registered deployment root is not canonical")
        bundle_root = _inside_root(self._root, expected_relative_root)
        index = load_and_verify_advanced_deployment_bundle_index(
            bundle_root,
            expected_manifest_sha256=record.registry_id,
        )
        expected = _registration(
            index,
            registered_relative_root=record.registered_relative_root,
            registered_at=record.registered_at,
        )
        if expected != record:
            raise ValueError("deployment registry record differs from verified source")
        return RegisteredAdvancedDeploymentBundle(
            record=record,
            bundle_root=bundle_root,
            index=index,
        )

    def _verify_managed_directories(self) -> None:
        _managed_directory(self._bundles, root=self._root, name="bundles")
        _managed_directory(self._records, root=self._root, name="records")


class AdvancedDeepModelArtifactCatalogSource:
    """Expose freshly verified Deep bundles through the existing catalog port."""

    def __init__(
        self,
        registry: AdvancedDeploymentBundleRegistry,
        registry_id: str,
    ) -> None:
        self._registry = registry
        self._registry_id = _sha256(registry_id, "registry_id")

    def list_artifact_ids(self) -> tuple[str, ...]:
        registered = self._registry.resolve(self._registry_id)
        return tuple(sorted(item.artifact_id for item in registered.index.artifacts))

    def resolve_all(self) -> tuple[VerifiedModelArtifactRegistration, ...]:
        """Verify the managed bundle once and expose its exact 15 candidates."""

        registered = self._registry.resolve(self._registry_id)
        artifacts = sorted(
            registered.index.artifacts,
            key=lambda item: item.artifact_id,
        )
        if len(artifacts) != 15:
            raise ValueError("Advanced deployment registry must contain 15 artifacts")
        return tuple(
            self._resolve_registered(registered, item.artifact_id)
            for item in artifacts
        )

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration:
        normalized_id = _uuid(artifact_id)
        registered = self._registry.resolve(self._registry_id)
        return self._resolve_registered(registered, normalized_id)

    @staticmethod
    def _resolve_registered(
        registered: RegisteredAdvancedDeploymentBundle,
        normalized_id: str,
    ) -> VerifiedModelArtifactRegistration:
        artifact = next(
            (
                item
                for item in registered.index.artifacts
                if item.artifact_id == normalized_id
            ),
            None,
        )
        if artifact is None:
            raise KeyError(f"unknown Advanced artifact: {normalized_id}")
        routes = tuple(
            route
            for route in registered.index.routes
            if route.deep_artifact_id == normalized_id
        )
        if not routes:
            raise ValueError("Advanced artifact is not bound to a deployment route")
        if len({route.run_id for route in routes}) != 1:
            raise ValueError("Advanced artifact routes disagree on model version")
        feature_file = next(
            item
            for item in artifact.files
            if item.role is DeepArtifactFileRole.FEATURE_CONFIG
        )
        feature = _strict_json(
            registered.bundle_root / "artifacts" / feature_file.relative_path
        )
        _validate_feature_context(feature, artifact=artifact)
        route = sorted(routes, key=lambda item: item.role)[0]
        return VerifiedModelArtifactRegistration(
            artifact_id=artifact.artifact_id,
            artifact_format="safetensors-bundle",
            object_uri=(
                f"verified-model-artifact://{artifact.artifact_id}/bundle"
            ),
            artifact_sha256=artifact.artifact_manifest_sha256,
            model_version=route.run_id,
            manifest_uri=(
                f"verified-model-artifact://{artifact.artifact_id}/manifest"
            ),
            manifest_sha256=artifact.artifact_manifest_sha256,
            metadata=VerifiedModelArtifactMetadata(
                artifact_kind=artifact.artifact_kind.value,
                dataset_id=artifact.dataset_id,
                data_version=artifact.data_version,
                feature_version=artifact.feature_version,
                split_version=artifact.split_version,
                schema_version="deep-model-artifact-v1",
                cutoff_cycle=artifact.cutoff_cycle,
                feature_names=VARIABLE_NAMES,
                advanced_provenance=AdvancedModelArtifactProvenance(
                    deployment_bundle_manifest_sha256=(
                        registered.index.manifest_sha256
                    ),
                    source_commit=registered.index.source_commit,
                    final_output_sha256=registered.index.final_output_sha256,
                    final_config_sha256=registered.index.final_config_sha256,
                    promotion_manifest_sha256=(
                        registered.index.promotion_manifest_sha256
                    ),
                    promotion_decisions_sha256=(
                        registered.index.promotion_decisions_sha256
                    ),
                    promotion_source_evidence_sha256=(
                        registered.index.promotion_source_evidence_sha256
                    ),
                    selection_manifest_sha256=(
                        registered.index.selection_manifest_sha256
                    ),
                    training_input_bundle_sha256=(
                        registered.index.training_input_bundle_sha256
                    ),
                    local_reconstructed_input_bundle_sha256=(
                        registered.index.local_reconstructed_input_bundle_sha256
                    ),
                    input_bundle_hashes_match=(
                        registered.index.input_bundle_hashes_match
                    ),
                    candidate_config_sha256=artifact.candidate_config_sha256,
                    normalization_sha256=artifact.normalization_sha256,
                    target_scaler_context_sha256=(
                        artifact.target_scaler_context_sha256
                    ),
                    reference_library_sha256=artifact.reference_library_sha256,
                    routes=tuple(
                        AdvancedModelRouteProvenance(
                            task=AdvancedModelTask(item.task),
                            role=AdvancedModelRouteRole(item.role),
                            family=item.family,
                            candidate_id=item.candidate_id,
                            cutoff_cycle=item.cutoff_cycle,
                            seed=item.seed,
                            best_epoch=item.best_epoch,
                            run_id=item.run_id,
                            checkpoint_manifest_sha256=(
                                item.checkpoint_manifest_sha256
                            ),
                            checkpoint_model_sha256=item.checkpoint_model_sha256,
                            checkpoint_context_sha256=(
                                item.checkpoint_context_sha256
                            ),
                        )
                        for item in sorted(routes, key=lambda value: value.role)
                    ),
                ),
            ),
        )


def _registration(
    index: AdvancedDeploymentBundleIndex,
    *,
    registered_relative_root: str,
    registered_at: datetime,
) -> AdvancedDeploymentBundleRegistration:
    if registered_at.tzinfo is None or registered_at.utcoffset() is None:
        raise ValueError("registered_at must include a timezone")
    payload = {
        "schema_version": "advanced-deployment-registration-v1",
        "registry_id": index.manifest_sha256,
        "activation_status": "NOT_ACTIVATED",
        "source_commit": index.source_commit,
        "final_output_sha256": index.final_output_sha256,
        "final_config_sha256": index.final_config_sha256,
        "data_version": index.data_version,
        "split_version": index.split_version,
        "feature_version": index.feature_version,
        "training_input_bundle_sha256": index.training_input_bundle_sha256,
        "local_reconstructed_input_bundle_sha256": (
            index.local_reconstructed_input_bundle_sha256
        ),
        "input_bundle_hashes_match": index.input_bundle_hashes_match,
        "promotion_manifest_sha256": index.promotion_manifest_sha256,
        "selection_manifest_sha256": index.selection_manifest_sha256,
        "route_count": len(index.routes),
        "artifact_count": len(index.artifacts),
        "registered_relative_root": registered_relative_root,
        "registered_at": (
            registered_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        ),
    }
    return AdvancedDeploymentBundleRegistration.model_validate(
        {**payload, "record_sha256": sha256_canonical(payload)}
    )


def _assert_registration_context(
    existing: AdvancedDeploymentBundleRegistration,
    candidate: AdvancedDeploymentBundleRegistration,
) -> None:
    excluded = {"registered_at", "record_sha256"}
    if existing.model_dump(exclude=excluded) != candidate.model_dump(
        exclude=excluded
    ):
        raise ValueError("deployment bundle registration context conflicts")


def _validate_feature_context(
    feature: dict[str, Any],
    *,
    artifact: Any,
) -> None:
    expected = {
        "dataset_id": artifact.dataset_id,
        "data_version": artifact.data_version,
        "split_version": artifact.split_version,
        "feature_version": artifact.feature_version,
        "cutoff_cycle": artifact.cutoff_cycle,
        "candidate_config_sha256": artifact.candidate_config_sha256,
        "normalization_sha256": artifact.normalization_sha256,
    }
    if any(feature.get(name) != value for name, value in expected.items()):
        raise ValueError("Advanced feature context differs from its deployment index")
    variable_names = feature.get("variable_names")
    if variable_names is not None and tuple(variable_names) != VARIABLE_NAMES:
        raise ValueError("Advanced feature variable names are unsupported")


def _inside_root(root: Path, relative: str) -> Path:
    candidate = root / relative
    if _is_reparse_point(candidate):
        raise ValueError("registered deployment root must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError("registered deployment root escaped its registry")
    return resolved


def _regular_directory(path: Path, label: str) -> Path:
    if _is_reparse_point(path):
        raise ValueError(f"{label} must be a regular directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return resolved


def _managed_directory(path: Path, *, root: Path, name: str) -> Path:
    if _is_reparse_point(path):
        raise ValueError("deployment registry directories contain a reparse point")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("deployment registry directories must exist") from exc
    expected = root / name
    if resolved != expected or not resolved.is_dir():
        raise ValueError("deployment registry directories escaped the registry root")
    return resolved


def _is_reparse_point(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _registered_relative_root(registry_id: str) -> str:
    normalized = _sha256(registry_id, "registry_id")
    return Path("bundles", normalized[:16]).as_posix()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    if path.is_symlink() or not path.is_file():
        raise ValueError("registered JSON evidence must be a regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("registered JSON evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("registered JSON evidence must contain an object")
    return payload


def _write_registration_atomic(
    path: Path,
    record: AdvancedDeploymentBundleRegistration,
    *,
    staging_root: Path,
    before_replace: Callable[[], None],
) -> None:
    temporary = staging_root / f".record-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        before_replace()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_registration(path: Path) -> AdvancedDeploymentBundleRegistration:
    return AdvancedDeploymentBundleRegistration.model_validate(_strict_json(path))


def _sha256(value: str, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise KeyError("unknown Advanced artifact") from exc


__all__ = [
    "AdvancedDeepModelArtifactCatalogSource",
    "AdvancedDeploymentBundleRegistration",
    "AdvancedDeploymentBundleRegistry",
    "RegisteredAdvancedDeploymentBundle",
]
