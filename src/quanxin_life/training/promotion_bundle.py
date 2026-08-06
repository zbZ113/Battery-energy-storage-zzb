"""Closed-world, inactive model bundles for offline inference.

The exporter deliberately accepts only an already selected ``best`` artifact.
It never loads executable model formats and never activates a route.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

_SAFE_SUFFIXES = frozenset({".safetensors", ".json", ".parquet", ".md", ".csv"})
_FORBIDDEN_SUFFIXES = frozenset(
    {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin", ".key", ".pem"}
)
_REQUIRED_FILES = frozenset(
    {
        "model_config.json",
        "input_schema.json",
        "preprocessing_manifest.json",
        "normalization.json",
        "calibration.json",
        "supported_domain.json",
        "metrics_validation.json",
        "metrics_test.json",
        "per_cell_summary.parquet",
        "inference_benchmark.json",
        "model_card.md",
    }
)
_SECRET_PATTERN = re.compile(
    rb"(?i)['\"]?(?:api[_-]?key|app[_-]?secret|token)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"
)


class PromotionBundleFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            normalized != path.as_posix()
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("bundle relative_path must remain inside the bundle")
        if path.suffix.lower() not in _SAFE_SUFFIXES:
            raise ValueError("bundle contains a forbidden format")
        return normalized


class PromotionBundleManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["promoted-model-bundle-v1"] = "promoted-model-bundle-v1"
    activation_status: Literal["VERIFIED_NOT_ACTIVATED"] = "VERIFIED_NOT_ACTIVATED"
    task: Literal[
        "RUL",
        "SOH",
        "CYCLE_LIFE",
        "FIELD_MONITORING",
        "CONDITION_DEGRADATION",
        "PARTIAL_CHARGE_FEATURE",
    ]
    family: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    created_at: datetime
    files: tuple[PromotionBundleFile, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def inventory_is_sorted_and_unique(self) -> PromotionBundleManifest:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("bundle files must be unique and sorted")
        if "artifact_manifest.json" in paths:
            raise ValueError("artifact_manifest.json is generated outside its inventory")
        return self


def export_promoted_model_bundle(
    source_root: Path,
    output_root: Path,
    *,
    task: Literal[
        "RUL",
        "SOH",
        "CYCLE_LIFE",
        "FIELD_MONITORING",
        "CONDITION_DEGRADATION",
        "PARTIAL_CHARGE_FEATURE",
    ],
    family: str,
    version: str,
    source_commit: str,
    data_version: str,
    split_version: str,
    created_at: datetime | None = None,
    checkpoint_kind: Literal["best"] = "best",
) -> PromotionBundleManifest:
    """Copy a selected checkpoint and evidence into a verified inactive bundle."""

    if checkpoint_kind != "best":
        raise ValueError("only best checkpoints may be exported")
    source = _regular_directory(source_root, "promotion source")
    if any(part.lower() in {"last", "last_checkpoint"} for part in source.parts):
        raise ValueError("promotion bundle must be exported from the best checkpoint")
    destination = output_root.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("promotion output must not be inside the source checkpoint")
    _audit_source(source)
    model_name = _model_file(source)
    missing = sorted(_REQUIRED_FILES - {path.name for path in source.iterdir() if path.is_file()})
    if missing:
        raise ValueError(f"required promotion evidence is missing: {missing}")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("promotion output must be a regular directory")
        if any(destination.iterdir()):
            raise ValueError("promotion output must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    included = sorted((*_REQUIRED_FILES, model_name))
    for name in included:
        _copy_regular(source / name, destination / name)
    files = tuple(
        PromotionBundleFile(
            relative_path=name,
            size_bytes=(destination / name).stat().st_size,
            sha256=_sha256_file(destination / name),
        )
        for name in included
    )
    manifest = PromotionBundleManifest(
        task=task,
        family=family,
        version=version,
        source_commit=source_commit,
        data_version=data_version,
        split_version=split_version,
        created_at=created_at or datetime.now(UTC),
        files=files,
        bundle_sha256=sha256_canonical(
            {
                "schema_version": "promoted-model-bundle-v1",
                "activation_status": "VERIFIED_NOT_ACTIVATED",
                "task": task,
                "family": family,
                "version": version,
                "source_commit": source_commit,
                "data_version": data_version,
                "split_version": split_version,
                "files": [item.model_dump(mode="json") for item in files],
            }
        ),
    )
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_promoted_model_bundle(root: Path, manifest: PromotionBundleManifest) -> None:
    """Verify the closed-world inventory and byte hashes of a bundle."""

    bundle = _regular_directory(root, "promotion bundle")
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    expected_paths = {item.relative_path for item in manifest.files} | {"artifact_manifest.json"}
    if actual_paths != expected_paths:
        raise ValueError("promotion bundle inventory does not match manifest")
    for item in manifest.files:
        path = _regular_file(bundle / item.relative_path, "promotion bundle file")
        if path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
            raise ValueError("promotion bundle file SHA-256 does not match manifest")
    disk = PromotionBundleManifest.model_validate_json(
        (bundle / "artifact_manifest.json").read_bytes()
    )
    if disk != manifest or disk.activation_status != "VERIFIED_NOT_ACTIVATED":
        raise ValueError("promotion bundle manifest is not VERIFIED_NOT_ACTIVATED")


def _audit_source(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("symbolic links are forbidden in promotion artifacts")
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in _FORBIDDEN_SUFFIXES:
                raise ValueError("forbidden executable or credential format in promotion source")
            if any(
                marker in part.lower()
                for part in path.parts
                for marker in (".env", "secret", "token", "credential")
            ):
                raise ValueError("forbidden secret-bearing path in promotion source")
            if suffix not in _SAFE_SUFFIXES:
                continue
            if (
                path.stat().st_size <= 16 * 1024 * 1024
                and suffix not in {".parquet", ".safetensors"}
                and _SECRET_PATTERN.search(path.read_bytes())
            ):
                raise ValueError("forbidden secret material detected in promotion source")


def _model_file(source: Path) -> str:
    candidates = [
        name
        for name in ("model.safetensors", "parameters.json")
        if (source / name).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "required model artifact must be exactly model.safetensors or parameters.json"
        )
    return candidates[0]


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return resolved


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def _copy_regular(source: Path, destination: Path) -> None:
    _regular_file(source, "promotion source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PromotionBundleFile",
    "PromotionBundleManifest",
    "export_promoted_model_bundle",
    "verify_promoted_model_bundle",
]
