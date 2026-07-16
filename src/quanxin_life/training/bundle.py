"""Hash-bound allow-list manifest for files sent to an A100 environment."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

_ALLOWED_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".lock",
        ".md",
        ".parquet",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {".cer", ".crt", ".joblib", ".key", ".pem", ".pickle", ".pkl", ".pt", ".pth"}
)
_FORBIDDEN_PART_MARKERS = ("credential", "secret", "token")
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        rb"(?i)['\"]?(?:api[_-]?key|app[_-]?secret)['\"]?\s*[:=]\s*"
        rb"['\"]?[A-Za-z0-9_\-]{8,}"
    ),
)
_MAX_CONTENT_SCAN_BYTES = 16 * 1024 * 1024


class TrainingBundleFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    media_type: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain inside the bundle")
        return path.as_posix()


class TrainingBundleManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "a100-training-bundle-v1"
    created_at: datetime
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    files: tuple[TrainingBundleFile, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


def build_training_bundle_manifest(
    bundle_root: Path,
    *,
    created_at: datetime,
    data_version: str,
    split_version: str,
) -> TrainingBundleManifest:
    root = _validated_root(bundle_root)
    files: list[TrainingBundleFile] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in tuple(directory_names):
            directory = current / name
            if directory.is_symlink():
                raise ValueError("symbolic link is forbidden in a training bundle")
            _validate_path_parts(directory.relative_to(root))
        for name in file_names:
            path = current / name
            relative = path.relative_to(root)
            _validate_relative_path(relative)
            if path.is_symlink():
                raise ValueError("symbolic link is forbidden in a training bundle")
            if not path.is_file():
                raise ValueError("non-regular file is forbidden in a training bundle")
            _validate_file_signature(path)
            _scan_for_secret_material(path)
            files.append(
                TrainingBundleFile(
                    relative_path=relative.as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=_sha256_file(path),
                    media_type=_media_type(path.suffix.lower()),
                )
            )
    files.sort(key=lambda item: item.relative_path)
    if not files:
        raise ValueError("training bundle must contain at least one safe file")
    bundle_hash = _bundle_hash(
        data_version=data_version,
        split_version=split_version,
        files=tuple(files),
    )
    return TrainingBundleManifest(
        created_at=created_at,
        data_version=data_version,
        split_version=split_version,
        files=tuple(files),
        bundle_sha256=bundle_hash,
    )


def verify_training_bundle_manifest(
    bundle_root: Path,
    manifest: TrainingBundleManifest,
) -> None:
    root = _validated_root(bundle_root)
    expected = {item.relative_path: item for item in manifest.files}
    actual = build_training_bundle_manifest(
        root,
        created_at=manifest.created_at,
        data_version=manifest.data_version,
        split_version=manifest.split_version,
    )
    actual_paths = {item.relative_path for item in actual.files}
    if actual_paths != set(expected):
        raise ValueError("training bundle file inventory does not match the manifest")
    for item in actual.files:
        recorded = expected[item.relative_path]
        if item.size_bytes != recorded.size_bytes:
            raise ValueError("training bundle file size does not match the manifest")
        if item.sha256 != recorded.sha256:
            raise ValueError("training bundle file SHA-256 does not match the manifest")
    if actual.bundle_sha256 != manifest.bundle_sha256:
        raise ValueError("training bundle SHA-256 does not match the manifest")


def _validated_root(bundle_root: Path) -> Path:
    try:
        root = bundle_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("training bundle root must exist") from exc
    if not root.is_dir() or bundle_root.is_symlink():
        raise ValueError("training bundle root must be a regular directory")
    return root


def _validate_relative_path(relative: Path) -> None:
    _validate_path_parts(relative)
    suffix = relative.suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        raise ValueError("forbidden executable or credential format in training bundle")
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("forbidden unapproved file format in training bundle")


def _validate_path_parts(relative: Path) -> None:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    if any(part.startswith(".env") for part in lowered_parts):
        raise ValueError("forbidden environment file in training bundle")
    if any(marker in part for part in lowered_parts for marker in _FORBIDDEN_PART_MARKERS):
        raise ValueError("forbidden secret-bearing path in training bundle")


def _scan_for_secret_material(path: Path) -> None:
    size = path.stat().st_size
    if size > _MAX_CONTENT_SCAN_BYTES or path.suffix.lower() == ".parquet":
        return
    payload = path.read_bytes()
    if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
        raise ValueError("forbidden secret material detected in training bundle")


def _validate_file_signature(path: Path) -> None:
    if path.suffix.lower() != ".parquet":
        return
    if path.stat().st_size < 8:
        raise ValueError("forbidden invalid Parquet payload in training bundle")
    with path.open("rb") as handle:
        leading = handle.read(4)
        handle.seek(-4, os.SEEK_END)
        trailing = handle.read(4)
    if leading != b"PAR1" or trailing != b"PAR1":
        raise ValueError("forbidden invalid Parquet payload in training bundle")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_hash(
    *,
    data_version: str,
    split_version: str,
    files: tuple[TrainingBundleFile, ...],
) -> str:
    return sha256_canonical(
        {
            "schema_version": "a100-training-bundle-v1",
            "data_version": data_version,
            "split_version": split_version,
            "files": [item.model_dump(mode="json") for item in files],
        }
    )


def _media_type(suffix: str) -> str:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".parquet": "application/vnd.apache.parquet",
        ".py": "text/x-python",
        ".toml": "application/toml",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }.get(suffix, "text/plain")
