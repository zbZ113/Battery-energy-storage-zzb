"""Build and verify the allow-listed MATR A100 training ZIP64 package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import IO, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file

_RAW_MATR_NAME = "2018-04-12_batchdata_updated_struct_errorcorrect.mat"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_FORBIDDEN_SUFFIXES = frozenset(
    {".joblib", ".key", ".pem", ".pickle", ".pkl", ".pt", ".pth"}
)
_FORBIDDEN_PARTS = frozenset({".git", ".venv", "__pycache__", "node_modules"})


class A100PackageFileRole(StrEnum):
    SOURCE = "source"
    RAW_MATR = "raw-matr"
    PROCESSED_DATA = "processed-data"
    SOURCE_REVISION = "source-revision"


class A100PackageFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    role: A100PackageFileRole
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class MatrA100PackageManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-a100-training-package-v1"] = (
        "matr-a100-training-package-v1"
    )
    created_at: datetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    raw_matr_sha256: Sha256
    files: tuple[A100PackageFile, ...] = Field(min_length=1)
    package_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def file_inventory_is_unique(self) -> MatrA100PackageManifest:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("package files must be unique and sorted")
        raw_files = [item for item in self.files if item.role is A100PackageFileRole.RAW_MATR]
        if len(raw_files) != 1 or raw_files[0].sha256 != self.raw_matr_sha256:
            raise ValueError("package must contain exactly one bound raw MATR file")
        return self


class MatrA100ArchiveIndex(ContractModel):
    """External whole-archive digest checked before ZIP contents are trusted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-a100-archive-index-v1"] = "matr-a100-archive-index-v1"
    archive_name: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    archive_sha256: Sha256
    package_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

    @field_validator("archive_name")
    @classmethod
    def archive_name_is_safe(cls, value: str) -> str:
        if PurePosixPath(value).name != value or not value.endswith(".zip"):
            raise ValueError("archive_name must be a simple ZIP filename")
        return value


def filter_a100_source_paths(tracked_files: tuple[str, ...]) -> tuple[str, ...]:
    """Exclude environment templates while preserving dangerous files for rejection."""

    approved: list[str] = []
    for relative in tracked_files:
        normalized = _safe_relative_path(relative)
        lowered = tuple(part.lower() for part in PurePosixPath(normalized).parts)
        if any(part.startswith(".env") for part in lowered):
            continue
        approved.append(normalized)
    return tuple(approved)


def build_matr_a100_archive(
    *,
    project_root: Path,
    output_archive: Path,
    tracked_files: tuple[str, ...],
    source_commit: str,
    raw_relative_path: str,
    raw_manifest_relative_path: str,
    processed_relative_paths: tuple[str, ...],
    created_at: datetime,
) -> MatrA100PackageManifest:
    """Create an atomic ZIP64 package from Git files and approved MATR payloads."""

    root = _regular_directory(project_root, label="project root")
    output = output_archive.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("output archive must be a new path")

    raw_relative = _safe_relative_path(raw_relative_path)
    if PurePosixPath(raw_relative).name != _RAW_MATR_NAME:
        raise ValueError("raw MATR path is not the approved 2018-04-12 batch")
    raw_path = _inside(root, raw_relative)
    manifest_path = _inside(root, _safe_relative_path(raw_manifest_relative_path))
    raw_manifest = RawFileManifest.model_validate_json(manifest_path.read_bytes())
    if raw_manifest.dataset_id != "MATR" or raw_manifest.relative_path != _RAW_MATR_NAME:
        raise ValueError("raw manifest is not bound to the approved MATR batch")
    verify_raw_file(raw_path, raw_manifest)
    _validate_matlab_v73_hdf5(raw_path)

    entries: dict[str, tuple[A100PackageFileRole, Path | bytes]] = {}
    for relative in tracked_files:
        normalized = _safe_relative_path(relative)
        _add_file(
            entries,
            normalized,
            A100PackageFileRole.SOURCE,
            _inside(root, normalized),
        )
    _add_file(entries, raw_relative, A100PackageFileRole.RAW_MATR, raw_path)
    for declared in processed_relative_paths:
        normalized = _safe_relative_path(declared)
        directory = _regular_directory(_inside(root, normalized), label="processed data root")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("symbolic links are forbidden in processed data")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                _validate_processed_file(path)
                _add_file(
                    entries,
                    relative,
                    A100PackageFileRole.PROCESSED_DATA,
                    path,
                )

    revision_payload = _canonical_json_bytes(
        {
            "schema_version": "source-revision-v1",
            "git_commit": source_commit,
            "git_dirty": False,
            "created_at": created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    _add_file(
        entries,
        "source_revision.json",
        A100PackageFileRole.SOURCE_REVISION,
        revision_payload,
    )
    files = tuple(
        A100PackageFile(
            relative_path=relative,
            role=role,
            size_bytes=_entry_size(value),
            sha256=_entry_sha256(value),
        )
        for relative, (role, value) in sorted(entries.items())
    )
    manifest_payload = {
        "schema_version": "matr-a100-training-package-v1",
        "created_at": created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "raw_matr_sha256": raw_manifest.sha256,
        "files": [item.model_dump(mode="json") for item in files],
    }
    manifest = MatrA100PackageManifest.model_validate(
        {
            **manifest_payload,
            "package_sha256": sha256_canonical(manifest_payload),
        }
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for relative, (_role, value) in sorted(entries.items()):
                _write_zip_entry(archive, relative, value)
            archive.writestr(
                "package_manifest.json",
                _canonical_json_bytes(manifest.model_dump(mode="json")),
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def verify_matr_a100_archive(archive_path: Path) -> MatrA100PackageManifest:
    """Verify exact inventory, sizes and byte hashes without extracting the ZIP."""

    path = archive_path.resolve(strict=True)
    if archive_path.is_symlink() or not path.is_file():
        raise ValueError("A100 package must be a regular non-symlinked file")
    with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("A100 package contains duplicate paths")
        for info in infos:
            _safe_relative_path(info.filename)
            if info.is_dir() or _zip_entry_is_symlink(info):
                raise ValueError("A100 package may contain regular files only")
        if "package_manifest.json" not in names:
            raise ValueError("A100 package manifest is missing")
        manifest = MatrA100PackageManifest.model_validate(
            _strict_json(archive.read("package_manifest.json"))
        )
        payload = manifest.model_dump(mode="json", exclude={"package_sha256"})
        if sha256_canonical(payload) != manifest.package_sha256:
            raise ValueError("package manifest SHA-256 does not match its contents")
        expected = {item.relative_path: item for item in manifest.files}
        if set(names) != {*expected, "package_manifest.json"}:
            raise ValueError("A100 package file inventory does not match its manifest")
        for relative, item in expected.items():
            info = archive.getinfo(relative)
            if info.file_size != item.size_bytes:
                raise ValueError(f"size mismatch in A100 package: {relative}")
            with archive.open(info, "r") as handle:
                digest = _sha256_stream(handle)
            if digest != item.sha256:
                raise ValueError(f"SHA-256 mismatch in A100 package: {relative}")
        raw = next(
            item for item in manifest.files if item.role is A100PackageFileRole.RAW_MATR
        )
        with archive.open(raw.relative_path, "r") as handle:
            _validate_matlab_v73_hdf5_stream(handle)
    return manifest


def build_matr_a100_archive_index(
    archive_path: Path,
    manifest: MatrA100PackageManifest,
) -> MatrA100ArchiveIndex:
    """Bind the complete ZIP byte stream to its reviewed internal manifest."""

    path = archive_path.resolve(strict=True)
    if archive_path.is_symlink() or not path.is_file():
        raise ValueError("A100 package must be a regular non-symlinked file")
    with path.open("rb") as handle:
        digest = _sha256_stream(handle)
    return MatrA100ArchiveIndex(
        archive_name=path.name,
        size_bytes=path.stat().st_size,
        archive_sha256=digest,
        package_sha256=manifest.package_sha256,
        source_commit=manifest.source_commit,
    )


def verify_matr_a100_archive_index(
    archive_path: Path,
    index: MatrA100ArchiveIndex,
) -> None:
    """Reject changed archive bytes before opening any ZIP member."""

    path = archive_path.resolve(strict=True)
    if archive_path.is_symlink() or not path.is_file():
        raise ValueError("A100 package must be a regular non-symlinked file")
    if path.name != index.archive_name:
        raise ValueError("A100 package filename does not match its external index")
    if path.stat().st_size != index.size_bytes:
        raise ValueError("A100 package size does not match its external index")
    with path.open("rb") as handle:
        digest = _sha256_stream(handle)
    if digest != index.archive_sha256:
        raise ValueError("A100 package SHA-256 does not match its external index")


def _add_file(
    entries: dict[str, tuple[A100PackageFileRole, Path | bytes]],
    relative: str,
    role: A100PackageFileRole,
    value: Path | bytes,
) -> None:
    if relative in entries:
        if entries[relative] != (role, value):
            raise ValueError(f"package path has conflicting sources: {relative}")
        return
    _validate_package_file(relative, value, role=role)
    entries[relative] = (role, value)


def _validate_package_file(
    relative: str,
    value: Path | bytes,
    *,
    role: A100PackageFileRole,
) -> None:
    path = PurePosixPath(relative)
    lowered = tuple(part.lower() for part in path.parts)
    if any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in lowered):
        raise ValueError("forbidden path in A100 training package")
    if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError("forbidden executable or credential artifact in A100 training package")
    if path.suffix.lower() == ".mat" and role is not A100PackageFileRole.RAW_MATR:
        raise ValueError("forbidden unapproved MAT file in A100 training package")
    if isinstance(value, Path) and (value.is_symlink() or not value.is_file()):
        raise ValueError("A100 package inputs must be regular non-symlinked files")


def _validate_processed_file(path: Path) -> None:
    if path.suffix.lower() not in {".json", ".parquet"}:
        raise ValueError("processed MATR package data must use JSON or Parquet")
    if path.suffix.lower() == ".parquet":
        with path.open("rb") as handle:
            leading = handle.read(4)
            handle.seek(-4, os.SEEK_END)
            trailing = handle.read(4)
        if leading != b"PAR1" or trailing != b"PAR1":
            raise ValueError("processed MATR package contains invalid Parquet")


def _validate_matlab_v73_hdf5(path: Path) -> None:
    with path.open("rb") as handle:
        _validate_matlab_v73_hdf5_stream(handle)


def _validate_matlab_v73_hdf5_stream(handle: IO[bytes]) -> None:
    if not handle.seekable():
        raise ValueError("MATLAB v7.3/HDF5 input must be seekable")
    handle.seek(0)
    matlab_header = handle.read(19)
    handle.seek(512)
    hdf5_signature = handle.read(8)
    handle.seek(0)
    if matlab_header != b"MATLAB 7.3 MAT-file" or hdf5_signature != _HDF5_SIGNATURE:
        raise ValueError("raw MATR file is not a MATLAB v7.3/HDF5 file")


def _write_zip_entry(archive: zipfile.ZipFile, relative: str, value: Path | bytes) -> None:
    if isinstance(value, bytes):
        archive.writestr(relative, value)
        return
    with value.open("rb") as source, archive.open(relative, "w", force_zip64=True) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _entry_size(value: Path | bytes) -> int:
    return len(value) if isinstance(value, bytes) else value.stat().st_size


def _entry_sha256(value: Path | bytes) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    with value.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("package paths must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("package path must remain relative and confined")
    if not path.parts or ":" in path.parts[0]:
        raise ValueError("package path must not contain a drive prefix")
    return path.as_posix()


def _inside(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink():
        raise ValueError("symbolic links are forbidden in A100 package inputs")
    resolved = path.resolve(strict=True)
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError("A100 package input escapes the project root")
    return resolved


def _regular_directory(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return resolved


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(payload: bytes) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("A100 package manifest must be strict UTF-8 JSON") from exc


def _zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000
