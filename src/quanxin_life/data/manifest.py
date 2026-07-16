import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from quanxin_life.data.security import assert_safe_external_data_file

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RawFileManifest(BaseModel):
    """Provenance record required before a raw file can enter processing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    source_uri: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_uri: str | None = None
    paper_doi: str | None = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("downloaded_at")
    @classmethod
    def normalize_download_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("downloaded_at must include a timezone")
        return value.astimezone(UTC)


class AuditedDatasetFile(BaseModel):
    """One immutable local filename, size and digest from a public dataset audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_portable_and_confined(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use portable forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain below the declared repository root")
        if not path.parts or ":" in path.parts[0]:
            raise ValueError("relative_path must not contain a drive or absolute prefix")
        return value


class DatasetFileAuditManifest(BaseModel):
    """Versioned collection of public dataset payloads audited outside Git."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(min_length=1)
    source_catalog: str = Field(min_length=1)
    file_count: int = Field(gt=0)
    total_size_bytes: int = Field(gt=0)
    files: tuple[AuditedDatasetFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def summary_matches_entries(self) -> "DatasetFileAuditManifest":
        if self.file_count != len(self.files):
            raise ValueError("file_count must match the number of audited files")
        if self.total_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("total_size_bytes must match the audited files")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("audited file relative_path values must be unique")
        return self


def load_dataset_file_audit_manifest(path: Path) -> DatasetFileAuditManifest:
    """Load a strict UTF-8 JSON audit manifest without touching dataset payloads."""

    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("dataset file audit manifest must use a .json file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"dataset file audit manifest does not exist: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset file audit manifest must contain valid UTF-8 JSON") from exc
    return DatasetFileAuditManifest.model_validate(payload)


def verify_audited_dataset_files(
    repository_root: Path,
    manifest: DatasetFileAuditManifest,
) -> tuple[str, ...]:
    """Recompute every declared file size and digest without parsing its contents."""

    root = Path(repository_root).resolve(strict=True)
    validated = DatasetFileAuditManifest.model_validate(manifest.model_dump(mode="json"))
    verified: list[str] = []
    for item in validated.files:
        candidate = root.joinpath(*PurePosixPath(item.relative_path).parts)
        if candidate.is_symlink():
            message = f"audited dataset file must not be a symbolic link: {item.relative_path}"
            raise ValueError(message)
        resolved = candidate.resolve(strict=True)
        if root not in resolved.parents:
            raise ValueError(f"audited dataset file escapes repository root: {item.relative_path}")
        if not resolved.is_file():
            raise ValueError(f"audited dataset path is not a file: {item.relative_path}")
        if resolved.stat().st_size != item.size_bytes:
            raise ValueError(f"size mismatch for audited dataset file: {item.relative_path}")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != item.sha256:
            raise ValueError(f"SHA-256 mismatch for audited dataset file: {item.relative_path}")
        verified.append(actual)
    return tuple(verified)


def verify_raw_file_stream(handle: BinaryIO, path: Path, manifest: RawFileManifest) -> str:
    """Verify a raw file from its already-open binary stream and rewind it."""
    assert_safe_external_data_file(path)
    if not handle.readable() or not handle.seekable():
        raise ValueError("raw data stream must be readable and seekable")

    digest = hashlib.sha256()
    handle.seek(0)
    try:
        while True:
            chunk = handle.read(1024 * 1024)
            if not isinstance(chunk, bytes):
                raise ValueError("raw data stream must be binary")
            if not chunk:
                break
            digest.update(chunk)
        actual = digest.hexdigest()
        if actual != manifest.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {manifest.relative_path}: "
                f"expected {manifest.sha256}, got {actual}"
            )
        return actual
    finally:
        handle.seek(0)


def verify_raw_file(path: Path, manifest: RawFileManifest) -> str:
    """Reject executable serialization and verify immutable raw-file provenance."""
    assert_safe_external_data_file(path)
    if not path.is_file():
        raise ValueError(f"raw data file does not exist: {path}")
    with path.open("rb") as handle:
        return verify_raw_file_stream(handle, path, manifest)
