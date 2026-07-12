"""Static safety audit for the official HUST ZIP archive.

The pickle members are treated as opaque bytes.  This module never extracts or
deserializes them; it only validates ZIP metadata and hashes bounded streams.
"""

import hashlib
import os
import re
import stat
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from quanxin_life.data.manifest import RawFileManifest, Sha256, verify_raw_file_stream
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalogEntry

InventoryVersion = Annotated[str, StringConstraints(min_length=1)]

_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class HustArchiveLimits(BaseModel):
    """Resource and compression limits applied before and during streaming."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_archive_size: int = Field(default=8 * 1024**3, gt=0)
    max_members: int = Field(default=512, gt=0)
    max_member_uncompressed_size: int = Field(default=2 * 1024**3, gt=0)
    max_total_uncompressed_size: int = Field(default=64 * 1024**3, gt=0)
    max_compression_ratio: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    allowed_compression_methods: tuple[int, ...] = Field(
        default=(zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), min_length=1
    )
    chunk_size: int = Field(default=1024 * 1024, gt=0)


_DEFAULT_LIMITS = HustArchiveLimits()


class ExpectedHustMember(BaseModel):
    """One member in a reviewed, immutable HUST archive inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class HustArchiveMember(BaseModel):
    """Audited metadata and digest for one opaque pickle member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)
    method: int = Field(ge=0)
    crc32: int = Field(ge=0, le=0xFFFFFFFF)
    sha256: Sha256


class HustArchiveAudit(BaseModel):
    """Traceable result of a successful HUST archive safety audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    paper_uri: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    raw_relative_path: str = Field(min_length=1)
    archive_sha256: Sha256
    member_count: int = Field(ge=0)
    total_compressed_size: int = Field(ge=0)
    total_uncompressed_size: int = Field(ge=0)
    members: tuple[HustArchiveMember, ...]
    inventory_version: InventoryVersion | None = None
    ready_for_conversion: bool
    warnings: tuple[str, ...] = ()
    audited_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("audited_at")
    @classmethod
    def normalize_audit_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audited_at must include a timezone")
        return value.astimezone(UTC)


def _validate_source_binding(
    archive_path: Path,
    raw_manifest: RawFileManifest,
    source: SourceCatalogEntry,
) -> None:
    if raw_manifest.dataset_id != "HUST" or source.dataset_id != "HUST":
        raise ValueError("HUST archive audit requires dataset_id=HUST")
    if source.ingestion_mode != IngestionMode.QUARANTINE_CONVERSION:
        raise ValueError("HUST source must use quarantine_conversion ingestion mode")
    manifest_relative_path = raw_manifest.relative_path
    if (
        manifest_relative_path in {".", ".."}
        or "/" in manifest_relative_path
        or "\\" in manifest_relative_path
        or "\x00" in manifest_relative_path
        or _DRIVE_PREFIX.match(manifest_relative_path)
    ):
        raise ValueError("raw manifest relative_path must be a safe simple basename")
    if manifest_relative_path != archive_path.name:
        raise ValueError("raw manifest relative_path does not match archive path name")
    if (
        archive_path.suffix.lower() != ".zip"
        or Path(manifest_relative_path).suffix.lower() != ".zip"
    ):
        raise ValueError("HUST archive and raw manifest must use the .zip suffix")
    if ".zip" not in {suffix.lower() for suffix in source.expected_suffixes}:
        raise ValueError("HUST source catalog must approve the .zip suffix")
    prohibited_suffixes = {suffix.casefold() for suffix in source.prohibited_direct_suffixes}
    if not {".pkl", ".pickle"}.issubset(prohibited_suffixes):
        raise ValueError("HUST source catalog must prohibit direct .pkl and .pickle ingestion")
    if raw_manifest.source_uri != source.source_uri:
        raise ValueError("raw manifest source URI does not match the source catalog")
    if raw_manifest.license_name != source.license_status:
        raise ValueError("raw manifest license does not match the source catalog")


def _validate_member_path(path: str, *, directory: bool) -> None:
    if not path or "\\" in path or "\x00" in path:
        raise ValueError(f"unsafe ZIP member path: {path!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ValueError(f"unsafe ZIP member path contains control characters: {path!r}")
    if path != unicodedata.normalize("NFC", path):
        raise ValueError(f"ZIP member path is not Unicode NFC: {path!r}")
    if path.startswith("/") or _DRIVE_PREFIX.match(path):
        raise ValueError(f"absolute ZIP member path is forbidden: {path!r}")

    if directory:
        if path != "our_data/":
            raise ValueError(f"unexpected ZIP member path: {path!r}")
        return

    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "our_data" or not parts[1]:
        raise ValueError(f"ZIP member path must be our_data/<cell>.pkl: {path!r}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe ZIP member path segment: {path!r}")

    filename = parts[1]
    if not filename.endswith(".pkl") or filename == ".pkl":
        raise ValueError(f"ZIP member path must end in a named .pkl file: {path!r}")
    if filename.endswith((" ", ".")) or any(char in _WINDOWS_INVALID_CHARS for char in filename):
        raise ValueError(f"ZIP member path is unsafe on Windows: {path!r}")
    if PureWindowsPath(filename).is_reserved():
        raise ValueError(f"ZIP member path uses a Windows reserved name: {path!r}")


def _validate_regular_member(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is forbidden: {info.filename!r}")
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if info.is_dir():
        if file_type not in {0, stat.S_IFDIR}:
            raise ValueError(f"ZIP directory member has an unsafe file type: {info.filename!r}")
        return
    if file_type not in {0, stat.S_IFREG}:
        raise ValueError(f"ZIP member must be a regular file: {info.filename!r}")


def _compression_ratio(info: zipfile.ZipInfo) -> float:
    if info.file_size == 0:
        return 0.0
    if info.compress_size == 0:
        return float("inf")
    return info.file_size / info.compress_size


def _stream_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limits: HustArchiveLimits,
    streamed_total: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with archive.open(info, "r") as member:
            while chunk := member.read(limits.chunk_size):
                actual_size += len(chunk)
                if actual_size > limits.max_member_uncompressed_size:
                    raise ValueError(f"ZIP member size exceeds streaming limit: {info.filename!r}")
                if streamed_total + actual_size > limits.max_total_uncompressed_size:
                    raise ValueError("ZIP total uncompressed size exceeds streaming limit")
                digest.update(chunk)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid ZIP member stream: {info.filename!r}") from exc
    if actual_size != info.file_size:
        raise ValueError(
            f"ZIP member declared size differs from streamed size: {info.filename!r}"
        )
    return digest.hexdigest(), actual_size


def _verify_inventory(
    members: tuple[HustArchiveMember, ...],
    expected_inventory: tuple[ExpectedHustMember, ...],
) -> None:
    expected: dict[str, str] = {}
    expected_keys: set[str] = set()
    for item in expected_inventory:
        _validate_member_path(item.path, directory=False)
        key = unicodedata.normalize("NFC", item.path).casefold()
        if item.path in expected or key in expected_keys:
            raise ValueError(f"duplicate or colliding expected inventory path: {item.path!r}")
        expected[item.path] = item.sha256
        expected_keys.add(key)

    actual = {member.path: member.sha256 for member in members}
    if actual != expected:
        unknown = sorted(actual.keys() - expected.keys())
        missing = sorted(expected.keys() - actual.keys())
        mismatched = sorted(
            path for path in actual.keys() & expected.keys() if actual[path] != expected[path]
        )
        raise ValueError(
            "HUST archive inventory mismatch: "
            f"unknown={unknown}, missing={missing}, hash_mismatch={mismatched}"
        )


def audit_hust_archive(
    archive_path: Path,
    raw_manifest: RawFileManifest,
    source: SourceCatalogEntry,
    expected_inventory: tuple[ExpectedHustMember, ...] = (),
    inventory_version: str | None = None,
    limits: HustArchiveLimits = _DEFAULT_LIMITS,
) -> HustArchiveAudit:
    """Audit a HUST ZIP without extracting or deserializing any member."""
    path = Path(archive_path)
    _validate_source_binding(path, raw_manifest, source)

    audited_members: list[HustArchiveMember] = []
    seen_paths: set[str] = set()
    seen_folded_paths: set[str] = set()
    declared_total = 0
    streamed_total = 0

    with path.open("rb") as archive_handle:
        if os.fstat(archive_handle.fileno()).st_size > limits.max_archive_size:
            raise ValueError("HUST ZIP archive exceeds the outer size limit")
        archive_sha256 = verify_raw_file_stream(archive_handle, path, raw_manifest)
        if expected_inventory and not inventory_version:
            raise ValueError("an enforced HUST inventory requires inventory_version")
        if not expected_inventory and inventory_version is not None:
            raise ValueError("inventory_version requires a non-empty expected inventory")

        try:
            with zipfile.ZipFile(archive_handle, "r") as archive:
                infos = archive.infolist()
                if len(infos) > limits.max_members:
                    raise ValueError("HUST ZIP archive exceeds the member count limit")
                for info in infos:
                    # ``ZipInfo.filename`` replaces the host OS separator.  Audit the
                    # original central-directory name so a backslash cannot become a
                    # harmless-looking slash on Windows before validation.
                    member_path = info.orig_filename
                    folded_path = unicodedata.normalize("NFC", member_path).casefold()
                    if member_path in seen_paths:
                        raise ValueError(f"duplicate ZIP central-directory member: {member_path!r}")
                    if folded_path in seen_folded_paths:
                        raise ValueError(f"Unicode/casefold ZIP member collision: {member_path!r}")
                    seen_paths.add(member_path)
                    seen_folded_paths.add(folded_path)

                    is_directory = info.is_dir()
                    _validate_member_path(member_path, directory=is_directory)
                    _validate_regular_member(info)
                    if info.compress_type not in limits.allowed_compression_methods:
                        raise ValueError(
                            "unapproved ZIP compression method "
                            f"{info.compress_type}: {member_path!r}"
                        )
                    if is_directory:
                        if info.file_size != 0:
                            raise ValueError("ZIP directory member declares non-zero content")
                        continue
                    if info.file_size > limits.max_member_uncompressed_size:
                        raise ValueError(f"ZIP member size exceeds limit: {member_path!r}")
                    declared_total += info.file_size
                    if declared_total > limits.max_total_uncompressed_size:
                        raise ValueError("ZIP total uncompressed size exceeds limit")
                    if _compression_ratio(info) > limits.max_compression_ratio:
                        raise ValueError(
                            f"ZIP member compression ratio exceeds limit: {member_path!r}"
                        )

                    member_sha256, actual_size = _stream_member(
                        archive, info, limits, streamed_total
                    )
                    streamed_total += actual_size
                    audited_members.append(
                        HustArchiveMember(
                            path=member_path,
                            compressed_size=info.compress_size,
                            uncompressed_size=actual_size,
                            method=info.compress_type,
                            crc32=info.CRC,
                            sha256=member_sha256,
                        )
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError("HUST archive is not a valid ZIP central directory") from exc

    members = tuple(audited_members)
    warnings: tuple[str, ...]
    if expected_inventory:
        _verify_inventory(members, expected_inventory)
        warnings = ()
    else:
        warnings = (
            "archive has no frozen expected inventory; conversion remains disabled",
        )

    return HustArchiveAudit(
        dataset_id=source.dataset_id,
        version=source.version,
        source_uri=source.source_uri,
        paper_uri=source.paper_uri,
        license_name=source.license_status,
        raw_relative_path=raw_manifest.relative_path,
        archive_sha256=archive_sha256,
        member_count=len(members),
        total_compressed_size=sum(member.compressed_size for member in members),
        total_uncompressed_size=streamed_total,
        members=members,
        inventory_version=inventory_version,
        ready_for_conversion=bool(expected_inventory),
        warnings=warnings,
    )
