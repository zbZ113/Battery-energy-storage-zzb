"""Shared fail-closed audit for externally sourced ZIP archives."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path, PurePosixPath

from quanxin_life.core.schemas import Sha256


def audit_zip_archive(
    path: Path,
    *,
    archive_sha256: Sha256,
    expected_members: tuple[str, ...] | None,
    allowed_suffixes: frozenset[str],
    ignored_members: tuple[str, ...] = (),
    max_compression_ratio: float = 10_000.0,
    max_total_uncompressed_bytes: int = 32 * 1024 * 1024 * 1024,
) -> tuple[zipfile.ZipInfo, ...]:
    """Verify archive bytes and require an exact, safe member inventory."""

    archive_path = Path(path).resolve(strict=True)
    if path.is_symlink() or not archive_path.is_file():
        raise ValueError("ZIP input must be a regular non-symlinked file")
    if _sha256_file(archive_path) != archive_sha256:
        raise ValueError("ZIP archive SHA mismatch")
    expected = set(expected_members or ())
    ignored = set(ignored_members)
    if (
        expected_members is not None
        and len(expected) != len(expected_members)
    ) or len(ignored) != len(ignored_members):
        raise ValueError("ZIP layout member paths must be unique")
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("ZIP archive contains duplicate members")
        total = 0
        observed: set[str] = set()
        selected: list[zipfile.ZipInfo] = []
        for info in infos:
            _validate_member_path(info.filename)
            if info.is_dir():
                continue
            total += info.file_size
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > max_compression_ratio:
                raise ValueError("ZIP member compression ratio exceeds policy")
            if info.filename in ignored:
                continue
            if PurePosixPath(info.filename).suffix.lower() not in allowed_suffixes:
                raise ValueError("ZIP member suffix is not approved")
            observed.add(info.filename)
            selected.append(info)
        if total > max_total_uncompressed_bytes:
            raise ValueError("ZIP archive uncompressed size exceeds policy")
        if expected_members is not None and observed != expected:
            raise ValueError("ZIP archive contains unregistered or missing members")
        return tuple(selected)


def _validate_member_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError("unsafe ZIP member path")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["audit_zip_archive"]
