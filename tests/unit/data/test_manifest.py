import hashlib
from io import StringIO
from pathlib import Path

import pytest

import quanxin_life.data.manifest as manifest_module
from quanxin_life.data.manifest import (
    AuditedDatasetFile,
    DatasetFileAuditManifest,
    RawFileManifest,
    verify_audited_dataset_files,
    verify_raw_file,
)


def test_raw_file_manifest_verifies_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "cell.csv"
    source.write_bytes(b"cycle,capacity\n1,1.1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cell.csv",
        sha256=digest,
        source_uri="https://data.matr.io/1/",
        license_name="dataset-specific terms",
        paper_doi="10.1038/s41560-019-0356-8",
    )

    assert verify_raw_file(source, manifest) == digest


def test_stream_verifier_uses_the_open_handle_and_rewinds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cell.csv"
    source.write_bytes(b"cycle,capacity\n1,1.1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cell.csv",
        sha256=digest,
        source_uri="https://data.matr.io/1/",
        license_name="dataset-specific terms",
    )

    with source.open("rb") as handle:
        handle.seek(5)

        def path_open_is_forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("stream verification must not reopen the path")

        monkeypatch.setattr(Path, "open", path_open_is_forbidden)

        assert manifest_module.verify_raw_file_stream(handle, source, manifest) == digest
        assert handle.tell() == 0


def test_stream_verifier_rejects_an_unsafe_serialized_filename(tmp_path: Path) -> None:
    source = tmp_path / "cells.pkl"
    source.write_bytes(b"not trusted")
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cells.pkl",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_uri="https://example.invalid/cells.pkl",
        license_name="unknown",
    )

    with source.open("rb") as handle, pytest.raises(ValueError, match="unsafe serialized artifact"):
        manifest_module.verify_raw_file_stream(handle, source, manifest)


def test_stream_verifier_rejects_a_non_binary_handle(tmp_path: Path) -> None:
    source = tmp_path / "cell.csv"
    source.write_bytes(b"cycle,capacity\n1,1.1\n")
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cell.csv",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_uri="https://data.matr.io/1/",
        license_name="dataset-specific terms",
    )

    with pytest.raises(ValueError, match="binary"):
        manifest_module.verify_raw_file_stream(StringIO("not binary"), source, manifest)


def test_raw_file_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "cell.csv"
    source.write_text("changed", encoding="utf-8")
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cell.csv",
        sha256="0" * 64,
        source_uri="https://data.matr.io/1/",
        license_name="dataset-specific terms",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_raw_file(source, manifest)


def test_raw_file_manifest_rejects_unsafe_artifact_before_hashing(tmp_path: Path) -> None:
    source = tmp_path / "cells.pkl"
    source.write_bytes(b"not trusted")
    manifest = RawFileManifest(
        dataset_id="MATR",
        relative_path="cells.pkl",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_uri="https://example.invalid/cells.pkl",
        license_name="unknown",
    )

    with pytest.raises(ValueError, match="unsafe serialized artifact"):
        verify_raw_file(source, manifest)


def test_dataset_file_audit_manifest_verifies_all_declared_files(tmp_path: Path) -> None:
    first = tmp_path / "data" / "NAUMANN_CALENDAR" / "capacity.xlsx"
    second = tmp_path / "data" / "NAUMANN_CYCLE" / "capacity.mat"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"calendar")
    second.write_bytes(b"cycle")
    files = (
        AuditedDatasetFile(
            dataset_id="NAUMANN_CALENDAR",
            relative_path="data/NAUMANN_CALENDAR/capacity.xlsx",
            size_bytes=first.stat().st_size,
            sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
        ),
        AuditedDatasetFile(
            dataset_id="NAUMANN_CYCLE",
            relative_path="data/NAUMANN_CYCLE/capacity.mat",
            size_bytes=second.stat().st_size,
            sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
        ),
    )
    manifest = DatasetFileAuditManifest(
        manifest_version="naumann-audit-v1",
        source_catalog="configs/data_sources.json",
        file_count=2,
        total_size_bytes=first.stat().st_size + second.stat().st_size,
        files=files,
    )

    assert verify_audited_dataset_files(tmp_path, manifest) == tuple(
        item.sha256 for item in files
    )


def test_dataset_file_audit_manifest_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="relative_path"):
        AuditedDatasetFile(
            dataset_id="NAUMANN_CYCLE",
            relative_path="../outside.mat",
            size_bytes=1,
            sha256="a" * 64,
        )
