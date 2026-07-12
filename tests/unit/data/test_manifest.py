import hashlib
from pathlib import Path

import pytest

from quanxin_life.data.manifest import RawFileManifest, verify_raw_file


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
