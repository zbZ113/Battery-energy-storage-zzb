from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quanxin_life.training.a100_package import (
    build_matr_a100_archive,
    build_matr_a100_archive_index,
    verify_matr_a100_archive,
    verify_matr_a100_archive_index,
)

RAW_NAME = "2018-04-12_batchdata_updated_struct_errorcorrect.mat"


def _write_matlab_v73(path: Path, payload: bytes = b"approved-matr") -> str:
    header = bytearray(520)
    header[:19] = b"MATLAB 7.3 MAT-file"
    header[512:520] = b"\x89HDF\r\n\x1a\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "configs" / "data_manifests").mkdir(parents=True)
    raw = root / "data" / RAW_NAME
    raw_sha256 = _write_matlab_v73(raw)
    manifest = {
        "dataset_id": "MATR",
        "relative_path": RAW_NAME,
        "sha256": raw_sha256,
        "source_uri": "https://data.matr.io/approved",
        "license_name": "reviewed-for-training-package",
        "license_uri": None,
        "paper_doi": "https://doi.org/10.1038/s41560-019-0356-8",
        "downloaded_at": "2026-07-17T02:22:49Z",
    }
    (root / "configs" / "data_manifests" / "matr.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    processed = root / "data" / "processed" / "MATR"
    (processed / "early").mkdir(parents=True)
    (processed / "early" / "cell.parquet").write_bytes(b"PAR1fixturePAR1")
    (processed / "supervision").mkdir()
    (processed / "supervision" / "labels.json").write_text("{}", encoding="utf-8")
    return root, raw_sha256


def _build(root: Path, output: Path) -> None:
    build_matr_a100_archive(
        project_root=root,
        output_archive=output,
        tracked_files=(
            "src/train.py",
            "configs/data_manifests/matr.json",
        ),
        source_commit="a" * 40,
        raw_relative_path=f"data/{RAW_NAME}",
        raw_manifest_relative_path="configs/data_manifests/matr.json",
        processed_relative_paths=(
            "data/processed/MATR/early",
            "data/processed/MATR/supervision",
        ),
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )


def test_matr_a100_archive_round_trip_is_hash_bound(tmp_path: Path) -> None:
    root, raw_sha256 = _project(tmp_path)
    output = tmp_path / "quanxin-a100.zip"

    manifest = build_matr_a100_archive(
        project_root=root,
        output_archive=output,
        tracked_files=(
            "src/train.py",
            "configs/data_manifests/matr.json",
        ),
        source_commit="a" * 40,
        raw_relative_path=f"data/{RAW_NAME}",
        raw_manifest_relative_path="configs/data_manifests/matr.json",
        processed_relative_paths=(
            "data/processed/MATR/early",
            "data/processed/MATR/supervision",
        ),
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )

    verified = verify_matr_a100_archive(output)
    assert verified.package_sha256 == manifest.package_sha256
    assert verified.raw_matr_sha256 == raw_sha256
    assert {item.relative_path for item in verified.files} == {
        "configs/data_manifests/matr.json",
        f"data/{RAW_NAME}",
        "data/processed/MATR/early/cell.parquet",
        "data/processed/MATR/supervision/labels.json",
        "source_revision.json",
        "src/train.py",
    }

    index = build_matr_a100_archive_index(output, manifest)
    verify_matr_a100_archive_index(output, index)
    output.write_bytes(output.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match=r"size|SHA-256"):
        verify_matr_a100_archive_index(output, index)


def test_matr_a100_archive_rejects_raw_mat_hash_mismatch(tmp_path: Path) -> None:
    root, _ = _project(tmp_path)
    raw = root / "data" / RAW_NAME
    raw.write_bytes(raw.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        _build(root, tmp_path / "rejected.zip")


def test_matr_a100_archive_rejects_mat_without_v73_hdf5_signature(tmp_path: Path) -> None:
    root, _ = _project(tmp_path)
    raw = root / "data" / RAW_NAME
    raw.write_bytes(b"not an HDF5 MATLAB file")
    manifest_path = root / "configs" / "data_manifests" / "matr.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"MATLAB v7.3/HDF5"):
        _build(root, tmp_path / "rejected.zip")


def test_matr_a100_archive_rejects_dangerous_tracked_artifact(tmp_path: Path) -> None:
    root, _ = _project(tmp_path)
    (root / "model.pth").write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="forbidden"):
        build_matr_a100_archive(
            project_root=root,
            output_archive=tmp_path / "rejected.zip",
            tracked_files=("model.pth",),
            source_commit="a" * 40,
            raw_relative_path=f"data/{RAW_NAME}",
            raw_manifest_relative_path="configs/data_manifests/matr.json",
            processed_relative_paths=("data/processed/MATR/early",),
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
        )


def test_matr_a100_archive_verifier_rejects_changed_payload(tmp_path: Path) -> None:
    root, _ = _project(tmp_path)
    archive = tmp_path / "original.zip"
    _build(root, archive)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
        changed, "w", allowZip64=True
    ) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "src/train.py":
                payload = b"print('changed')\n"
            target.writestr(info.filename, payload)

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        verify_matr_a100_archive(changed)
