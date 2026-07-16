from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from quanxin_life.training.bundle import (
    build_training_bundle_manifest,
    verify_training_bundle_manifest,
)


def test_training_bundle_manifest_hashes_only_safe_inputs(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.json").write_text('{"cutoff": 50}', encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cells.parquet").write_bytes(b"PAR1-safe-fixture-PAR1")
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")

    manifest = build_training_bundle_manifest(
        tmp_path,
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
        data_version="hust-safe-v1",
        split_version="cell-split-v1",
    )

    assert tuple(item.relative_path for item in manifest.files) == (
        "configs/train.json",
        "data/cells.parquet",
        "train.py",
    )
    assert len(manifest.bundle_sha256) == 64
    verify_training_bundle_manifest(tmp_path, manifest)

    (tmp_path / "train.py").write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"size|SHA-256"):
        verify_training_bundle_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    "relative_path",
    [
        "raw/cell.pkl",
        "models/model.pt",
        ".env",
        "secrets/token.json",
        "deploy/private.key",
    ],
)
def test_training_bundle_rejects_executable_or_secret_material(
    tmp_path: Path,
    relative_path: str,
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not-safe-for-a100")

    with pytest.raises(ValueError, match="forbidden"):
        build_training_bundle_manifest(
            tmp_path,
            created_at=datetime(2026, 7, 16, tzinfo=UTC),
            data_version="hust-safe-v1",
            split_version="cell-split-v1",
        )


def test_training_bundle_rejects_symlinks_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symbolic links are unavailable for this Windows account")

    with pytest.raises(ValueError, match="symbolic link"):
        build_training_bundle_manifest(
            tmp_path,
            created_at=datetime(2026, 7, 16, tzinfo=UTC),
            data_version="hust-safe-v1",
            split_version="cell-split-v1",
        )


def test_training_bundle_rejects_secret_content(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"APP_SECRET": "this-must-not-leave-the-machine"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="secret material"):
        build_training_bundle_manifest(
            tmp_path,
            created_at=datetime(2026, 7, 16, tzinfo=UTC),
            data_version="hust-safe-v1",
            split_version="cell-split-v1",
        )


def test_training_bundle_rejects_file_renamed_to_parquet(tmp_path: Path) -> None:
    (tmp_path / "fake.parquet").write_bytes(b"this-is-not-parquet")

    with pytest.raises(ValueError, match="invalid Parquet"):
        build_training_bundle_manifest(
            tmp_path,
            created_at=datetime(2026, 7, 16, tzinfo=UTC),
            data_version="hust-safe-v1",
            split_version="cell-split-v1",
        )
