from __future__ import annotations

from pathlib import Path

import pytest

from quanxin_life.data.processing import (
    DatasetBuildSpec,
    DatasetProcessor,
    RawDatasetFile,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(raw: Path) -> DatasetBuildSpec:
    return DatasetBuildSpec(
        dataset_id="SYNTHETIC",
        dataset_version="v1",
        adapter_version="synthetic-v1",
        raw_files=(
            RawDatasetFile(
                relative_path="data/raw/SYNTHETIC/v1/source.csv",
                sha256=_sha256(raw),
            ),
        ),
    )


def test_second_identical_build_is_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "SYNTHETIC" / "v1" / "source.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("value\n1\n", encoding="utf-8")
    processor = DatasetProcessor(tmp_path)

    first = processor.build(_spec(raw))
    second = processor.build(_spec(raw))

    assert first.output_sha256 == second.output_sha256
    assert second.status == "SKIPPED_VALID"


def test_changed_raw_sha_requires_new_dataset_version(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "SYNTHETIC" / "v1" / "source.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("value\n1\n", encoding="utf-8")
    processor = DatasetProcessor(tmp_path)
    spec = _spec(raw)
    processor.build(spec)
    raw.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="new dataset version"):
        processor.build(spec)
