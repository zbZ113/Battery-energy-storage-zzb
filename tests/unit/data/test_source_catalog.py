import json
from pathlib import Path

import pytest

from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalog, SourceCatalogEntry


def test_catalog_marks_hust_pickle_payload_for_quarantine() -> None:
    catalog = SourceCatalog.load(Path("configs/data_sources.json"))

    hust = catalog.require("HUST")

    assert hust.ingestion_mode == IngestionMode.QUARANTINE_CONVERSION
    assert ".pkl" in hust.prohibited_direct_suffixes


def test_catalog_uses_safe_modes_for_matr_and_naumann() -> None:
    catalog = SourceCatalog.load(Path("configs/data_sources.json"))

    assert catalog.require("MATR").ingestion_mode == IngestionMode.HDF5
    assert catalog.require("NAUMANN_CYCLE").ingestion_mode == IngestionMode.MATLAB
    assert catalog.require("NAUMANN_CYCLE").expected_suffixes == (".mat",)
    assert catalog.require("NAUMANN_CALENDAR").ingestion_mode == IngestionMode.TABULAR


def test_three_reviewed_matr_raw_manifests_are_registered() -> None:
    expected = {
        "2017-05-12": (
            "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
            3_025_320_241,
            "9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba",
        ),
        "2017-06-30": (
            "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
            2_007_331_155,
            "63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8",
        ),
        "2018-04-12": (
            "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
            3_236_690_412,
            "62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998",
        ),
    }
    for batch_date, (name, size_bytes, sha256) in expected.items():
        path = Path(
            "configs/data_manifests"
        ) / f"matr_{batch_date.replace('-', '_')}_batch_v1.json"
        manifest = RawFileManifest.model_validate_json(path.read_bytes())
        assert manifest.relative_path == name
        assert manifest.sha256 == sha256
        raw_path = Path("data", name)
        if raw_path.exists():
            assert raw_path.stat().st_size == size_bytes


def test_catalog_rejects_duplicate_dataset_ids(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    entry = {
        "dataset_id": "MATR",
        "version": "1",
        "source_uri": "https://example.invalid",
        "paper_uri": "https://doi.org/10.1038/s41560-019-0356-8",
        "license_status": "must_verify_before_download",
        "ingestion_mode": "hdf5",
        "expected_suffixes": [".mat"],
    }
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate dataset_id"):
        SourceCatalog.load(path)


def test_quarantine_mode_requires_prohibited_suffix() -> None:
    with pytest.raises(ValueError, match="prohibited_direct_suffixes"):
        SourceCatalogEntry(
            dataset_id="HUST",
            version="2",
            source_uri="https://example.invalid",
            paper_uri="https://doi.org/10.1039/D2EE01676A",
            license_status="must_verify_before_download",
            ingestion_mode=IngestionMode.QUARANTINE_CONVERSION,
            expected_suffixes=(".zip",),
        )
