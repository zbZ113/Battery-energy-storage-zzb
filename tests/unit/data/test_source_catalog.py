import json
from pathlib import Path

import pytest

from quanxin_life.data.source_catalog import IngestionMode, SourceCatalog, SourceCatalogEntry


def test_catalog_marks_hust_pickle_payload_for_quarantine() -> None:
    catalog = SourceCatalog.load(Path("configs/data_sources.json"))

    hust = catalog.require("HUST")

    assert hust.ingestion_mode == IngestionMode.QUARANTINE_CONVERSION
    assert ".pkl" in hust.prohibited_direct_suffixes


def test_catalog_uses_safe_modes_for_matr_and_naumann() -> None:
    catalog = SourceCatalog.load(Path("configs/data_sources.json"))

    assert catalog.require("MATR").ingestion_mode == IngestionMode.HDF5
    assert catalog.require("NAUMANN_CALENDAR").ingestion_mode == IngestionMode.TABULAR


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
