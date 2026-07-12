import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from quanxin_life.core import CellMetadata
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.storage import write_cell_artifacts


def _metadata(cell_id: str = "MATR_b1c0") -> CellMetadata:
    return CellMetadata(
        dataset_id="MATR",
        cell_id=cell_id,
        raw_cell_id="b1c0",
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        source_uri="https://data.matr.io/1/",
        source_sha256="a" * 64,
        schema_version="1.0",
        adapter_version="matr-hdf5-v1.0.0",
    )


def _records(cell_id: str = "MATR_b1c0") -> tuple[CycleRecord, ...]:
    return tuple(
        CycleRecord(
            dataset_id="MATR",
            cell_id=cell_id,
            cycle_index=1,
            sample_index=index,
            time_s=float(index),
            voltage_v=3.2 + index * 0.1,
            current_a=-1.0,
            temperature_c=None,
            discharge_capacity_ah=index * 0.1,
        )
        for index in range(2)
    )


def test_writes_parquet_metadata_and_hashed_manifest(tmp_path: Path) -> None:
    manifest = write_cell_artifacts(tmp_path, _metadata(), _records())

    parquet_path = tmp_path / manifest.parquet_relative_path
    metadata_path = tmp_path / manifest.metadata_relative_path
    table = pq.read_table(parquet_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert table.num_rows == 2
    assert table.schema.field("voltage_v").type.bit_width == 64
    assert table.column("temperature_c").null_count == 2
    assert metadata_payload["reference_capacity_ah"] is None
    assert hashlib.sha256(parquet_path.read_bytes()).hexdigest() == manifest.parquet_sha256
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == manifest.metadata_sha256


def test_rejects_path_traversal_cell_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe file identifier"):
        write_cell_artifacts(tmp_path, _metadata("../escape"), _records("../escape"))


def test_rejects_records_from_another_cell(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match metadata"):
        write_cell_artifacts(tmp_path, _metadata(), _records("MATR_b1c1"))


def test_rejects_empty_record_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one cycle record"):
        write_cell_artifacts(tmp_path, _metadata(), ())
