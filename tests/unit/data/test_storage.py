import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from quanxin_life.core import CellMetadata
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.storage import (
    CYCLE_RECORD_ARROW_SCHEMA,
    ProcessedCellManifest,
    verify_cell_artifacts,
    write_cell_artifacts,
)


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
    manifest_path = tmp_path / manifest.manifest_relative_path
    table = pq.read_table(parquet_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_payload = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())

    assert table.num_rows == 2
    assert table.schema == CYCLE_RECORD_ARROW_SCHEMA
    assert CYCLE_RECORD_ARROW_SCHEMA.names == [
        "dataset_id",
        "cell_id",
        "cycle_index",
        "sample_index",
        "time_s",
        "voltage_v",
        "current_a",
        "temperature_c",
        "charge_capacity_ah",
        "discharge_capacity_ah",
        "internal_resistance_ohm",
        "diagnostic",
        "valid",
    ]
    assert CYCLE_RECORD_ARROW_SCHEMA.field("cycle_index").type == pa.int32()
    assert CYCLE_RECORD_ARROW_SCHEMA.field("voltage_v").type == pa.float64()
    assert CYCLE_RECORD_ARROW_SCHEMA.field("diagnostic").type == pa.bool_()
    assert not CYCLE_RECORD_ARROW_SCHEMA.field("dataset_id").nullable
    assert not CYCLE_RECORD_ARROW_SCHEMA.field("valid").nullable
    assert CYCLE_RECORD_ARROW_SCHEMA.field("temperature_c").nullable
    assert table.column("temperature_c").null_count == 2
    assert metadata_payload["reference_capacity_ah"] is None
    assert manifest_payload == manifest
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


def test_rejects_records_from_another_dataset(tmp_path: Path) -> None:
    records = tuple(record.model_copy(update={"dataset_id": "HUST"}) for record in _records())

    with pytest.raises(ValueError, match="do not match metadata"):
        write_cell_artifacts(tmp_path, _metadata(), records)


@pytest.mark.parametrize("reserved", ["CON", "nul.json", "COM1", "LPT9.txt", "cell."])
def test_rejects_windows_reserved_device_names(tmp_path: Path, reserved: str) -> None:
    with pytest.raises(ValueError, match="safe file identifier"):
        write_cell_artifacts(tmp_path, _metadata(reserved), _records(reserved))


def test_verifies_complete_artifact_set_and_detects_tampering(tmp_path: Path) -> None:
    manifest = write_cell_artifacts(tmp_path, _metadata(), _records())

    verified = verify_cell_artifacts(tmp_path, manifest)

    assert verified.metadata.cell_id == "MATR_b1c0"
    assert verified.row_count == 2

    parquet_path = tmp_path / manifest.parquet_relative_path
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="parquet SHA-256 mismatch"):
        verify_cell_artifacts(tmp_path, manifest)


def test_manifest_paths_are_content_addressed_and_bound_to_cell(tmp_path: Path) -> None:
    manifest = write_cell_artifacts(tmp_path, _metadata(), _records())

    assert manifest.parquet_relative_path == (
        f"cells/MATR_b1c0/{manifest.parquet_sha256}.parquet"
    )
    assert manifest.metadata_relative_path == (
        f"metadata/MATR_b1c0/{manifest.metadata_sha256}.json"
    )
    assert manifest.manifest_relative_path == "manifests/MATR_b1c0.json"

    aliased = manifest.model_copy(
        update={"parquet_relative_path": "cells/../metadata/MATR_b1c0.json"}
    )
    with pytest.raises(ValueError, match=r"canonical|contract"):
        verify_cell_artifacts(tmp_path, aliased)


def test_failed_update_preserves_previous_committed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = write_cell_artifacts(tmp_path, _metadata(), _records())
    original_replace = os.replace

    def fail_manifest_publication(source: Path | str, destination: Path | str) -> None:
        if Path(destination).as_posix().endswith("manifests/MATR_b1c0.json"):
            raise OSError("simulated manifest publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_manifest_publication)
    changed = tuple(
        record.model_copy(update={"voltage_v": record.voltage_v + 0.05})
        for record in _records()
    )
    with pytest.raises(OSError, match="manifest publication"):
        write_cell_artifacts(tmp_path, _metadata(), changed)

    verified = verify_cell_artifacts(tmp_path, previous)
    assert verified.row_count == 2


def test_rejects_manifest_that_points_to_another_cell_artifact(tmp_path: Path) -> None:
    first = write_cell_artifacts(tmp_path, _metadata(), _records())
    second = write_cell_artifacts(tmp_path, _metadata("MATR_b1c1"), _records("MATR_b1c1"))
    crossed = first.model_copy(
        update={
            "parquet_relative_path": second.parquet_relative_path,
            "parquet_sha256": second.parquet_sha256,
        }
    )

    with pytest.raises(ValueError, match="contract"):
        verify_cell_artifacts(tmp_path, crossed)


def test_detects_metadata_tampering_and_missing_commit_marker(tmp_path: Path) -> None:
    manifest = write_cell_artifacts(tmp_path, _metadata(), _records())
    metadata_path = tmp_path / manifest.metadata_relative_path
    metadata_path.write_bytes(metadata_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        verify_cell_artifacts(tmp_path, manifest)

    manifest_path = tmp_path / manifest.manifest_relative_path
    manifest_path.unlink()
    with pytest.raises(ValueError, match="commit marker is missing"):
        verify_cell_artifacts(tmp_path, manifest)


def test_manifest_rejects_naive_created_at(tmp_path: Path) -> None:
    manifest = write_cell_artifacts(tmp_path, _metadata(), _records())

    with pytest.raises(ValidationError, match="timezone"):
        ProcessedCellManifest.model_validate(
            {**manifest.model_dump(), "created_at": datetime(2026, 7, 12)}
        )


def test_failed_publication_leaves_no_commit_marker(tmp_path: Path) -> None:
    (tmp_path / "metadata").write_text("blocks directory creation", encoding="utf-8")

    with pytest.raises(OSError):
        write_cell_artifacts(tmp_path, _metadata(), _records())

    assert not (tmp_path / "manifests" / "MATR_b1c0.json").exists()
    assert not (tmp_path / "cells" / "MATR_b1c0.parquet").exists()
