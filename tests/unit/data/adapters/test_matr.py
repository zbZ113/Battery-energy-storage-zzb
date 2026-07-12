import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from quanxin_life.data.adapters.matr import load_matr_batch
from quanxin_life.data.manifest import RawFileManifest


def _write_matr_file(
    path: Path,
    *,
    include_temperature: bool = True,
    include_resistance: bool = True,
    mismatched_voltage: bool = False,
    mismatched_auxiliary: bool = False,
) -> None:
    with h5py.File(path, "w") as handle:
        batch = handle.create_group("batch")
        summary_refs = batch.create_dataset("summary", (1, 1), dtype=h5py.ref_dtype)
        cycle_refs = batch.create_dataset("cycles", (1, 1), dtype=h5py.ref_dtype)
        policy_refs = batch.create_dataset("policy_readable", (1, 1), dtype=h5py.ref_dtype)
        life_refs = batch.create_dataset("cycle_life", (1, 1), dtype=h5py.ref_dtype)

        policy = handle.create_dataset(
            "policy_0", data=np.array([ord(char) for char in "3.6C(80%)-1C"], dtype=np.uint16)
        )
        cycle_life = handle.create_dataset("cycle_life_0", data=np.array([1000.0]))
        policy_refs[0, 0] = policy.ref
        life_refs[0, 0] = cycle_life.ref

        summary = handle.create_group("summary_0")
        if include_resistance:
            summary.create_dataset("IR", data=np.array([[0.01, 0.02]]))
        summary_refs[0, 0] = summary.ref

        cycles = handle.create_group("cycles_0")
        for field in ("t", "V", "I", "Qc", "Qd"):
            refs = cycles.create_dataset(field, (2, 1), dtype=h5py.ref_dtype)
            for cycle_index in range(2):
                values = np.array([0.0, 1.0, 2.0])
                if field == "V":
                    values = np.array([3.2, 3.3]) if mismatched_voltage else values + 3.2
                elif field == "I":
                    values = values - 1.0
                elif field in {"Qc", "Qd"}:
                    values = values / 10.0
                dataset = handle.create_dataset(f"{field}_{cycle_index}", data=values)
                refs[cycle_index, 0] = dataset.ref

        if include_temperature:
            refs = cycles.create_dataset("T", (2, 1), dtype=h5py.ref_dtype)
            for cycle_index in range(2):
                dataset = handle.create_dataset(
                    f"T_{cycle_index}", data=np.array([25.0, 26.0, 27.0])
                )
                refs[cycle_index, 0] = dataset.ref
        auxiliary = cycles.create_dataset(
            "Qdlin", (1 if mismatched_auxiliary else 2, 1), dtype=h5py.ref_dtype
        )
        for cycle_index in range(auxiliary.shape[0]):
            dataset = handle.create_dataset(
                f"Qdlin_{cycle_index}", data=np.array([0.0, 0.1, 0.2])
            )
            auxiliary[cycle_index, 0] = dataset.ref
        cycle_refs[0, 0] = cycles.ref


def _manifest(path: Path, *, sha256: str | None = None) -> RawFileManifest:
    digest = sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    return RawFileManifest(
        dataset_id="MATR",
        relative_path=path.name,
        sha256=digest,
        source_uri="https://data.matr.example/batch.mat",
        license_name="dataset-specific terms",
    )


def test_loads_cells_with_provenance_and_preserves_cycle_zero(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)
    manifest = _manifest(path)

    cells = load_matr_batch(path, manifest, batch_index=3, time_unit="seconds")

    assert len(cells) == 1
    metadata, records = cells[0]
    assert metadata.cell_id == "MATR_b3c0"
    assert metadata.source_uri == manifest.source_uri
    assert metadata.source_sha256 == manifest.sha256
    assert metadata.nominal_capacity_ah == 1.1
    assert metadata.reference_capacity_ah is None
    assert metadata.protocol_id.startswith("MATR_policy_")
    assert metadata.raw_cell_id == "b3c0"
    assert metadata.protocol_description == "3.6C(80%)-1C"
    assert metadata.official_life_label == 1000
    assert metadata.official_life_label_name == "MATR_cycle_life"
    assert metadata.adapter_version == "matr-hdf5-v1.0.0"
    assert metadata.ingestion_parameters == {
        "batch_index": 3,
        "skip_cycle_zero": False,
        "time_unit": "seconds",
    }
    assert isinstance(records, tuple)
    assert {record.cycle_index for record in records} == {0, 1}
    assert records[0].cell_id == metadata.cell_id
    assert records[0].time_s == 0.0
    assert records[0].voltage_v == 3.2
    assert records[0].internal_resistance_ohm == 0.01


def test_missing_optional_temperature_and_resistance_become_none(tmp_path: Path) -> None:
    path = tmp_path / "batch.h5"
    _write_matr_file(path, include_temperature=False, include_resistance=False)

    _, records = load_matr_batch(
        path, _manifest(path), batch_index=1, time_unit="seconds"
    )[0]

    assert all(record.temperature_c is None for record in records)
    assert all(record.internal_resistance_ohm is None for record in records)


def test_rejects_inconsistent_sample_array_lengths(tmp_path: Path) -> None:
    path = tmp_path / "batch.hdf5"
    _write_matr_file(path, mismatched_voltage=True)

    with pytest.raises(ValueError, match="inconsistent sample array lengths"):
        load_matr_batch(path, _manifest(path), batch_index=1, time_unit="seconds")


def test_verifies_hash_before_opening_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_matr_batch(
            path,
            _manifest(path, sha256="0" * 64),
            batch_index=1,
            time_unit="seconds",
        )


def test_rejects_non_hdf5_suffix_even_when_file_is_valid_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "batch.bin"
    _write_matr_file(path)

    with pytest.raises(ValueError, match="unsupported MATR file suffix"):
        load_matr_batch(path, _manifest(path), batch_index=1, time_unit="seconds")


def test_cycle_zero_is_skipped_only_when_explicitly_requested(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    _, records = load_matr_batch(
        path,
        _manifest(path),
        batch_index=1,
        time_unit="seconds",
        skip_cycle_zero=True,
    )[0]

    assert {record.cycle_index for record in records} == {1}


def test_minutes_are_explicitly_converted_to_seconds(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    _, records = load_matr_batch(
        path, _manifest(path), batch_index=1, time_unit="minutes"
    )[0]

    assert records[1].time_s == 60.0


def test_time_unit_must_be_explicit(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    with pytest.raises(TypeError, match="time_unit"):
        load_matr_batch(path, _manifest(path), batch_index=1)  # type: ignore[call-arg]


def test_unconsumed_auxiliary_cycle_arrays_do_not_block_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, mismatched_auxiliary=True)

    _, records = load_matr_batch(
        path, _manifest(path), batch_index=1, time_unit="seconds"
    )[0]

    assert len(records) == 6


def test_rejects_manifest_for_another_dataset(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)
    manifest = _manifest(path).model_copy(update={"dataset_id": "HUST"})

    with pytest.raises(ValueError, match="dataset_id must be MATR"):
        load_matr_batch(path, manifest, batch_index=1, time_unit="seconds")
