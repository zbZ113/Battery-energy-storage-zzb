import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from quanxin_life.data.adapters.matr import iter_matr_batch, load_matr_batch
from quanxin_life.data.manifest import RawFileManifest


def _write_matr_file(
    path: Path,
    *,
    cell_count: int = 1,
    include_temperature: bool = True,
    include_resistance: bool = True,
    mismatched_voltage: bool = False,
    mismatched_auxiliary: bool = False,
    extra_policy_reference: bool = False,
    cycle_count: int = 2,
    nan_life_cell_indices: tuple[int, ...] = (),
) -> None:
    with h5py.File(path, "w") as handle:
        batch = handle.create_group("batch")
        summary_refs = batch.create_dataset("summary", (cell_count, 1), dtype=h5py.ref_dtype)
        cycle_refs = batch.create_dataset("cycles", (cell_count, 1), dtype=h5py.ref_dtype)
        policy_refs = batch.create_dataset(
            "policy_readable",
            (cell_count + int(extra_policy_reference), 1),
            dtype=h5py.ref_dtype,
        )
        life_refs = batch.create_dataset("cycle_life", (cell_count, 1), dtype=h5py.ref_dtype)

        for cell_index in range(cell_count):
            policy = handle.create_dataset(
                f"policy_{cell_index}",
                data=np.array([ord(char) for char in "3.6C(80%)-1C"], dtype=np.uint16),
            )
            cycle_life = handle.create_dataset(
                f"cycle_life_{cell_index}",
                data=np.array(
                    [float("nan") if cell_index in nan_life_cell_indices else 1000.0 + cell_index]
                ),
            )
            policy_refs[cell_index, 0] = policy.ref
            life_refs[cell_index, 0] = cycle_life.ref

            summary = handle.create_group(f"summary_{cell_index}")
            if include_resistance:
                summary.create_dataset(
                    "IR", data=np.array([np.linspace(0.01, 0.02, cycle_count)])
                )
            summary.create_dataset(
                "QDischarge",
                data=np.array(
                    [[0.9 + 0.01 * cycle_index for cycle_index in range(cycle_count)]]
                ),
            )
            summary_refs[cell_index, 0] = summary.ref

            cycles = handle.create_group(f"cycles_{cell_index}")
            for field in ("t", "V", "I", "Qc", "Qd"):
                refs = cycles.create_dataset(
                    field, (cycle_count, 1), dtype=h5py.ref_dtype
                )
                for cycle_index in range(cycle_count):
                    values = np.array([0.0, 1.0, 2.0])
                    if field == "V":
                        values = np.array([3.2, 3.3]) if mismatched_voltage else values + 3.2
                    elif field == "I":
                        values = values - 1.0
                    elif field in {"Qc", "Qd"}:
                        values = values / 10.0
                    dataset = handle.create_dataset(
                        f"{field}_{cell_index}_{cycle_index}", data=values
                    )
                    refs[cycle_index, 0] = dataset.ref

            if include_temperature:
                refs = cycles.create_dataset(
                    "T", (cycle_count, 1), dtype=h5py.ref_dtype
                )
                for cycle_index in range(cycle_count):
                    dataset = handle.create_dataset(
                        f"T_{cell_index}_{cycle_index}", data=np.array([25.0, 26.0, 27.0])
                    )
                    refs[cycle_index, 0] = dataset.ref
            auxiliary = cycles.create_dataset(
                "Qdlin",
                (1 if mismatched_auxiliary else cycle_count, 1),
                dtype=h5py.ref_dtype,
            )
            for cycle_index in range(auxiliary.shape[0]):
                dataset = handle.create_dataset(
                    f"Qdlin_{cell_index}_{cycle_index}", data=np.array([0.0, 0.1, 0.2])
                )
                auxiliary[cycle_index, 0] = dataset.ref
            cycle_refs[cell_index, 0] = cycles.ref

        if extra_policy_reference:
            policy_refs[cell_count, 0] = policy.ref


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
    assert metadata.adapter_version == "matr-hdf5-v1.2.0"
    assert metadata.ingestion_parameters == {
        "batch_index": 3,
        "max_cycle_index": None,
        "observed_cycle_count": 2,
        "official_life_right_censored": False,
        "reference_capacity_cycles": [1, 2, 3, 4, 5],
        "selected_raw_cell_ids": None,
        "skip_cycle_zero": False,
        "time_unit": "seconds",
    }
    assert isinstance(records, tuple)
    assert {record.cycle_index for record in records} == {0, 1}
    assert records[0].cell_id == metadata.cell_id
    assert records[0].time_s == 0.0
    assert records[0].voltage_v == 3.2
    assert records[0].internal_resistance_ohm == 0.01
    assert records[0].diagnostic is False
    assert all(record.diagnostic for record in records if record.cycle_index == 1)


def test_uses_fixed_post_formation_reference_capacity_window(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, cycle_count=6)

    metadata, records = load_matr_batch(
        path,
        _manifest(path),
        batch_index=3,
        time_unit="minutes",
        max_cycle_index=5,
    )[0]

    assert metadata.reference_capacity_ah == pytest.approx(0.93)
    assert metadata.ingestion_parameters["reference_capacity_cycles"] == [1, 2, 3, 4, 5]
    assert records[0].diagnostic is False
    assert all(record.diagnostic for record in records if record.cycle_index >= 1)


def test_iterates_cells_without_materializing_the_batch_tuple(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, cell_count=2)

    cells = iter_matr_batch(
        path,
        _manifest(path),
        batch_index=3,
        time_unit="seconds",
    )

    assert iter(cells) is cells
    first_metadata, _ = next(cells)
    second_metadata, _ = next(cells)
    assert (first_metadata.raw_cell_id, second_metadata.raw_cell_id) == ("b3c0", "b3c1")
    with pytest.raises(StopIteration):
        next(cells)


def test_preserves_nan_official_life_as_explicit_right_censoring(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, cycle_count=6, nan_life_cell_indices=(0,))

    metadata, _ = load_matr_batch(
        path,
        _manifest(path),
        batch_index=3,
        time_unit="minutes",
        max_cycle_index=5,
    )[0]

    assert metadata.official_life_label is None
    assert metadata.ingestion_parameters["official_life_right_censored"] is True
    assert metadata.ingestion_parameters["observed_cycle_count"] == 6


def test_selects_requested_cell_and_bounds_materialized_cycles(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, cell_count=2)

    cells = load_matr_batch(
        path,
        _manifest(path),
        batch_index=3,
        time_unit="seconds",
        selected_raw_cell_ids=("b3c1",),
        max_cycle_index=0,
    )

    assert len(cells) == 1
    metadata, records = cells[0]
    assert metadata.raw_cell_id == "b3c1"
    assert metadata.official_life_label == 1001
    assert {record.cycle_index for record in records} == {0}
    assert metadata.ingestion_parameters["selected_raw_cell_ids"] == ["b3c1"]
    assert metadata.ingestion_parameters["max_cycle_index"] == 0


def test_rejects_unknown_requested_raw_cell_id(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    with pytest.raises(ValueError, match="unknown raw MATR cell IDs"):
        load_matr_batch(
            path,
            _manifest(path),
            batch_index=3,
            time_unit="seconds",
            selected_raw_cell_ids=("b3c9",),
        )


def test_rejects_string_instead_of_a_collection_of_raw_cell_ids(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    with pytest.raises(ValueError, match="collection"):
        load_matr_batch(
            path,
            _manifest(path),
            batch_index=3,
            time_unit="seconds",
            selected_raw_cell_ids="b3c0",  # type: ignore[arg-type]
        )


def test_rejects_non_integer_cycle_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path)

    with pytest.raises(ValueError, match="max_cycle_index"):
        load_matr_batch(
            path,
            _manifest(path),
            batch_index=3,
            time_unit="seconds",
            max_cycle_index="50",  # type: ignore[arg-type]
        )


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


def test_rejects_inconsistent_batch_reference_counts(tmp_path: Path) -> None:
    path = tmp_path / "batch.mat"
    _write_matr_file(path, extra_policy_reference=True)

    with pytest.raises(ValueError, match="inconsistent cell reference counts"):
        load_matr_batch(path, _manifest(path), batch_index=1, time_unit="seconds")
