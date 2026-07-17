import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import h5py
import numpy as np

from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.matr_pipeline import (
    audit_matr_eol80_labels,
    build_matr_life_strata,
    build_matr_split_evidence,
    build_matr_supervision_artifact,
    convert_matr_batch,
)
from quanxin_life.data.storage import ProcessedCellManifest, verify_cell_artifacts


def _write_matr_file(path: Path, *, cell_count: int = 1, cycle_count: int = 6) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "batch_date",
            data=np.array([ord(char) for char in "2018-04-12"], dtype=np.uint16),
        )
        batch = handle.create_group("batch")
        summary_refs = batch.create_dataset("summary", (cell_count, 1), dtype=h5py.ref_dtype)
        cycle_refs = batch.create_dataset("cycles", (cell_count, 1), dtype=h5py.ref_dtype)
        policy_refs = batch.create_dataset(
            "policy_readable", (cell_count, 1), dtype=h5py.ref_dtype
        )
        life_refs = batch.create_dataset("cycle_life", (cell_count, 1), dtype=h5py.ref_dtype)
        for cell_index in range(cell_count):
            policy = handle.create_dataset(
                f"policy_{cell_index}",
                data=np.array([ord(char) for char in "3.6C(80%)-1C"], dtype=np.uint16),
            )
            life = handle.create_dataset(
                f"life_{cell_index}", data=np.array([1000.0 + cell_index])
            )
            policy_refs[cell_index, 0] = policy.ref
            life_refs[cell_index, 0] = life.ref

            summary = handle.create_group(f"summary_{cell_index}")
            summary.create_dataset("IR", data=np.array([np.linspace(0.01, 0.02, cycle_count)]))
            summary.create_dataset(
                "QDischarge",
                data=np.array([[0.9 + 0.01 * cycle for cycle in range(cycle_count)]]),
            )
            summary_refs[cell_index, 0] = summary.ref

            cycles = handle.create_group(f"cycles_{cell_index}")
            for field in ("t", "V", "I", "Qc", "Qd", "T"):
                references = cycles.create_dataset(
                    field, (cycle_count, 1), dtype=h5py.ref_dtype
                )
                for cycle_index in range(cycle_count):
                    values = np.array([0.0, 1.0, 2.0])
                    if field == "V":
                        values += 3.2
                    elif field == "I":
                        values -= 1.0
                    elif field in {"Qc", "Qd"}:
                        values = values / 10.0
                    elif field == "T":
                        values += 25.0
                    dataset = handle.create_dataset(
                        f"{field}_{cell_index}_{cycle_index}", data=values
                    )
                    references[cycle_index, 0] = dataset.ref
            cycle_refs[cell_index, 0] = cycles.ref


def _manifest(path: Path) -> RawFileManifest:
    return RawFileManifest(
        dataset_id="MATR",
        relative_path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_uri="https://data.matr.example/batch.mat",
        license_name="dataset-specific terms",
    )


def test_converts_matr_batch_cell_by_cell_and_returns_verified_evidence(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "batch.mat"
    _write_matr_file(raw_path, cell_count=2, cycle_count=6)
    manifest: RawFileManifest = _manifest(raw_path)
    output_root = tmp_path / "processed"

    report = convert_matr_batch(
        raw_path=raw_path,
        raw_manifest=manifest,
        output_root=output_root,
        batch_index=3,
        batch_date=date(2018, 4, 12),
        time_unit="minutes",
        max_cycle_index=5,
        created_at=datetime(2026, 7, 17, 3, 0, tzinfo=UTC),
    )

    assert report.schema_version == "matr-batch-conversion-v1"
    assert report.batch_index == 3
    assert report.batch_date == date(2018, 4, 12)
    assert report.raw_sha256 == manifest.sha256
    assert report.source_uri == manifest.source_uri
    assert report.license_name == manifest.license_name
    assert report.cell_count == 2
    assert report.total_row_count == 36
    assert report.quality_issue_counts == {}
    assert tuple(cell.cell_id for cell in report.cells) == ("MATR_b3c0", "MATR_b3c1")
    assert all(cell.reference_capacity_ah == 0.93 for cell in report.cells)
    assert all(cell.official_life_right_censored is False for cell in report.cells)

    for cell in report.cells:
        processed_manifest = ProcessedCellManifest.model_validate_json(
            (output_root / cell.manifest_relative_path).read_bytes()
        )
        verified = verify_cell_artifacts(output_root, processed_manifest)
        assert verified.row_count == cell.row_count
        assert processed_manifest.parquet_sha256 == cell.parquet_sha256


def test_rejects_batch_date_that_does_not_match_hdf5_metadata(tmp_path: Path) -> None:
    raw_path = tmp_path / "batch.mat"
    _write_matr_file(raw_path, cycle_count=6)

    try:
        convert_matr_batch(
            raw_path=raw_path,
            raw_manifest=_manifest(raw_path),
            output_root=tmp_path / "processed",
            batch_index=3,
            batch_date=date(2018, 4, 13),
            time_unit="minutes",
            max_cycle_index=5,
            created_at=datetime(2026, 7, 17, 3, 0, tzinfo=UTC),
        )
    except ValueError as exc:
        assert "batch_date" in str(exc)
    else:  # pragma: no cover - documents the required fail-closed contract
        raise AssertionError("mismatched MATR batch_date was accepted")


def test_builds_protocol_and_life_strata_without_treating_censoring_as_an_event(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "batch.mat"
    _write_matr_file(raw_path, cell_count=2, cycle_count=6)
    report = convert_matr_batch(
        raw_path=raw_path,
        raw_manifest=_manifest(raw_path),
        output_root=tmp_path / "processed",
        batch_index=3,
        batch_date=date(2018, 4, 12),
        time_unit="minutes",
        max_cycle_index=5,
        created_at=datetime(2026, 7, 17, 3, 0, tzinfo=UTC),
    )
    first, second = report.cells
    censored = second.model_copy(
        update={
            "cell_id": "MATR_b3c2",
            "raw_cell_id": "b3c2",
            "official_life_label": None,
            "official_life_right_censored": True,
        }
    )

    strata = build_matr_life_strata((first, second, censored), quantile_count=2)

    assert strata[first.cell_id].endswith("|life-q1")
    assert strata[second.cell_id].endswith("|life-q2")
    assert strata[censored.cell_id].endswith("|right-censored")

    split_evidence = build_matr_split_evidence(
        report.model_copy(update={"cells": (first, second, censored), "cell_count": 3}),
        split_version="matr-b3-cell-split-v1",
        created_at=datetime(2026, 7, 17, 3, 30, tzinfo=UTC),
        quantile_count=2,
    )
    assert set(split_evidence.split_manifest.all_cells) == {
        first.cell_id,
        second.cell_id,
        censored.cell_id,
    }
    assert split_evidence.cell_strata == strata

    label_audit = audit_matr_eol80_labels(
        raw_path=raw_path,
        raw_manifest=_manifest(raw_path),
        conversion_report=report,
        created_at=datetime(2026, 7, 17, 3, 45, tzinfo=UTC),
    )
    assert label_audit.unified_event_count == 0
    assert label_audit.unified_right_censored_count == 2
    assert all(cell.unified_eol80_cycle is None for cell in label_audit.cells)
    assert all(cell.minimum_observed_soh > 0.8 for cell in label_audit.cells)


def test_builds_cycle_500_supervision_separately_from_early_samples(tmp_path: Path) -> None:
    raw_path = tmp_path / "batch.mat"
    _write_matr_file(raw_path, cell_count=2, cycle_count=8)
    manifest = _manifest(raw_path)
    early_root = tmp_path / "early-inputs"
    report = convert_matr_batch(
        raw_path=raw_path,
        raw_manifest=manifest,
        output_root=early_root,
        batch_index=3,
        batch_date=date(2018, 4, 12),
        time_unit="minutes",
        max_cycle_index=5,
        created_at=datetime(2026, 7, 17, 3, 0, tzinfo=UTC),
    )

    supervision = build_matr_supervision_artifact(
        raw_path=raw_path,
        raw_manifest=manifest,
        conversion_report=report,
        output_root=tmp_path / "supervision",
        horizon_cycle=7,
        created_at=datetime(2026, 7, 17, 4, 0, tzinfo=UTC),
    )

    assert supervision.schema_version == "matr-supervision-v1"
    assert supervision.horizon_cycle == 7
    assert supervision.cell_count == 2
    assert supervision.row_count == 14
    assert supervision.parquet_relative_path.startswith("trajectories/")
    assert not (early_root / supervision.parquet_relative_path).exists()

    table = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
        tmp_path / "supervision" / supervision.parquet_relative_path
    )
    assert table.column_names == [
        "dataset_id",
        "cell_id",
        "cycle_index",
        "discharge_capacity_ah",
        "reference_capacity_ah",
        "soh",
    ]
    assert set(table.column("cycle_index").to_pylist()) == set(range(1, 8))
