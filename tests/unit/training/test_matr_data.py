from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from quanxin_life.core import CellMetadata, PredictionTarget
from quanxin_life.data.matr_pipeline import (
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.data.storage import write_cell_artifacts
from quanxin_life.training.matr_data import load_matr_cycle_life_curve_cohorts


def _write_cell(root: Path, cell_id: str, *, official_life_label: int) -> None:
    metadata = CellMetadata(
        dataset_id="MATR",
        cell_id=cell_id,
        raw_cell_id=cell_id.removeprefix("MATR_"),
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        reference_capacity_ah=1.05,
        official_life_label=official_life_label,
        official_life_label_name="cycle_life",
        source_uri="https://data.matr.io/1/",
        source_sha256="a" * 64,
        schema_version="1.0",
        adapter_version="matr-hdf5-v1.0.0",
    )
    records = tuple(
        CycleRecord(
            dataset_id="MATR",
            cell_id=cell_id,
            cycle_index=cycle,
            sample_index=sample,
            time_s=float(cycle * 10 + sample),
            voltage_v=voltage,
            current_a=-1.0,
            discharge_capacity_ah=capacity,
        )
        for cycle in range(1, 21)
        for sample, (voltage, capacity) in enumerate(
            ((3.6, 0.0), (2.8, 0.5), (2.0, 1.0))
        )
    )
    write_cell_artifacts(root, metadata, records)


def test_loads_cell_disjoint_official_life_batches_from_early_parquet_only(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "early"
    cell_ids = ("MATR_a", "MATR_b", "MATR_c", "MATR_d")
    for index, cell_id in enumerate(cell_ids):
        _write_cell(processed, cell_id, official_life_label=200 + index)

    supervision_root = tmp_path / "supervision"
    supervision_file = supervision_root / "trajectories" / "labels.parquet"
    supervision_file.parent.mkdir(parents=True)
    supervision_file.write_bytes(b"separate-supervision-fixture")
    supervision_hash = hashlib.sha256(supervision_file.read_bytes()).hexdigest()
    supervision = MatrSupervisionArtifact(
        source_report_sha256="b" * 64,
        raw_sha256="a" * 64,
        horizon_cycle=500,
        cell_count=4,
        row_count=2000,
        parquet_relative_path="trajectories/labels.parquet",
        parquet_sha256=supervision_hash,
        cells=tuple(
            MatrSupervisionCellEvidence(
                cell_id=cell_id,
                raw_cell_id=cell_id.removeprefix("MATR_"),
                official_life_label=200 + index,
                official_life_right_censored=False,
                reference_capacity_ah=1.05,
                observed_cycle_count=600,
            )
            for index, cell_id in enumerate(cell_ids)
        ),
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    split = SplitManifest(
        dataset_id="MATR",
        train=("MATR_a",),
        validation=("MATR_b",),
        calibration=("MATR_c",),
        test=("MATR_d",),
    )

    cohorts = load_matr_cycle_life_curve_cohorts(
        processed_root=processed,
        supervision_root=supervision_root,
        supervision=supervision,
        split_manifest=split,
        cutoff_cycle=20,
        voltage_min_v=2.0,
        voltage_max_v=3.6,
        voltage_grid_step_v=0.1,
    )

    assert cohorts.train.cell_ids == ("MATR_a",)
    assert cohorts.validation.cell_ids == ("MATR_b",)
    assert cohorts.calibration.cell_ids == ("MATR_c",)
    assert cohorts.test.cell_ids == ("MATR_d",)
    assert cohorts.train.target is PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert cohorts.train.curve_values.shape == (1, 21, 17)
    assert cohorts.train.observed_cycles.tolist() == [200.0]
