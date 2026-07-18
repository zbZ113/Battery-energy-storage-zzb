from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import quanxin_life.training.matr_data as matr_data_module
from quanxin_life.core import CellMetadata, PredictionTarget
from quanxin_life.data.matr_pipeline import (
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.data.storage import write_cell_artifacts
from quanxin_life.features.early_cycle import (
    EarlyCycleFeatureConfig,
    EarlyCycleFeatureSet,
)
from quanxin_life.training.matr_data import (
    MatrCurveCohorts,
    MatrHybridCohorts,
    MatrOfficialLifeEvidence,
    load_matr_cycle_life_curve_cohorts,
    load_matr_cycle_life_curve_cohorts_from_evidence,
    load_matr_hybrid_trajectory_cohorts,
    merge_matr_curve_cohorts,
    merge_matr_hybrid_cohorts,
    restrict_matr_split_to_cells,
)
from quanxin_life.training.tasks import CycleLifeCurveBatch, HybridTrajectoryBatch


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
            (
                (3.6, 0.0),
                (2.8, (1.05 - cycle * 0.0001) / 2),
                (2.0, 1.05 - cycle * 0.0001),
            )
        )
    )
    write_cell_artifacts(root, metadata, records)


def test_loads_cell_disjoint_official_life_batches_from_early_parquet_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "early"
    cell_ids = ("MATR_a", "MATR_b", "MATR_c", "MATR_d")
    for index, cell_id in enumerate(cell_ids):
        _write_cell(processed, cell_id, official_life_label=200 + index)

    supervision_root = tmp_path / "supervision"
    supervision_file = supervision_root / "trajectories" / "labels.parquet"
    supervision_file.parent.mkdir(parents=True)
    rows = [
        {
            "dataset_id": "MATR",
            "cell_id": cell_id,
            "cycle_index": cycle,
            "discharge_capacity_ah": 1.05 - cycle * 0.0001,
            "reference_capacity_ah": 1.05,
            "soh": (1.05 - cycle * 0.0001) / 1.05,
        }
        for cell_id in cell_ids
        for cycle in range(1, 501)
    ]
    pq.write_table(pa.Table.from_pylist(rows), supervision_file)
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

    observed_time_tolerances: list[float] = []
    original_extract = matr_data_module.extract_early_cycle_features

    def capture_early_feature_config(
        records: tuple[CycleRecord, ...],
        *,
        config: EarlyCycleFeatureConfig,
    ) -> EarlyCycleFeatureSet:
        observed_time_tolerances.append(config.time_monotonic_tolerance_s)
        return original_extract(records, config=config)

    monkeypatch.setattr(
        matr_data_module,
        "extract_early_cycle_features",
        capture_early_feature_config,
    )
    hybrid = load_matr_hybrid_trajectory_cohorts(
        processed_root=processed,
        supervision_root=supervision_root,
        supervision=supervision,
        split_manifest=split,
        cutoff_cycle=20,
    )
    assert hybrid.train.cell_ids == ("MATR_a",)
    assert hybrid.validation.cell_ids == ("MATR_b",)
    assert hybrid.train.prediction_cycles[0] == 21
    assert hybrid.train.prediction_cycles[-1] == 500
    assert hybrid.train.target_soh.shape == (1, 480)
    assert observed_time_tolerances == pytest.approx([1e-9] * len(cell_ids))


def test_scalar_life_loading_does_not_require_cycle500_trajectory_membership(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "early"
    cell_ids = ("MATR_b2c0", "MATR_b2c1", "MATR_b2c2", "MATR_b2c3")
    for index, cell_id in enumerate(cell_ids):
        _write_cell(processed, cell_id, official_life_label=300 + index)
    split = SplitManifest(
        dataset_id="MATR",
        train=("MATR_b2c0",),
        validation=("MATR_b2c1",),
        calibration=("MATR_b2c2",),
        test=("MATR_b2c3",),
    )
    evidence = tuple(
        MatrOfficialLifeEvidence(
            cell_id=cell_id,
            official_life_label=300 + index,
            official_life_right_censored=False,
            reference_capacity_ah=1.05,
        )
        for index, cell_id in enumerate(cell_ids)
    )

    cohorts = load_matr_cycle_life_curve_cohorts_from_evidence(
        processed_root=processed,
        raw_sha256="a" * 64,
        evidence=evidence,
        split_manifest=split,
        cutoff_cycle=20,
        voltage_min_v=2.0,
        voltage_max_v=3.6,
        voltage_grid_step_v=0.1,
    )

    assert cohorts.train.observed_cycles.tolist() == [300.0]
    assert cohorts.test.observed_cycles.tolist() == [303.0]


def _curve_batch(cell_id: str, *, cutoff_cycle: int = 20) -> CycleLifeCurveBatch:
    return CycleLifeCurveBatch(
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        cell_ids=(cell_id,),
        curve_values=torch.full((1, cutoff_cycle + 1, 3), 0.5),
        observed_mask=torch.ones((1, cutoff_cycle + 1), dtype=torch.bool),
        observed_cycles=torch.tensor([600.0]),
        cutoff_cycle=cutoff_cycle,
    )


def _hybrid_batch(cell_id: str, *, cutoff_cycle: int = 20) -> HybridTrajectoryBatch:
    cycles = tuple(range(cutoff_cycle + 1, 501))
    return HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=(cell_id,),
        features=torch.ones((1, 5), dtype=torch.float32),
        initial_soh=torch.tensor([0.99], dtype=torch.float32),
        target_soh=torch.full((1, len(cycles)), 0.95, dtype=torch.float32),
        prediction_cycles=cycles,
        cutoff_cycle=cutoff_cycle,
    )


def test_merges_three_batch_cohorts_without_changing_partition_membership() -> None:
    curve_components = tuple(
        MatrCurveCohorts(
            train=_curve_batch(f"MATR_b{batch}c0"),
            validation=_curve_batch(f"MATR_b{batch}c1"),
            calibration=_curve_batch(f"MATR_b{batch}c2"),
            test=_curve_batch(f"MATR_b{batch}c3"),
        )
        for batch in (1, 2, 3)
    )
    hybrid_components = tuple(
        MatrHybridCohorts(
            train=_hybrid_batch(f"MATR_b{batch}c0"),
            validation=_hybrid_batch(f"MATR_b{batch}c1"),
            calibration=_hybrid_batch(f"MATR_b{batch}c2"),
            test=_hybrid_batch(f"MATR_b{batch}c3"),
        )
        for batch in (1, 2, 3)
    )

    curves = merge_matr_curve_cohorts(curve_components)
    hybrid = merge_matr_hybrid_cohorts(hybrid_components)

    assert curves.train.cell_ids == ("MATR_b1c0", "MATR_b2c0", "MATR_b3c0")
    assert curves.train.curve_values.shape == (3, 21, 3)
    assert hybrid.test.cell_ids == ("MATR_b1c3", "MATR_b2c3", "MATR_b3c3")
    assert hybrid.test.target_soh.shape == (3, 480)


def test_merge_rejects_cross_batch_duplicate_cells() -> None:
    duplicate = MatrCurveCohorts(
        train=_curve_batch("MATR_b1c0"),
        validation=_curve_batch("MATR_b1c1"),
        calibration=_curve_batch("MATR_b1c2"),
        test=_curve_batch("MATR_b1c3"),
    )

    with pytest.raises(ValueError, match="unique"):
        merge_matr_curve_cohorts((duplicate, duplicate, duplicate))


def test_restricts_hybrid_split_without_reassigning_eligible_cells() -> None:
    split = SplitManifest(
        dataset_id="MATR",
        train=("MATR_b2c0", "MATR_b2c1"),
        validation=("MATR_b2c2", "MATR_b2c3"),
        calibration=("MATR_b2c4", "MATR_b2c5"),
        test=("MATR_b2c6", "MATR_b2c7"),
    )

    restricted = restrict_matr_split_to_cells(
        split,
        {"MATR_b2c0", "MATR_b2c2", "MATR_b2c4", "MATR_b2c6"},
    )

    assert restricted.train == ("MATR_b2c0",)
    assert restricted.validation == ("MATR_b2c2",)
    assert restricted.calibration == ("MATR_b2c4",)
    assert restricted.test == ("MATR_b2c6",)
