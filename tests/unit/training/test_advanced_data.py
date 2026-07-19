from __future__ import annotations

import hashlib
from dataclasses import fields
from datetime import UTC, date, datetime
from inspect import signature
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from pydantic import BaseModel

import quanxin_life.training.advanced_data as advanced_data_module
from quanxin_life.core import CellMetadata
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.matr_multibatch import (
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    MatrTrajectoryExclusion,
    combine_matr_batch_splits,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrCellConversionEvidence,
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.data.storage import write_cell_artifacts
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig
from quanxin_life.training.advanced_data import (
    AdvancedFinalMatrData,
    AdvancedSelectionMatrData,
    _assemble_advanced_matr_data,
    _load_early_sequence,
    _load_supervision_rows,
    _validate_component_contract,
    load_advanced_matr_final_data,
    load_advanced_matr_selection_data,
)


def _component_contract() -> tuple[
    MatrBatchArtifactReference,
    MatrBatchConversionReport,
    SplitManifest,
    MatrTrajectoryEligibilityAudit,
    MatrSupervisionArtifact,
]:
    cells = (
        MatrCellConversionEvidence(
            cell_id="MATR_b1c0",
            raw_cell_id="b1c0",
            official_life_label=700,
            official_life_right_censored=False,
            protocol_id="p1",
            reference_capacity_ah=1.1,
            row_count=1000,
            cycle_count=501,
            quality_issue_counts={},
            manifest_relative_path="MATR_b1c0/manifest.json",
            parquet_sha256="1" * 64,
            metadata_sha256="2" * 64,
        ),
        MatrCellConversionEvidence(
            cell_id="MATR_b1c1",
            raw_cell_id="b1c1",
            official_life_label=None,
            official_life_right_censored=True,
            protocol_id="p2",
            reference_capacity_ah=1.2,
            row_count=800,
            cycle_count=400,
            quality_issue_counts={},
            manifest_relative_path="MATR_b1c1/manifest.json",
            parquet_sha256="3" * 64,
            metadata_sha256="4" * 64,
        ),
    )
    conversion = MatrBatchConversionReport(
        batch_index=1,
        batch_date=date(2017, 5, 12),
        raw_relative_path="data/batch1.mat",
        raw_size_bytes=100,
        raw_sha256="a" * 64,
        source_uri="https://example.invalid/matr",
        license_name="research",
        adapter_version="matr-v1",
        time_unit="seconds",
        max_cycle_index=501,
        cell_count=2,
        total_row_count=1800,
        quality_issue_counts={},
        cells=cells,
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    split = SplitManifest(
        dataset_id="MATR",
        train=("MATR_b1c0",),
        validation=("MATR_b1c1",),
        calibration=(),
        test=(),
    )
    eligibility = MatrTrajectoryEligibilityAudit(
        batch_index=1,
        horizon_cycle=500,
        eligible_cell_ids=("MATR_b1c0",),
        excluded=(
            MatrTrajectoryExclusion(
                cell_id="MATR_b1c1",
                observed_cycle_count=400,
            ),
        ),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    supervision = MatrSupervisionArtifact(
        source_report_sha256=sha256_canonical(conversion.model_dump(mode="json")),
        raw_sha256="a" * 64,
        horizon_cycle=500,
        cell_count=1,
        row_count=500,
        parquet_relative_path="supervision.parquet",
        parquet_sha256="5" * 64,
        cells=(
            MatrSupervisionCellEvidence(
                cell_id="MATR_b1c0",
                raw_cell_id="b1c0",
                official_life_label=700,
                official_life_right_censored=False,
                reference_capacity_ah=1.1,
                observed_cycle_count=501,
            ),
        ),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    component = MatrBatchArtifactReference(
        batch_index=1,
        batch_date=date(2017, 5, 12),
        raw_relative_path="data/batch1.mat",
        raw_manifest="configs/batch1.json",
        raw_sha256="a" * 64,
        processed_root="processed/batch1",
        conversion_report="reports/batch1-conversion.json",
        conversion_report_sha256="6" * 64,
        split_manifest="splits/batch1.json",
        split_manifest_sha256="7" * 64,
        supervision_root="supervision/batch1",
        supervision_report="reports/batch1-supervision.json",
        supervision_report_sha256="8" * 64,
        eligibility_report="reports/batch1-eligibility.json",
        eligibility_report_sha256="9" * 64,
        cell_count=2,
        scalar_label_count=1,
        hybrid_eligible_count=1,
        hybrid_excluded_count=1,
    )
    return component, conversion, split, eligibility, supervision


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("raw_cell_id", "b1c9"),
        ("official_life_label", 701),
        ("official_life_right_censored", True),
        ("reference_capacity_ah", 1.01),
        ("observed_cycle_count", 500),
    ],
)
def test_component_contract_rejects_cell_level_supervision_disagreement(
    field_name: str,
    changed_value: object,
) -> None:
    component, conversion, split, eligibility, supervision = _component_contract()
    changed_cell = supervision.cells[0].model_copy(update={field_name: changed_value})
    changed_supervision = supervision.model_copy(update={"cells": (changed_cell,)})

    with pytest.raises(ValueError, match="component artifacts"):
        _validate_component_contract(
            component, conversion, split, eligibility, changed_supervision
        )


def test_component_contract_rejects_exclusion_observed_count_disagreement() -> None:
    component, conversion, split, eligibility, supervision = _component_contract()
    changed_eligibility = eligibility.model_copy(
        update={
            "excluded": (
                eligibility.excluded[0].model_copy(
                    update={"observed_cycle_count": 399}
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="component artifacts"):
        _validate_component_contract(
            component, conversion, split, changed_eligibility, supervision
        )


def test_selection_and_final_data_types_keep_heldout_partitions_closed() -> None:
    selection_fields = {field.name for field in fields(AdvancedSelectionMatrData)}
    final_fields = {field.name for field in fields(AdvancedFinalMatrData)}

    assert selection_fields == {
        "source_split",
        "scalar_train",
        "scalar_validation",
        "scalar_normalizer",
        "hybrid_train",
        "hybrid_validation",
        "hybrid_normalizer",
    }
    assert final_fields == selection_fields | {
        "scalar_calibration",
        "scalar_test",
        "hybrid_calibration",
        "hybrid_test",
    }

    expected_parameters = {
        "project_root",
        "manifest",
        "combined_split",
        "cutoff_cycle",
        "feature_version",
    }
    assert set(signature(load_advanced_matr_selection_data).parameters) == expected_parameters
    assert set(signature(load_advanced_matr_final_data).parameters) == expected_parameters


def _records(cell_id: str) -> tuple[CycleRecord, ...]:
    rows: list[CycleRecord] = []
    for cycle in (1, 20, 21):
        for sample in range(3):
            rows.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id=cell_id,
                    cycle_index=cycle,
                    sample_index=sample,
                    time_s=cycle * 100 + sample,
                    voltage_v=3.0 + sample * 0.1,
                    current_a=1.0,
                    temperature_c=25.0,
                    charge_capacity_ah=0.1 + sample * 0.1,
                )
            )
        for sample in range(3):
            rows.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id=cell_id,
                    cycle_index=cycle,
                    sample_index=3 + sample,
                    time_s=cycle * 100 + 10 + sample,
                    voltage_v=3.2 - sample * 0.1,
                    current_a=-1.0,
                    temperature_c=25.0,
                    discharge_capacity_ah=0.1 + sample * 0.1,
                )
            )
    return tuple(rows)


def test_early_sequence_loader_never_passes_future_rows_to_feature_builder(
    tmp_path: Path,
) -> None:
    raw_sha = "a" * 64
    metadata = CellMetadata(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        raw_cell_id="b1c0",
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        reference_capacity_ah=1.0,
        official_life_label=100,
        official_life_label_name="cycle_life",
        source_uri="https://example.invalid/matr",
        source_sha256=raw_sha,
        schema_version="battery-cell-v1",
    )
    manifest = write_cell_artifacts(tmp_path, metadata, _records(metadata.cell_id))
    evidence = MatrCellConversionEvidence(
        cell_id=metadata.cell_id,
        raw_cell_id="b1c0",
        official_life_label=100,
        official_life_right_censored=False,
        protocol_id="p1",
        reference_capacity_ah=1.0,
        row_count=manifest.row_count,
        cycle_count=3,
        quality_issue_counts={},
        manifest_relative_path=manifest.manifest_relative_path,
        parquet_sha256=manifest.parquet_sha256,
        metadata_sha256=manifest.metadata_sha256,
    )

    sequence = _load_early_sequence(
        processed_root=tmp_path,
        evidence=evidence,
        raw_sha256=raw_sha,
        config=MultichannelCycleConfig(cutoff_cycle=20, feature_version="advanced-v1"),
        data_version="matr-three-batch-v1",
    )

    assert sequence.cutoff_cycle == 20
    assert sequence.values.shape == (21, 2, 150, 3)
    assert bool(sequence.cycle_mask[20])


def test_selection_supervision_reader_filters_heldout_rows_before_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "supervision.parquet"
    rows: list[dict[str, object]] = []
    for cell_id, soh in (("MATR_b1c0", 0.9), ("MATR_b1c1", -1.0)):
        rows.extend(
            {
                "dataset_id": "MATR",
                "cell_id": cell_id,
                "cycle_index": cycle,
                "discharge_capacity_ah": soh,
                "reference_capacity_ah": 1.0,
                "soh": soh,
            }
            for cycle in range(1, 501)
        )
    pq.write_table(pa.Table.from_pylist(rows), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = MatrSupervisionArtifact(
        source_report_sha256="b" * 64,
        raw_sha256="a" * 64,
        horizon_cycle=500,
        cell_count=2,
        row_count=1000,
        parquet_relative_path=path.name,
        parquet_sha256=digest,
        cells=tuple(
            MatrSupervisionCellEvidence(
                cell_id=cell_id,
                raw_cell_id=f"b1c{index}",
                official_life_label=100,
                official_life_right_censored=False,
                reference_capacity_ah=1.0,
                observed_cycle_count=501,
            )
            for index, cell_id in enumerate(("MATR_b1c0", "MATR_b1c1"))
        ),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )

    trajectories = _load_supervision_rows(
        supervision_path=path,
        supervision=artifact,
        allowed_cell_ids=("MATR_b1c0",),
        allowed_evidence={
            "MATR_b1c0": MatrCellConversionEvidence(
                cell_id="MATR_b1c0",
                raw_cell_id="b1c0",
                official_life_label=100,
                official_life_right_censored=False,
                protocol_id="p1",
                reference_capacity_ah=1.0,
                row_count=1000,
                cycle_count=501,
                quality_issue_counts={},
                manifest_relative_path="MATR_b1c0/manifest.json",
                parquet_sha256="1" * 64,
                metadata_sha256="2" * 64,
            )
        },
    )

    assert tuple(trajectories) == ("MATR_b1c0",)
    assert len(trajectories["MATR_b1c0"]) == 500


@pytest.mark.parametrize(
    "changed_row",
    [
        {"reference_capacity_ah": 1.1},
        {"discharge_capacity_ah": -0.1},
        {"discharge_capacity_ah": float("nan")},
        {"soh": 0.8},
    ],
)
def test_supervision_reader_rejects_selected_row_that_disagrees_with_evidence(
    tmp_path: Path,
    changed_row: dict[str, float],
) -> None:
    path = tmp_path / "supervision.parquet"
    rows = [
        {
            "dataset_id": "MATR",
            "cell_id": "MATR_b1c0",
            "cycle_index": cycle,
            "discharge_capacity_ah": 0.9,
            "reference_capacity_ah": 1.0,
            "soh": 0.9,
        }
        for cycle in range(1, 501)
    ]
    rows[0].update(changed_row)
    pq.write_table(pa.Table.from_pylist(rows), path)
    artifact = MatrSupervisionArtifact(
        source_report_sha256="b" * 64,
        raw_sha256="a" * 64,
        horizon_cycle=500,
        cell_count=1,
        row_count=500,
        parquet_relative_path=path.name,
        parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cells=(
            MatrSupervisionCellEvidence(
                cell_id="MATR_b1c0",
                raw_cell_id="b1c0",
                official_life_label=700,
                official_life_right_censored=False,
                reference_capacity_ah=1.0,
                observed_cycle_count=501,
            ),
        ),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    evidence = MatrCellConversionEvidence(
        cell_id="MATR_b1c0",
        raw_cell_id="b1c0",
        official_life_label=700,
        official_life_right_censored=False,
        protocol_id="p1",
        reference_capacity_ah=1.0,
        row_count=1000,
        cycle_count=501,
        quality_issue_counts={},
        manifest_relative_path="MATR_b1c0/manifest.json",
        parquet_sha256="1" * 64,
        metadata_sha256="2" * 64,
    )

    with pytest.raises(ValueError, match="invalid selected row"):
        _load_supervision_rows(
            supervision_path=path,
            supervision=artifact,
            allowed_cell_ids=("MATR_b1c0",),
            allowed_evidence={"MATR_b1c0": evidence},
        )


def _sequence(cell_id: str, value: float) -> EarlyCycleSequence:
    values = torch.full((21, 2, 150, 3), float("nan"), dtype=torch.float32)
    sample_mask = torch.zeros((21, 2, 150), dtype=torch.bool)
    values[1, :, :, :] = value
    sample_mask[1] = True
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        data_version="matr-three-batch-v1",
        feature_version="advanced-v1",
        cycle_indices=tuple(range(21)),
        values=values,
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=("temperature_c",),
        condition_values=torch.tensor([value]),
        condition_mask=torch.tensor([True]),
    )


def test_selection_assembly_fits_each_normalizer_to_its_exact_training_cells() -> None:
    split = SplitManifest(
        dataset_id="MATR",
        train=("train-scalar", "train-censored"),
        validation=("validation",),
        calibration=("calibration",),
        test=("test",),
    )
    sequences = {
        cell_id: _sequence(cell_id, float(index + 1))
        for index, cell_id in enumerate(split.all_cells)
    }
    trajectories = {
        cell_id: {cycle: 1.0 - cycle * 0.0001 for cycle in range(1, 501)}
        for cell_id in ("train-scalar", "train-censored", "validation")
    }

    result = _assemble_advanced_matr_data(
        source_split=split,
        scalar_sequences={
            "train": ((sequences["train-scalar"], 100.0),),
            "validation": ((sequences["validation"], 120.0),),
        },
        hybrid_sequences={
            "train": (
                (sequences["train-scalar"], trajectories["train-scalar"]),
                (sequences["train-censored"], trajectories["train-censored"]),
            ),
            "validation": ((sequences["validation"], trajectories["validation"]),),
        },
        final=False,
    )

    assert isinstance(result, AdvancedSelectionMatrData)
    assert result.scalar_normalizer.training_cell_ids_sha256 == sha256_canonical(["train-scalar"])
    assert result.hybrid_normalizer.training_cell_ids_sha256 == sha256_canonical(
        ["train-censored", "train-scalar"]
    )
    assert result.scalar_train.cell_ids == ("train-scalar",)
    assert result.hybrid_train.cell_ids == ("train-scalar", "train-censored")


def test_selection_does_not_open_heldout_cell_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[str, Path] = {}
    component_splits: list[SplitManifest] = []
    components: list[MatrBatchArtifactReference] = []
    batch_cells = {
        batch_index: {
            "train": (f"MATR_b{batch_index}c0",),
            "validation": (f"MATR_b{batch_index}c1",),
            "calibration": (f"MATR_b{batch_index}c2",),
            "test": (f"MATR_b{batch_index}c3",),
        }
        for batch_index in (1, 2, 3)
    }
    batch_dates = {
        1: date(2017, 5, 12),
        2: date(2017, 6, 30),
        3: date(2018, 4, 12),
    }

    def write_registered(relative: str, model: BaseModel) -> None:
        path = tmp_path / "registered" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(), encoding="utf-8")
        registered[relative] = path

    for batch_index in (1, 2, 3):
        groups = batch_cells[batch_index]
        split = SplitManifest(
            dataset_id="MATR",
            train=groups.get("train", ()),
            validation=groups.get("validation", ()),
            calibration=groups.get("calibration", ()),
            test=groups.get("test", ()),
        )
        component_splits.append(split)
        processed_root = tmp_path / "processed" / f"batch{batch_index}"
        raw_sha256 = str(batch_index) * 64
        conversion_cells: list[MatrCellConversionEvidence] = []
        for raw_index, cell_id in enumerate(split.all_cells):
            metadata = CellMetadata(
                dataset_id="MATR",
                cell_id=cell_id,
                raw_cell_id=f"b{batch_index}c{raw_index}",
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.1,
                reference_capacity_ah=1.0,
                official_life_label=700,
                official_life_label_name="cycle_life",
                source_uri="https://example.invalid/matr",
                source_sha256=raw_sha256,
                schema_version="battery-cell-v1",
            )
            persisted = write_cell_artifacts(
                processed_root, metadata, _records(cell_id)
            )
            conversion_cells.append(
                MatrCellConversionEvidence(
                    cell_id=cell_id,
                    raw_cell_id=f"b{batch_index}c{raw_index}",
                    official_life_label=700,
                    official_life_right_censored=False,
                    protocol_id=f"p{batch_index}",
                    reference_capacity_ah=1.0,
                    row_count=persisted.row_count,
                    cycle_count=501,
                    quality_issue_counts={},
                    manifest_relative_path=persisted.manifest_relative_path,
                    parquet_sha256=persisted.parquet_sha256,
                    metadata_sha256=persisted.metadata_sha256,
                )
            )
        conversion = MatrBatchConversionReport(
            batch_index=batch_index,
            batch_date=batch_dates[batch_index],
            raw_relative_path=f"data/batch{batch_index}.mat",
            raw_size_bytes=100,
            raw_sha256=raw_sha256,
            source_uri="https://example.invalid/matr",
            license_name="research",
            adapter_version="matr-v1",
            time_unit="seconds",
            max_cycle_index=501,
            cell_count=len(conversion_cells),
            total_row_count=sum(cell.row_count for cell in conversion_cells),
            quality_issue_counts={},
            cells=tuple(conversion_cells),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )
        eligibility = MatrTrajectoryEligibilityAudit(
            batch_index=batch_index,
            horizon_cycle=500,
            eligible_cell_ids=split.all_cells,
            excluded=(),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )
        supervision_root = tmp_path / "supervision" / f"batch{batch_index}"
        supervision_root.mkdir(parents=True)
        supervision_path = supervision_root / "supervision.parquet"
        supervision_rows = [
            {
                "dataset_id": "MATR",
                "cell_id": cell.cell_id,
                "cycle_index": cycle,
                "discharge_capacity_ah": 0.9,
                "reference_capacity_ah": 1.0,
                "soh": 0.9,
            }
            for cell in conversion_cells
            for cycle in range(1, 501)
        ]
        pq.write_table(pa.Table.from_pylist(supervision_rows), supervision_path)
        supervision = MatrSupervisionArtifact(
            source_report_sha256=sha256_canonical(
                conversion.model_dump(mode="json")
            ),
            raw_sha256=raw_sha256,
            horizon_cycle=500,
            cell_count=len(conversion_cells),
            row_count=500 * len(conversion_cells),
            parquet_relative_path=supervision_path.name,
            parquet_sha256=hashlib.sha256(
                supervision_path.read_bytes()
            ).hexdigest(),
            cells=tuple(
                MatrSupervisionCellEvidence(
                    cell_id=cell.cell_id,
                    raw_cell_id=cell.raw_cell_id,
                    official_life_label=cell.official_life_label,
                    official_life_right_censored=False,
                    reference_capacity_ah=cell.reference_capacity_ah,
                    observed_cycle_count=cell.cycle_count,
                )
                for cell in conversion_cells
            ),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )
        paths = {
            "conversion": f"batch{batch_index}/conversion.json",
            "split": f"batch{batch_index}/split.json",
            "eligibility": f"batch{batch_index}/eligibility.json",
            "supervision": f"batch{batch_index}/supervision.json",
        }
        write_registered(paths["conversion"], conversion)
        write_registered(paths["split"], split)
        write_registered(paths["eligibility"], eligibility)
        write_registered(paths["supervision"], supervision)
        components.append(
            MatrBatchArtifactReference(
                batch_index=batch_index,
                batch_date=batch_dates[batch_index],
                raw_relative_path=f"data/batch{batch_index}.mat",
                raw_manifest=f"configs/batch{batch_index}.json",
                raw_sha256=raw_sha256,
                processed_root=processed_root.relative_to(tmp_path).as_posix(),
                conversion_report=paths["conversion"],
                conversion_report_sha256="a" * 64,
                split_manifest=paths["split"],
                split_manifest_sha256="b" * 64,
                supervision_root=supervision_root.relative_to(tmp_path).as_posix(),
                supervision_report=paths["supervision"],
                supervision_report_sha256="c" * 64,
                eligibility_report=paths["eligibility"],
                eligibility_report_sha256="d" * 64,
                cell_count=len(conversion_cells),
                scalar_label_count=len(conversion_cells),
                hybrid_eligible_count=len(conversion_cells),
                hybrid_excluded_count=0,
            )
        )

    combined_split = combine_matr_batch_splits(tuple(component_splits))
    combined_relative = "combined-split.json"
    write_registered(combined_relative, combined_split)
    manifest = MatrThreeBatchManifest(
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-cell-split-v1",
        combined_split_manifest=combined_relative,
        combined_split_sha256="e" * 64,
        batches=tuple(components),
        total_cell_count=12,
        scalar_label_count=12,
        hybrid_eligible_count=12,
        hybrid_excluded_count=0,
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )

    def registered_file(_root: Path, relative: str, _sha256: str) -> Path:
        return registered[relative]

    opened: list[Path] = []
    real_read_table = pq.read_table

    def read_table_spy(source: object, *args: object, **kwargs: object) -> pa.Table:
        if isinstance(source, (str, Path)):
            opened.append(Path(source))
        return real_read_table(source, *args, **kwargs)

    monkeypatch.setattr(advanced_data_module, "_verified_registered_file", registered_file)
    monkeypatch.setattr(pq, "read_table", read_table_spy)

    result = load_advanced_matr_selection_data(
        project_root=tmp_path,
        manifest=manifest,
        combined_split=combined_split,
        cutoff_cycle=20,
        feature_version="advanced-v1",
    )

    assert result.scalar_train.cell_ids == (
        "MATR_b1c0",
        "MATR_b2c0",
        "MATR_b3c0",
    )
    assert result.scalar_validation.cell_ids == (
        "MATR_b1c1",
        "MATR_b2c1",
        "MATR_b3c1",
    )
    assert any("MATR_b1c0" in path.as_posix() for path in opened)
    assert any("MATR_b3c1" in path.as_posix() for path in opened)
    assert all(
        f"MATR_b{batch_index}c{heldout_index}" not in path.as_posix()
        for path in opened
        for batch_index in (1, 2, 3)
        for heldout_index in (2, 3)
    )
