from __future__ import annotations

from pathlib import Path

import pytest
import torch

import quanxin_life.features.verified_matr_sequence as verified_module
from quanxin_life.core import CellMetadata
from quanxin_life.data.matr_pipeline import MatrCellConversionEvidence
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.data.storage import write_cell_artifacts
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig
from quanxin_life.features.verified_matr_sequence import (
    load_verified_matr_early_sequence,
)
from quanxin_life.training.advanced_data import _load_early_sequence


def test_verified_loader_is_the_training_loader_numerical_source(
    tmp_path: Path,
) -> None:
    raw_sha256 = "a" * 64
    metadata = CellMetadata(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        raw_cell_id="b1c0",
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
        tmp_path,
        metadata,
        _records(metadata.cell_id),
    )
    evidence = MatrCellConversionEvidence(
        cell_id=metadata.cell_id,
        raw_cell_id=metadata.raw_cell_id,
        official_life_label=metadata.official_life_label,
        official_life_right_censored=False,
        protocol_id="p1",
        reference_capacity_ah=metadata.reference_capacity_ah,
        row_count=persisted.row_count,
        cycle_count=22,
        quality_issue_counts={},
        manifest_relative_path=persisted.manifest_relative_path,
        parquet_sha256=persisted.parquet_sha256,
        metadata_sha256=persisted.metadata_sha256,
    )
    config = MultichannelCycleConfig(
        cutoff_cycle=20,
        feature_version="cyclepatch-multichannel-v1",
    )

    shared = load_verified_matr_early_sequence(
        processed_root=tmp_path,
        evidence=evidence,
        reference_capacity_ah=evidence.reference_capacity_ah,
        raw_sha256=raw_sha256,
        config=config,
        data_version="matr-three-batch-v1",
    )
    training = _load_early_sequence(
        processed_root=tmp_path,
        evidence=evidence,
        reference_capacity_ah=evidence.reference_capacity_ah,
        raw_sha256=raw_sha256,
        config=config,
        data_version="matr-three-batch-v1",
    )

    assert shared.sequence.input_hash == training.input_hash
    torch.testing.assert_close(
        shared.sequence.values,
        training.values,
        equal_nan=True,
    )
    assert torch.equal(shared.sequence.cycle_mask, training.cycle_mask)
    assert torch.equal(shared.sequence.sample_mask, training.sample_mask)
    assert torch.equal(
        shared.sequence.condition_values,
        training.condition_values,
    )
    assert torch.equal(
        shared.sequence.condition_mask,
        training.condition_mask,
    )
    assert shared.masked_cycle_indices == ()
    assert shared.sequence.cutoff_cycle == 20


def test_loader_hashes_the_exact_parquet_bytes_it_consumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_sha256 = "a" * 64
    metadata = CellMetadata(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        raw_cell_id="b1c0",
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
        tmp_path,
        metadata,
        _records(metadata.cell_id),
    )
    evidence = MatrCellConversionEvidence(
        cell_id=metadata.cell_id,
        raw_cell_id=metadata.raw_cell_id,
        official_life_label=metadata.official_life_label,
        official_life_right_censored=False,
        protocol_id="p1",
        reference_capacity_ah=metadata.reference_capacity_ah,
        row_count=persisted.row_count,
        cycle_count=22,
        quality_issue_counts={},
        manifest_relative_path=persisted.manifest_relative_path,
        parquet_sha256=persisted.parquet_sha256,
        metadata_sha256=persisted.metadata_sha256,
    )
    real_verify = verified_module.verify_cell_artifacts

    def verify_then_replace(*args, **kwargs):
        verified = real_verify(*args, **kwargs)
        verified.parquet_path.write_bytes(b"tampered-after-verify")
        return verified

    monkeypatch.setattr(
        verified_module,
        "verify_cell_artifacts",
        verify_then_replace,
    )

    with pytest.raises(ValueError, match="Parquet SHA-256"):
        load_verified_matr_early_sequence(
            processed_root=tmp_path,
            evidence=evidence,
            reference_capacity_ah=evidence.reference_capacity_ah,
            raw_sha256=raw_sha256,
            config=MultichannelCycleConfig(
                cutoff_cycle=20,
                feature_version="cyclepatch-multichannel-v1",
            ),
            data_version="matr-three-batch-v1",
        )


def _records(cell_id: str) -> tuple[CycleRecord, ...]:
    rows: list[CycleRecord] = []
    for cycle in range(1, 23):
        for sample in range(3):
            rows.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id=cell_id,
                    cycle_index=cycle,
                    sample_index=sample,
                    time_s=float(cycle * 100 + sample),
                    voltage_v=3.0 + 0.1 * sample,
                    current_a=1.0,
                    temperature_c=25.0,
                    charge_capacity_ah=0.2 + 0.1 * sample,
                )
            )
        for sample in range(3, 6):
            rows.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id=cell_id,
                    cycle_index=cycle,
                    sample_index=sample,
                    time_s=float(cycle * 100 + sample),
                    voltage_v=3.5 - 0.1 * (sample - 3),
                    current_a=-1.0,
                    temperature_c=25.0,
                    discharge_capacity_ah=0.3 - 0.05 * (sample - 3),
                )
            )
    return tuple(rows)
