from __future__ import annotations

from pathlib import Path

import pytest
import torch

import quanxin_life.application.advanced_calibration_inputs as inputs_module
from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationCellSource,
    AdvancedCalibrationSourceIdentity,
)
from quanxin_life.application.advanced_calibration_inputs import (
    RegisteredMatrAdvancedCalibrationCellInputResolver,
)
from quanxin_life.core import AdvancedModelTask
from quanxin_life.data.matr_pipeline import MatrCellConversionEvidence
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig
from quanxin_life.features.verified_matr_sequence import (
    VerifiedMatrEarlySequenceLoad,
)


@pytest.mark.parametrize(
    ("task", "initial_soh"),
    (
        (AdvancedModelTask.RUL, None),
        (AdvancedModelTask.SOH, 0.91),
    ),
)
def test_resolves_server_owned_cell_source_through_shared_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: AdvancedModelTask,
    initial_soh: float | None,
) -> None:
    identity = _source_identity()
    source = AdvancedCalibrationCellSource(
        source_identity=identity,
        processed_root=tmp_path,
        raw_sha256="a" * 64,
        cell_evidence=_cell_evidence(),
        initial_soh=initial_soh,
    )
    source_resolver = _SourceResolver(source)
    sequence = _sequence()
    loader_calls: list[dict[str, object]] = []

    def load(**kwargs: object) -> VerifiedMatrEarlySequenceLoad:
        loader_calls.append(kwargs)
        return VerifiedMatrEarlySequenceLoad(
            sequence=sequence,
            masked_cycle_indices=(),
        )

    monkeypatch.setattr(
        inputs_module,
        "load_verified_matr_early_sequence",
        load,
    )
    resolver = RegisteredMatrAdvancedCalibrationCellInputResolver(
        source_resolver=source_resolver
    )

    resolved = resolver.resolve(
        source_registration_id=identity.registration_id,
        source_identity=identity,
        task=task,
        cell_id="cell-a",
        cutoff_cycle=100,
        feature_version="advanced-feature-v1",
    )

    assert resolved.raw_sequence.input_hash == sequence.input_hash
    assert resolved.initial_soh == initial_soh
    assert resolved.source_identity_sha256 == identity.source_identity_sha256
    assert source_resolver.calls == [
        (
            identity.registration_id,
            identity.source_identity_sha256,
            task,
            100,
            "cell-a",
        )
    ]
    assert loader_calls[0]["processed_root"] == tmp_path
    assert loader_calls[0]["evidence"] == source.cell_evidence
    config = loader_calls[0]["config"]
    assert isinstance(config, MultichannelCycleConfig)
    assert config.cutoff_cycle == 100
    assert config.feature_version == "advanced-feature-v1"


class _SourceResolver:
    def __init__(self, source: AdvancedCalibrationCellSource) -> None:
        self.source = source
        self.calls: list[
            tuple[str, str, AdvancedModelTask, int, str]
        ] = []

    def resolve_cell_source(
        self,
        registration_id: str,
        *,
        source_identity: AdvancedCalibrationSourceIdentity,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        cell_id: str,
    ) -> AdvancedCalibrationCellSource:
        self.calls.append(
            (
                registration_id,
                source_identity.source_identity_sha256,
                task,
                cutoff_cycle,
                cell_id,
            )
        )
        return self.source


def _source_identity() -> AdvancedCalibrationSourceIdentity:
    return AdvancedCalibrationSourceIdentity(
        registration_id="matr-three-batch-final-v1",
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-split-v1",
        three_batch_manifest_sha256="1" * 64,
        combined_split_sha256="2" * 64,
        conversion_report_sha256s=("3" * 64, "4" * 64, "5" * 64),
        component_split_sha256s=("6" * 64, "7" * 64, "8" * 64),
        eligibility_report_sha256s=("9" * 64, "a" * 64, "b" * 64),
        supervision_report_sha256s=("c" * 64, "d" * 64, "e" * 64),
        supervision_parquet_sha256s=("f" * 64, "0" * 64, "1" * 64),
        source_identity_sha256="2" * 64,
    )


def _cell_evidence() -> MatrCellConversionEvidence:
    return MatrCellConversionEvidence(
        cell_id="cell-a",
        raw_cell_id="b1c0",
        official_life_label=700,
        official_life_right_censored=False,
        protocol_id="p1",
        reference_capacity_ah=1.0,
        row_count=100,
        cycle_count=500,
        quality_issue_counts={},
        manifest_relative_path="cells/cell-a/manifest.json",
        parquet_sha256="3" * 64,
        metadata_sha256="4" * 64,
    )


def _sequence() -> EarlyCycleSequence:
    sample_mask = torch.ones((101, 2, 150), dtype=torch.bool)
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id="cell-a",
        cutoff_cycle=100,
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        cycle_indices=tuple(range(101)),
        values=torch.ones((101, 2, 150, 3), dtype=torch.float32),
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=("temperature_c", "charge_rate_c"),
        condition_values=torch.tensor([25.0, 1.0]),
        condition_mask=torch.tensor([True, True]),
    )
