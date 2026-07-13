from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import pytest
import torch

from quanxin_life.data.schemas import SplitManifest


def _dann_symbols() -> tuple[object, object, object, object, object, object]:
    module = import_module("quanxin_life.adaptation.dann")
    return (
        module.DANNConfig,
        module.CPMLPDANNAdapter,
        module.SourceDomainBatch,
        module.TargetDomainBatch,
        module.TargetDomainCohort,
        module.gradient_reverse,
    )


def _feature_contract(
    *,
    feature_version: str = "curve-tensor-v1",
    cutoff_cycle: int = 20,
    cycle_indices: tuple[int, ...] = (0, 10, 20),
    voltage_grid_v: tuple[float, ...] = (3.0, 3.2),
) -> object:
    module = import_module("quanxin_life.adaptation.dann")
    return module.CurveFeatureContract(
        feature_version=feature_version,
        cutoff_cycle=cutoff_cycle,
        cycle_indices=cycle_indices,
        voltage_grid_v=voltage_grid_v,
    )


def _source_manifest() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("source-a", "source-b", "source-c", "source-d"),
        validation=("source-validation",),
        calibration=("source-calibration",),
        test=("source-test",),
    )


def _target_manifest() -> SplitManifest:
    return SplitManifest(
        dataset_id="HUST",
        train=("target-adapt-a", "target-adapt-b"),
        validation=("target-validation",),
        calibration=("target-calibration-a", "target-calibration-b"),
        test=("target-test",),
    )


def _curve_values(row_count: int, *, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty((row_count, 3, 2), dtype=torch.float32)
    for row_index in range(row_count):
        values[row_index, 0] = torch.tensor((offset + row_index, offset + row_index + 0.2))
        values[row_index, 1] = torch.tensor(
            (offset + row_index + 0.1, offset + row_index + 0.3)
        )
        values[row_index, 2] = torch.tensor((float("nan"), float("nan")))
    mask = torch.tensor([[True, True, False]] * row_count, dtype=torch.bool)
    return values, mask


def _source_batch() -> object:
    _, _, SourceDomainBatch, _, _, _ = _dann_symbols()
    values, mask = _curve_values(4, offset=0.1)
    return SourceDomainBatch(
        dataset_id="MATR",
        cell_ids=("source-a", "source-b", "source-c", "source-d"),
        curve_values=values,
        observed_mask=mask,
        eol80_labels=torch.tensor((100.0, 130.0, 170.0, 210.0)),
        feature_contract=_feature_contract(),
    )


def _target_adaptation_batch() -> object:
    _, _, _, TargetDomainBatch, _, _ = _dann_symbols()
    values, mask = _curve_values(2, offset=3.0)
    return TargetDomainBatch(
        dataset_id="HUST",
        cell_ids=("target-adapt-a", "target-adapt-b"),
        curve_values=values,
        observed_mask=mask,
        feature_contract=_feature_contract(),
    )


def _target_test_batch() -> object:
    _, _, _, TargetDomainBatch, _, _ = _dann_symbols()
    values, mask = _curve_values(1, offset=4.0)
    return TargetDomainBatch(
        dataset_id="HUST",
        cell_ids=("target-test",),
        curve_values=values,
        observed_mask=mask,
        feature_contract=_feature_contract(),
    )


def _adapter() -> object:
    DANNConfig, CPMLPDANNAdapter, _, _, _, _ = _dann_symbols()
    return CPMLPDANNAdapter(
        DANNConfig(
            model_version="cpmlp-dann-v1",
            feature_version="curve-tensor-v1",
            source_split_version="matr-split-v1",
            target_split_version="hust-split-v1",
            epochs=10,
            batch_size=2,
            learning_rate=0.02,
            curve_hidden_dim=6,
            representation_dim=5,
            domain_hidden_dim=4,
            domain_loss_weight=0.3,
        )
    )


def test_gradient_reversal_negates_the_encoder_gradient() -> None:
    *_, gradient_reverse = _dann_symbols()
    value = torch.tensor((2.0,), requires_grad=True)

    gradient_reverse(value, coefficient=0.75).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([-0.75])


def test_fit_rejects_target_test_cells_and_target_labels() -> None:
    _, _, _, TargetDomainBatch, TargetDomainCohort, _ = _dann_symbols()
    with pytest.raises(TypeError, match="eol80_labels"):
        TargetDomainBatch(  # type: ignore[call-arg]
            dataset_id="HUST",
            cell_ids=("target-test",),
            curve_values=torch.ones((1, 3, 2)),
            observed_mask=torch.ones((1, 3), dtype=torch.bool),
            eol80_labels=torch.tensor((123.0,)),
            feature_contract=_feature_contract(),
        )

    with pytest.raises(ValueError, match="target adaptation cell_ids"):
        _adapter().fit(
            source_batch=_source_batch(),
            target_batch=_target_test_batch(),
            source_split_manifest=_source_manifest(),
            target_split_manifest=_target_manifest(),
            target_cohort=TargetDomainCohort.ADAPTATION,
        )


def test_batches_reject_duplicate_cell_ids() -> None:
    _, _, _, TargetDomainBatch, _, _ = _dann_symbols()
    values, mask = _curve_values(2, offset=2.0)

    with pytest.raises(ValueError, match="cell_ids must be non-empty and unique"):
        TargetDomainBatch(
            dataset_id="HUST",
            cell_ids=("target-adapt-a", "target-adapt-a"),
            curve_values=values,
            observed_mask=mask,
            feature_contract=_feature_contract(),
        )


def test_fit_rejects_cross_domain_cell_id_overlap() -> None:
    _, _, _, TargetDomainBatch, TargetDomainCohort, _ = _dann_symbols()
    target_values, target_mask = _curve_values(2, offset=3.0)
    overlapping_target = TargetDomainBatch(
        dataset_id="HUST",
        cell_ids=("source-a", "target-adapt-b"),
        curve_values=target_values,
        observed_mask=target_mask,
        feature_contract=_feature_contract(),
    )
    overlapping_target_manifest = SplitManifest(
        dataset_id="HUST",
        train=("source-a", "target-adapt-b"),
        validation=("target-validation",),
        calibration=("target-calibration-a", "target-calibration-b"),
        test=("target-test",),
    )
    with pytest.raises(ValueError, match="global bare cell_id overlap"):
        _adapter().fit(
            source_batch=_source_batch(),
            target_batch=overlapping_target,
            source_split_manifest=_source_manifest(),
            target_split_manifest=overlapping_target_manifest,
            target_cohort=TargetDomainCohort.ADAPTATION,
        )


def test_fit_rejects_incompatible_curve_axes() -> None:
    _, _, _, TargetDomainBatch, _, _ = _dann_symbols()
    incompatible_values = torch.ones((2, 4, 2), dtype=torch.float32)
    incompatible_values[:, 3, :] = float("nan")
    with pytest.raises(ValueError, match="cycle dimension"):
        TargetDomainBatch(
            dataset_id="HUST",
            cell_ids=("target-adapt-a", "target-adapt-b"),
            curve_values=incompatible_values,
            observed_mask=torch.tensor([[True, True, True, False]] * 2, dtype=torch.bool),
            feature_contract=_feature_contract(),
        )


def test_fit_predicts_finite_target_test_values_deterministically() -> None:
    _, _, _, _, TargetDomainCohort, _ = _dann_symbols()
    first = _adapter().fit(
        source_batch=_source_batch(),
        target_batch=_target_adaptation_batch(),
        source_split_manifest=_source_manifest(),
        target_split_manifest=_target_manifest(),
        target_cohort=TargetDomainCohort.ADAPTATION,
    )
    second = _adapter().fit(
        source_batch=_source_batch(),
        target_batch=_target_adaptation_batch(),
        source_split_manifest=_source_manifest(),
        target_split_manifest=_target_manifest(),
        target_cohort=TargetDomainCohort.ADAPTATION,
    )

    first_prediction = first.predict(_target_test_batch())
    second_prediction = second.predict(_target_test_batch())

    assert first_prediction.dataset_id == "HUST"
    assert first_prediction.cell_ids == ("target-test",)
    assert first_prediction.target_domain_cohort == TargetDomainCohort.ADAPTATION
    assert torch.isfinite(torch.tensor(first_prediction.predicted_eol_cycles)).all()
    assert first_prediction.predicted_eol_cycles == pytest.approx(
        second_prediction.predicted_eol_cycles, abs=1e-6
    )


def test_curve_feature_contract_rejects_invalid_axes_and_dann_requires_exact_match() -> None:
    with pytest.raises(ValueError, match="cycle_indices"):
        _feature_contract(cycle_indices=(0, 20, 10))

    _, _, _, TargetDomainBatch, TargetDomainCohort, _ = _dann_symbols()
    target_values, target_mask = _curve_values(2, offset=3.0)
    mismatched_target = TargetDomainBatch(
        dataset_id="HUST",
        cell_ids=("target-adapt-a", "target-adapt-b"),
        curve_values=target_values,
        observed_mask=target_mask,
        feature_contract=_feature_contract(voltage_grid_v=(3.0, 3.3)),
    )
    with pytest.raises(ValueError, match="feature contracts"):
        _adapter().fit(
            source_batch=_source_batch(),
            target_batch=mismatched_target,
            source_split_manifest=_source_manifest(),
            target_split_manifest=_target_manifest(),
            target_cohort=TargetDomainCohort.ADAPTATION,
        )

    mismatched_config = replace(_adapter().config, feature_version="curve-tensor-v2")
    with pytest.raises(ValueError, match="feature_version"):
        _dann_symbols()[1](mismatched_config).fit(
            source_batch=_source_batch(),
            target_batch=_target_adaptation_batch(),
            source_split_manifest=_source_manifest(),
            target_split_manifest=_target_manifest(),
            target_cohort=TargetDomainCohort.ADAPTATION,
        )


def test_fit_rejects_global_bare_cell_id_overlap_across_domain_manifests() -> None:
    _, _, _, _, TargetDomainCohort, _ = _dann_symbols()
    unsafe_target_manifest = SplitManifest(
        dataset_id="HUST",
        train=("target-adapt-a", "target-adapt-b"),
        validation=("target-validation",),
        calibration=("target-calibration-a", "target-calibration-b"),
        test=("source-a",),
    )

    with pytest.raises(ValueError, match="global bare cell_id overlap"):
        _adapter().fit(
            source_batch=_source_batch(),
            target_batch=_target_adaptation_batch(),
            source_split_manifest=_source_manifest(),
            target_split_manifest=unsafe_target_manifest,
            target_cohort=TargetDomainCohort.ADAPTATION,
        )


def test_prediction_carries_versions_seed_and_normalization_provenance() -> None:
    _, _, _, _, TargetDomainCohort, _ = _dann_symbols()
    adapter = _adapter().fit(
        source_batch=_source_batch(),
        target_batch=_target_adaptation_batch(),
        source_split_manifest=_source_manifest(),
        target_split_manifest=_target_manifest(),
        target_cohort=TargetDomainCohort.ADAPTATION,
    )

    prediction = adapter.predict(_target_test_batch())

    assert prediction.feature_version == "curve-tensor-v1"
    assert prediction.source_split_version == "matr-split-v1"
    assert prediction.target_split_version == "hust-split-v1"
    assert prediction.random_seed == 20260712
    assert prediction.normalization_version == "source-train-zscore-v1"
    assert len(prediction.normalization_input_hash) == 64
