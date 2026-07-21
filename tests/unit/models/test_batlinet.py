from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifePairBatch,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
    CyclePatchBatLiNet,
)
from quanxin_life.models.cyclepatch import CyclePatchConfig, EarlyCycleBatch


def _labels(count: int = 12) -> dict[str, float]:
    return {f"cell-{index:02d}": float(100 + index * 50) for index in range(count)}


def _split(training_cell_ids: tuple[str, ...]) -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=training_cell_ids,
        validation=("validation-heldout",),
        calibration=("calibration-heldout",),
        test=("test-heldout",),
    )


def _scaler(labels: dict[str, float] | None = None) -> CycleLifeTargetScaler:
    values = labels or _labels()
    return CycleLifeTargetScaler.fit(
        values,
        training_cell_ids=tuple(values),
        split_manifest=_split(tuple(values)),
        cutoff_cycle=20,
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )


def _encoder_config() -> CyclePatchConfig:
    return CyclePatchConfig(d_model=24, layers=1, heads=4, dropout=0.0)


def _model_config(**updates: object) -> BatLiNetConfig:
    config = BatLiNetConfig(
        encoder=_encoder_config(),
        lambda_pair=1.0,
        lambda_rank=0.1,
        fusion_alpha=0.4,
        reference_count=4,
    )
    return replace(config, **updates)


def _early_batch(batch_size: int = 4) -> EarlyCycleBatch:
    torch.manual_seed(20260712)
    values = torch.full(
        (batch_size, 21, 2, 150, 3), float("nan"), dtype=torch.float32
    )
    sample_mask = torch.zeros((batch_size, 21, 2, 150), dtype=torch.bool)
    for cycle in (1, 4):
        sample_mask[:, cycle] = True
        values[:, cycle] = torch.randn(batch_size, 2, 150, 3)
    return EarlyCycleBatch(
        dataset_id="MATR",
        data_version="matr-v1",
        feature_version="multichannel-v1",
        normalization_statistics_sha256="a" * 64,
        cell_ids=tuple(f"cell-{index:02d}" for index in range(batch_size)),
        condition_names=("temperature", "charge", "discharge"),
        values=values,
        cycle_indices=torch.arange(21).expand(batch_size, -1),
        cycle_mask=sample_mask.any(dim=(2, 3)),
        sample_mask=sample_mask,
        condition_values=torch.randn(batch_size, 3),
        condition_mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )


def test_target_scaler_uses_only_declared_matr_training_cells_and_round_trips() -> None:
    labels = _labels(4)
    scaler = _scaler(labels)
    raw = torch.tensor(tuple(labels.values()), dtype=torch.float32)

    standardized = scaler.transform(raw)

    assert torch.allclose(scaler.inverse_transform(standardized), raw)
    assert len(scaler.context_sha256) == 64
    with pytest.raises(ValueError, match="training"):
        CycleLifeTargetScaler.fit(
            {**labels, "heldout": 900.0},
            training_cell_ids=tuple(labels),
            split_manifest=_split(tuple(labels)),
            cutoff_cycle=20,
            dataset_id="MATR",
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )
    with pytest.raises(ValueError, match="MATR official"):
        CycleLifeTargetScaler.fit(
            labels,
            training_cell_ids=tuple(labels),
            split_manifest=_split(tuple(labels)),
            cutoff_cycle=20,
            dataset_id="MATR",
            target=PredictionTarget.UNIFIED_EOL80_CYCLE,
        )


def test_target_scaler_constant_labels_use_unit_scale_and_validate_context() -> None:
    labels = {"cell-a": 100.0, "cell-b": 100.0}
    scaler = _scaler(labels)

    assert scaler.scale == 1.0
    assert scaler.transform(torch.tensor([100.0])).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="context_sha256"):
        replace(scaler, context_sha256="0" * 64)
    with pytest.raises(ValueError, match="unique"):
        CycleLifeTargetScaler.fit(
            labels,
            training_cell_ids=("cell-a", "cell-a"),
            split_manifest=_split(tuple(labels)),
            cutoff_cycle=20,
            dataset_id="MATR",
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )


def test_target_scaler_hash_binds_labels_to_cell_identity() -> None:
    first = _scaler({"cell-a": 100.0, "cell-b": 200.0})
    permuted = _scaler({"cell-a": 200.0, "cell-b": 100.0})

    assert first.mean == permuted.mean
    assert first.scale == permuted.scale
    assert first.training_labels_sha256 != permuted.training_labels_sha256
    assert first.context_sha256 != permuted.context_sha256


def test_manifest_prevents_heldout_from_being_declared_as_training() -> None:
    labels = {"cell-a": 100.0, "cell-b": 200.0}
    split = _split(tuple(labels))
    claimed_training = (*tuple(labels), "validation-heldout")
    contaminated = {**labels, "validation-heldout": 300.0}

    with pytest.raises(ValueError, match=r"SplitManifest|split.*train|training"):
        CycleLifeTargetScaler.fit(
            contaminated,
            training_cell_ids=claimed_training,
            split_manifest=split,
            cutoff_cycle=20,
            dataset_id="MATR",
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )

    scaler = _scaler(labels)
    with pytest.raises(ValueError, match=r"SplitManifest|split.*train|training"):
        CycleLifeReferenceLibrary.build(
            contaminated,
            training_cell_ids=claimed_training,
            split_manifest=split,
            scaler=scaler,
            reference_count=2,
            seed=1,
        )


@pytest.mark.parametrize("bad_label", [float("nan"), float("inf"), 20.0])
def test_target_scaler_rejects_invalid_official_labels(bad_label: float) -> None:
    with pytest.raises(ValueError, match=r"finite|cutoff"):
        _scaler({"cell-a": 100.0, "cell-b": bad_label})


def test_reference_library_is_stratified_seeded_and_reproducible() -> None:
    labels = _labels()
    scaler = _scaler(labels)
    first = CycleLifeReferenceLibrary.build(
        labels,
        training_cell_ids=tuple(labels),
        split_manifest=_split(tuple(labels)),
        scaler=scaler,
        reference_count=4,
        seed=20260712,
    )
    repeated = CycleLifeReferenceLibrary.build(
        labels,
        training_cell_ids=tuple(labels),
        split_manifest=_split(tuple(labels)),
        scaler=scaler,
        reference_count=4,
        seed=20260712,
    )
    changed = CycleLifeReferenceLibrary.build(
        labels,
        training_cell_ids=tuple(labels),
        split_manifest=_split(tuple(labels)),
        scaler=scaler,
        reference_count=4,
        seed=20260713,
    )

    assert first == repeated
    assert first.cell_ids != changed.cell_ids
    assert first.quantile_bins == (0, 1, 2, 3)
    assert len(first.library_sha256) == 64
    assert first.training_labels_sha256 == scaler.training_labels_sha256


def test_reference_library_rejects_heldout_or_excess_reference_count() -> None:
    labels = _labels(6)
    scaler = _scaler(labels)
    with pytest.raises(ValueError, match=r"heldout|training"):
        CycleLifeReferenceLibrary.build(
            {**labels, "heldout": 900.0},
            training_cell_ids=tuple(labels),
            split_manifest=_split(tuple(labels)),
            scaler=scaler,
            reference_count=3,
            seed=1,
        )
    with pytest.raises(ValueError, match="reference_count"):
        CycleLifeReferenceLibrary.build(
            labels,
            training_cell_ids=tuple(labels),
            split_manifest=_split(tuple(labels)),
            scaler=scaler,
            reference_count=7,
            seed=1,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"lambda_pair": -1.0}, "lambda_pair"),
        ({"lambda_pair": 0.0}, "lambda_pair"),
        ({"lambda_rank": -1.0}, "lambda_rank"),
        ({"fusion_alpha": -0.1}, "fusion_alpha"),
        ({"fusion_alpha": 1.1}, "fusion_alpha"),
        ({"reference_count": 0}, "reference_count"),
    ],
)
def test_batlinet_config_rejects_invalid_values(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _model_config(**updates)


def test_pair_batch_validates_alignment_and_model_loss_has_gradients() -> None:
    torch.manual_seed(20260712)
    model = CyclePatchBatLiNet(_model_config(), condition_count=3)
    embeddings = model.encode(_early_batch(batch_size=6))
    pair_batch = CycleLifePairBatch(
        target_embeddings=embeddings[4:],
        target_labels=torch.tensor([0.5, 1.0]),
        reference_embeddings=embeddings[:4],
        reference_labels=torch.tensor([-1.0, -0.5, 0.0, 0.25]),
    )

    loss = model.loss(pair_batch)
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(ValueError, match="target_labels"):
        replace(pair_batch, target_labels=torch.tensor([0.5]))


def test_reference_median_is_order_invariant_and_alpha_one_is_direct_only() -> None:
    torch.manual_seed(20260712)
    target = torch.randn(3, 24)
    references = torch.randn(4, 24)
    labels = torch.tensor([-1.0, -0.2, 0.4, 1.3])
    model = CyclePatchBatLiNet(_model_config(), condition_count=3).eval()
    direct_model = CyclePatchBatLiNet(
        _model_config(fusion_alpha=1.0), condition_count=3
    ).eval()

    with torch.no_grad():
        expected = model.fuse_standardized(target, references, labels)
        order = torch.tensor([2, 0, 3, 1])
        reordered = model.fuse_standardized(target, references[order], labels[order])
        direct = direct_model.direct_head(target).squeeze(1)
        direct_fused = direct_model.fuse_standardized(target, references, labels)

    assert expected.shape == (3,)
    assert torch.allclose(expected, reordered)
    assert torch.allclose(direct, direct_fused)


def test_fusion_does_not_require_nondeterministic_median_dim_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_median = torch.median

    def reject_dimension_median(
        values: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if args or "dim" in kwargs:
            raise RuntimeError("median CUDA with indices output is nondeterministic")
        return original_median(values)

    monkeypatch.setattr(torch, "median", reject_dimension_median)
    model = CyclePatchBatLiNet(_model_config(), condition_count=3).eval()

    with torch.no_grad():
        fused = model.fuse_standardized(
            torch.randn(3, 24),
            torch.randn(4, 24),
            torch.tensor([-1.0, -0.2, 0.4, 1.3]),
        )

    assert fused.shape == (3,)
    assert torch.isfinite(fused).all()


def test_pair_batch_requires_one_floating_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        CycleLifePairBatch(
            target_embeddings=torch.randn(2, 24, dtype=torch.float32),
            target_labels=torch.randn(2, dtype=torch.float64),
            reference_embeddings=torch.randn(4, 24, dtype=torch.float32),
            reference_labels=torch.randn(4, dtype=torch.float32),
        )


def test_model_rejects_pair_batch_dtype_incompatible_with_parameters() -> None:
    model = CyclePatchBatLiNet(_model_config(), condition_count=3)
    pair_batch = CycleLifePairBatch(
        target_embeddings=torch.randn(2, 24, dtype=torch.float64),
        target_labels=torch.randn(2, dtype=torch.float64),
        reference_embeddings=torch.randn(4, 24, dtype=torch.float64),
        reference_labels=torch.randn(4, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match=r"model parameters|dtype"):
        model.loss(pair_batch)


def test_fusion_rejects_reference_label_dtype_mismatch() -> None:
    model = CyclePatchBatLiNet(_model_config(), condition_count=3)
    targets = torch.randn(2, 24)
    references = torch.randn(4, 24)

    with pytest.raises(ValueError, match=r"dtype|floating"):
        model.fuse_standardized(
            targets,
            references,
            torch.randn(4, dtype=torch.float64),
        )



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fusion_rejects_reference_label_device_mismatch() -> None:
    model = CyclePatchBatLiNet(_model_config(), condition_count=3)
    targets = torch.randn(2, 24)
    references = torch.randn(4, 24)
    with pytest.raises(ValueError, match="device"):
        model.fuse_standardized(targets, references, torch.randn(4, device="cuda"))


def test_all_pair_entrypoints_require_configured_reference_count() -> None:
    model = CyclePatchBatLiNet(_model_config(), condition_count=3)
    targets = torch.randn(2, 24)
    references = torch.randn(3, 24)
    labels = torch.randn(3)

    with pytest.raises(ValueError, match="reference_count"):
        model.pair_delta(targets, references)
    with pytest.raises(ValueError, match="reference_count"):
        model.fuse_standardized(targets, references, labels)
    pair_batch = CycleLifePairBatch(
        target_embeddings=targets,
        target_labels=torch.randn(2),
        reference_embeddings=references,
        reference_labels=labels,
    )
    with pytest.raises(ValueError, match="reference_count"):
        model.loss(pair_batch)
