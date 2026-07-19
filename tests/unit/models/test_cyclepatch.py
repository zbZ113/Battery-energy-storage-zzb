from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quanxin_life.features.early_cycle_sequence import (
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchLifeRegressor,
    EarlyCycleBatch,
    PhasePatchEncoder,
    stack_early_cycle_sequences,
)


def _raw_sequence(*, cell_id: str, offset: float = 0.0) -> EarlyCycleSequence:
    values = torch.full((21, 2, 150, 3), float("nan"), dtype=torch.float32)
    sample_mask = torch.zeros((21, 2, 150), dtype=torch.bool)
    grid = torch.linspace(0.0, 1.0, 150)
    for cycle in (1, 4):
        sample_mask[cycle] = True
        values[cycle, 0, :, 0] = 3.0 + grid + offset
        values[cycle, 0, :, 1] = 1.0 + offset
        values[cycle, 0, :, 2] = grid
        values[cycle, 1, :, 0] = 4.1 - grid + offset
        values[cycle, 1, :, 1] = -1.0 - offset
        values[cycle, 1, :, 2] = grid
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        data_version="matr-v1",
        feature_version="multichannel-v1",
        cycle_indices=tuple(range(21)),
        values=values,
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=("temperature", "charge_current", "discharge_current"),
        condition_values=torch.tensor([25.0 + offset, 1.0, -1.0]),
        condition_mask=torch.ones(3, dtype=torch.bool),
    )


def _normalised_sequences() -> tuple[EarlyCycleSequence, EarlyCycleSequence]:
    first = _raw_sequence(cell_id="cell-a")
    second = _raw_sequence(cell_id="cell-b", offset=0.2)
    normalizer = EarlyCycleNormalizer.fit(
        (first, second), training_cell_ids=frozenset({"cell-a", "cell-b"})
    )
    return normalizer.transform(first), normalizer.transform(second)


def _batch() -> EarlyCycleBatch:
    return stack_early_cycle_sequences(_normalised_sequences())


def _config() -> CyclePatchConfig:
    return CyclePatchConfig(d_model=24, layers=1, heads=4, dropout=0.0)


def test_stack_builds_mask_preserving_batch_from_normalised_sequences() -> None:
    batch = _batch()

    assert batch.values.shape == (2, 21, 2, 150, 3)
    assert batch.cycle_mask.shape == (2, 21)
    assert batch.sample_mask.shape == (2, 21, 2, 150)
    assert batch.condition_values.shape == (2, 3)
    assert batch.condition_mask.shape == (2, 3)
    assert batch.cell_ids == ("cell-a", "cell-b")
    assert len(batch.normalization_statistics_sha256) == 64


def test_stack_rejects_mixed_normalizers_and_mutated_sequence() -> None:
    first, second = _normalised_sequences()
    mismatched = replace(second, normalization_statistics_sha256="a" * 64)
    with pytest.raises(ValueError, match="normalization_statistics_sha256"):
        stack_early_cycle_sequences((first, mismatched))

    first.values[1, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="input_hash"):
        stack_early_cycle_sequences((first, second))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"d_model": 0}, "d_model"),
        ({"layers": 0}, "layers"),
        ({"heads": 0}, "heads"),
        ({"d_model": 22, "heads": 4}, "divisible"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"max_cycles": 150}, "151"),
    ],
)
def test_config_rejects_invalid_architecture(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **updates)


def test_phase_encoder_masks_samples_and_zeroes_missing_phase() -> None:
    torch.manual_seed(20260712)
    encoder = PhasePatchEncoder(d_model=24).eval()
    values = torch.randn(1, 2, 2, 150, 3)
    mask = torch.ones(1, 2, 2, 150, dtype=torch.bool)
    mask[:, 1, 1] = False
    altered = values.clone()
    altered[:, 1, 1] = 9999.0

    with torch.no_grad():
        first = encoder(values, mask)
        second = encoder(altered, mask)

    assert first.shape == (1, 2, 2, 24)
    assert torch.allclose(first, second)
    assert torch.count_nonzero(first[:, 1, 1]) == 0


def test_regressor_forward_is_finite_scalar_per_cell() -> None:
    torch.manual_seed(20260712)
    model = CyclePatchLifeRegressor(_config(), condition_count=3).eval()

    with torch.no_grad():
        output = model(_batch())

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_masked_batch_values_do_not_change_prediction() -> None:
    torch.manual_seed(20260712)
    model = CyclePatchLifeRegressor(_config(), condition_count=3).eval()
    batch = _batch()
    altered_values = batch.values.clone()
    altered_values[~batch.sample_mask] = 12345.0
    altered = replace(batch, values=altered_values)

    with torch.no_grad():
        expected = model(batch)
        actual = model(altered)

    assert torch.allclose(expected, actual)


def test_legal_cycle_position_and_conditions_both_influence_prediction() -> None:
    torch.manual_seed(20260712)
    model = CyclePatchLifeRegressor(_config(), condition_count=3).eval()
    batch = _batch()
    shifted_values = batch.values.clone()
    shifted_sample_mask = batch.sample_mask.clone()
    shifted_values[:, 2] = shifted_values[:, 1]
    shifted_sample_mask[:, 2] = shifted_sample_mask[:, 1]
    shifted_values[:, 1] = float("nan")
    shifted_sample_mask[:, 1] = False
    shifted = replace(
        batch,
        values=shifted_values,
        sample_mask=shifted_sample_mask,
        cycle_mask=shifted_sample_mask.any(dim=(2, 3)),
    )
    changed_conditions = batch.condition_values.clone()
    changed_conditions[:, 0] += 2.0
    conditioned = replace(batch, condition_values=changed_conditions)

    with torch.no_grad():
        baseline = model(batch)
        position_output = model(shifted)
        condition_output = model(conditioned)

    assert not torch.allclose(baseline, position_output)
    assert not torch.allclose(baseline, condition_output)


@pytest.mark.parametrize("mode", ["duplicate", "reordered"])
def test_batch_rejects_changed_fixed_cycle_axis(mode: str) -> None:
    batch = _batch()
    changed = batch.cycle_indices.clone()
    if mode == "duplicate":
        changed[:, 2] = 1
    else:
        changed[:, [1, 2]] = changed[:, [2, 1]]

    with pytest.raises(ValueError, match=r"cycle_indices|0\.\.cutoff"):
        replace(batch, cycle_indices=changed)


def test_batch_rejects_sample_without_any_valid_cycle() -> None:
    batch = _batch()
    sample_mask = batch.sample_mask.clone()
    sample_mask[0] = False
    cycle_mask = sample_mask.any(dim=(2, 3))

    with pytest.raises(ValueError, match="valid cycle"):
        replace(batch, sample_mask=sample_mask, cycle_mask=cycle_mask)
