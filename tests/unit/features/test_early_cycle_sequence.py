from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quanxin_life.features.early_cycle_sequence import (
    PHASE_NAMES,
    VARIABLE_NAMES,
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)


def _sequence(
    *,
    cell_id: str = "b1c0",
    values: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    condition_values: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
) -> EarlyCycleSequence:
    point_mask = sample_mask
    if point_mask is None:
        point_mask = torch.ones((21, 2, 150), dtype=torch.bool)
        point_mask[2, 0, 149] = False
    tensor = values
    if tensor is None:
        tensor = torch.arange(21 * 2 * 150 * 3, dtype=torch.float32).reshape(
            21, 2, 150, 3
        )
        tensor[..., 0] = 3.0 + tensor[..., 0] / 100_000.0
        tensor[..., 1] = tensor[..., 1] / 10_000.0
        tensor[..., 2] = tensor[..., 2] / 20_000.0
        tensor[~point_mask] = float("nan")
    conditions = condition_values
    if conditions is None:
        conditions = torch.tensor([25.0, 1.0], dtype=torch.float32)
    conditions_mask = condition_mask
    if conditions_mask is None:
        conditions_mask = torch.tensor([True, True], dtype=torch.bool)
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        data_version="matr-three-batch-v1",
        feature_version="early-cycle-sequence-v1",
        cycle_indices=tuple(range(21)),
        values=tensor,
        cycle_mask=point_mask.any(dim=(1, 2)),
        sample_mask=point_mask,
        condition_names=("temperature_c", "charge_rate_c"),
        condition_values=conditions,
        condition_mask=conditions_mask,
    )


def test_sequence_accepts_fixed_axes_and_one_missing_phase() -> None:
    sample_mask = torch.ones((21, 2, 150), dtype=torch.bool)
    sample_mask[3, 0] = False
    values = torch.ones((21, 2, 150, 3), dtype=torch.float32)
    values[~sample_mask] = float("nan")

    sequence = _sequence(values=values, sample_mask=sample_mask)

    assert sequence.values.shape == (21, 2, 150, 3)
    assert sequence.phase_names == PHASE_NAMES == ("charge", "discharge")
    assert sequence.variable_names == VARIABLE_NAMES == (
        "voltage_v",
        "current_a",
        "capacity_ah",
    )
    assert sequence.cycle_mask[3]
    assert sequence.normalization_version == "none"
    assert sequence.normalization_statistics_sha256 is None
    assert len(sequence.input_hash) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "", "dataset_id"),
        ("cell_id", "", "cell_id"),
        ("data_version", "", "data_version"),
        ("feature_version", "", "feature_version"),
        (
            "cycle_indices",
            (*range(20), 19),
            "strictly increasing",
        ),
        ("cycle_indices", tuple(range(1, 22)), "0..cutoff_cycle"),
        ("condition_names", ("temperature_c", "temperature_c"), "unique"),
    ],
)
def test_sequence_rejects_invalid_identity_axes_and_conditions(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_sequence(), **{field: value})


def test_sequence_rejects_wrong_tensor_dtype_shape_or_sample_count() -> None:
    with pytest.raises(ValueError, match=r"torch\.float32"):
        replace(_sequence(), values=_sequence().values.to(torch.float64))
    with pytest.raises(ValueError, match="variable=3"):
        replace(_sequence(), values=_sequence().values[..., :2])
    with pytest.raises(ValueError, match="150"):
        replace(
            _sequence(),
            values=_sequence().values[:, :, :149],
            sample_mask=_sequence().sample_mask[:, :, :149],
        )


def test_sequence_enforces_nan_and_exact_cycle_mask_semantics() -> None:
    values = _sequence().values.clone()
    values[2, 0, 149] = torch.tensor([3.0, 1.0, 0.1])
    with pytest.raises(ValueError, match="unobserved points"):
        replace(_sequence(), values=values)

    values = _sequence().values.clone()
    values[0, 0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="observed points"):
        replace(_sequence(), values=values)

    with pytest.raises(ValueError, match="exactly equal"):
        replace(
            _sequence(),
            cycle_mask=torch.tensor([False, *([True] * 20)], dtype=torch.bool),
        )

    sample_mask = _sequence().sample_mask.clone()
    sample_mask[0] = False
    values = _sequence().values.clone()
    values[0] = float("nan")
    with pytest.raises(ValueError, match="exactly equal"):
        replace(_sequence(), values=values, sample_mask=sample_mask)


def test_sequence_enforces_condition_nan_and_mask_semantics() -> None:
    with pytest.raises(ValueError, match="unobserved conditions"):
        replace(
            _sequence(),
            condition_mask=torch.tensor([True, False], dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="observed conditions"):
        replace(
            _sequence(),
            condition_values=torch.tensor(
                [float("nan"), 1.0], dtype=torch.float32
            ),
        )


def test_input_hash_is_stable_order_sensitive_and_detects_mutation() -> None:
    source_values = _sequence().values.clone()
    first = _sequence(values=source_values)
    source_values[0, 0, 0, 0] = 99.0
    identical = _sequence()
    changed = _sequence(
        values=identical.values.flip(0),
        sample_mask=identical.sample_mask.flip(0),
    )

    assert first.input_hash == identical.input_hash
    assert first.input_hash != changed.input_hash
    first.verify_input_hash()
    first.values[0, 0, 0, 0] = 99.0
    with pytest.raises(ValueError, match="input_hash"):
        first.verify_input_hash()


def test_normalizer_fits_measurements_and_conditions_from_declared_training_cells() -> None:
    first = _sequence(cell_id="train-a")
    second_values = _sequence().values.clone()
    second_values[_sequence().sample_mask] += torch.tensor([0.2, 0.5, 0.1])
    second = _sequence(
        cell_id="train-b",
        values=second_values,
        condition_values=torch.tensor([35.0, 2.0], dtype=torch.float32),
    )

    normalizer = EarlyCycleNormalizer.fit(
        (first, second),
        training_cell_ids=frozenset({"train-a", "train-b"}),
    )
    transformed = normalizer.transform(first)

    assert normalizer.training_cell_ids_sha256
    assert normalizer.statistics_sha256
    assert len(normalizer.variable_means) == len(VARIABLE_NAMES)
    assert len(normalizer.variable_stds) == len(VARIABLE_NAMES)
    assert normalizer.condition_means == pytest.approx((30.0, 1.5))
    assert normalizer.condition_stds == pytest.approx((5.0, 0.5))
    assert torch.isnan(transformed.values[~transformed.sample_mask]).all()
    assert torch.isfinite(transformed.values[transformed.sample_mask]).all()
    assert transformed.condition_values.tolist() == pytest.approx([-1.0, -1.0])
    assert transformed.normalization_version == "train-zscore-v1"
    assert (
        transformed.normalization_statistics_sha256
        == normalizer.statistics_sha256
    )
    assert transformed.input_hash != first.input_hash

    with pytest.raises(ValueError, match="exactly match"):
        EarlyCycleNormalizer.fit(
            (first, second),
            training_cell_ids=frozenset({"train-a"}),
        )


def test_normalizer_rejects_unobserved_training_condition_and_schema_mismatch() -> None:
    missing_conditions = torch.tensor([25.0, float("nan")], dtype=torch.float32)
    missing_mask = torch.tensor([True, False], dtype=torch.bool)
    first = _sequence(
        cell_id="train-a",
        condition_values=missing_conditions,
        condition_mask=missing_mask,
    )
    second = _sequence(
        cell_id="train-b",
        condition_values=missing_conditions,
        condition_mask=missing_mask,
    )
    with pytest.raises(ValueError, match="no observed training values"):
        EarlyCycleNormalizer.fit(
            (first, second),
            training_cell_ids=frozenset({"train-a", "train-b"}),
        )

    normalizer = EarlyCycleNormalizer.fit(
        (_sequence(cell_id="train-a"),),
        training_cell_ids=frozenset({"train-a"}),
    )
    with pytest.raises(ValueError, match="schema"):
        normalizer.transform(
            replace(_sequence(), feature_version="different-v2")
        )


def test_normalizer_refuses_a_mutated_source_sequence() -> None:
    training = _sequence(cell_id="train-a")
    normalizer = EarlyCycleNormalizer.fit(
        (training,), training_cell_ids=frozenset({"train-a"})
    )
    source = _sequence(cell_id="test-a")
    source.condition_values[0] = 99.0

    with pytest.raises(ValueError, match="input_hash"):
        normalizer.transform(source)


@pytest.mark.parametrize("mutated_field", ["values", "sample_mask"])
def test_normalizer_fit_refuses_mutated_training_sequence(mutated_field: str) -> None:
    training = _sequence(cell_id="train-a")
    if mutated_field == "values":
        training.values[0, 0, 0, 0] = 99.0
    else:
        training.sample_mask[0, 0, 0] = False

    with pytest.raises(ValueError, match="input_hash"):
        EarlyCycleNormalizer.fit(
            (training,), training_cell_ids=frozenset({"train-a"})
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"normalization_statistics_sha256": "a" * 64},
            "raw sequence",
        ),
        (
            {"normalization_version": "train-zscore-v1"},
            "requires normalization_statistics_sha256",
        ),
        (
            {
                "normalization_version": "unknown-v9",
                "normalization_statistics_sha256": "a" * 64,
            },
            "normalization_version",
        ),
    ],
)
def test_sequence_rejects_invalid_normalization_binding(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_sequence(), **updates)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("statistics_sha256", "0" * 64, "statistics_sha256"),
        ("training_cell_ids_sha256", "not-a-hash", "training_cell_ids_sha256"),
        ("variable_means", (float("nan"), 0.0, 0.0), "finite"),
        ("variable_stds", (-1.0, 1.0, 1.0), "non-negative"),
        ("variable_stds", (1.0, 1.0), "variable"),
        ("condition_means", (25.0,), "condition"),
    ],
)
def test_normalizer_rejects_forged_or_invalid_statistics(
    field: str,
    value: object,
    message: str,
) -> None:
    normalizer = EarlyCycleNormalizer.fit(
        (_sequence(cell_id="train-a"),),
        training_cell_ids=frozenset({"train-a"}),
    )
    with pytest.raises(ValueError, match=message):
        replace(normalizer, **{field: value})


def test_normalizer_rejects_double_normalization() -> None:
    training = _sequence(cell_id="train-a")
    normalizer = EarlyCycleNormalizer.fit(
        (training,), training_cell_ids=frozenset({"train-a"})
    )
    normalized = normalizer.transform(_sequence(cell_id="test-a"))

    with pytest.raises(ValueError, match="already normalized"):
        normalizer.transform(normalized)


def test_tensor_dataclasses_do_not_use_generated_tensor_equality() -> None:
    assert _sequence() != _sequence()
