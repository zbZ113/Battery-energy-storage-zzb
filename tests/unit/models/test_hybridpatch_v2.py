from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from quanxin_life.models.cyclepatch import CyclePatchConfig, EarlyCycleBatch
from quanxin_life.models.hybridpatch_v2 import (
    ConditionAdaLN,
    HybridPatchV2Config,
    HybridPatchV2Inputs,
    HybridPatchV2Output,
    HybridPatchV2Predictor,
    HybridPatchV2Targets,
    StructuralMonotoneSOHDecoder,
    compute_hybridpatch_v2_loss,
)


def _batch(batch_size: int = 2) -> EarlyCycleBatch:
    torch.manual_seed(20260712)
    values = torch.full((batch_size, 21, 2, 150, 3), float("nan"))
    sample_mask = torch.zeros((batch_size, 21, 2, 150), dtype=torch.bool)
    for cycle in (1, 4):
        sample_mask[:, cycle] = True
        values[:, cycle] = torch.randn(batch_size, 2, 150, 3)
    return EarlyCycleBatch(
        dataset_id="MATR",
        data_version="matr-v1",
        feature_version="multi-v1",
        normalization_statistics_sha256="a" * 64,
        cell_ids=tuple(f"cell-{index}" for index in range(batch_size)),
        condition_names=("temperature", "charge", "discharge"),
        values=values,
        cycle_indices=torch.arange(21).expand(batch_size, -1),
        cycle_mask=sample_mask.any(dim=(2, 3)),
        sample_mask=sample_mask,
        condition_values=torch.tensor([[0.2, 0.1, float("nan")]]).expand(
            batch_size, -1
        ),
        condition_mask=torch.tensor([[True, True, False]]).expand(batch_size, -1),
    )


def _config(query_count: int = 0) -> HybridPatchV2Config:
    return HybridPatchV2Config(
        cyclepatch=CyclePatchConfig(d_model=12, layers=1, heads=3, dropout=0.0),
        query_token_count=query_count,
        query_layers=1,
        decoder_hidden_dim=16,
        huber_delta=1.0,
        lambda_history=0.2,
        lambda_smooth=0.1,
        lambda_order=0.1,
        lambda_residual=0.01,
    )


def _inputs(batch_size: int = 2) -> HybridPatchV2Inputs:
    return HybridPatchV2Inputs(
        early_batch=_batch(batch_size),
        initial_soh=torch.tensor([1.0, 0.98])[:batch_size],
        prediction_cycles=torch.tensor([21, 22, 60, 500], dtype=torch.int64),
    )


def _targets(inputs: HybridPatchV2Inputs) -> HybridPatchV2Targets:
    batch_size, cycle_count = inputs.early_batch.cycle_mask.shape
    history_mask = inputs.early_batch.cycle_mask.clone()
    history = torch.full((batch_size, cycle_count), float("nan"))
    history[history_mask] = 0.99
    target_mask = torch.ones((batch_size, len(inputs.prediction_cycles)), dtype=torch.bool)
    target = torch.tensor([[0.97, 0.96, 0.90, 0.80]]).expand(batch_size, -1).clone()
    return HybridPatchV2Targets(
        history_soh=history,
        history_mask=history_mask,
        target_soh=target,
        target_mask=target_mask,
    )


@pytest.mark.parametrize("query_count", [0, 8, 16])
def test_forward_shapes_finite_and_gradients(query_count: int) -> None:
    torch.manual_seed(20260712)
    model = HybridPatchV2Predictor(_config(query_count), condition_count=3)
    inputs = _inputs()

    output = model(inputs)
    output.predicted_soh.sum().backward()

    assert output.predicted_soh.shape == (2, 4)
    assert output.history_reconstruction.shape == (2, 21)
    assert output.query_tokens is None if query_count == 0 else output.query_tokens.shape == (
        2,
        query_count,
        12,
    )
    assert torch.isfinite(output.predicted_soh).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_zero_query_mode_has_no_query_or_decoder_parameters() -> None:
    model = HybridPatchV2Predictor(_config(0), condition_count=3)
    names = tuple(name for name, _ in model.named_parameters())

    assert not any("query_tokens" in name or "query_decoder" in name for name in names)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"query_token_count": 4}, "query_token_count"),
        ({"query_layers": 0}, "query_layers"),
        ({"decoder_hidden_dim": 0}, "decoder_hidden_dim"),
        ({"huber_delta": 0.0}, "huber_delta"),
        ({"lambda_history": -1.0}, "lambda_history"),
        ({"lambda_smooth": -1.0}, "lambda_smooth"),
        ({"lambda_order": -1.0}, "lambda_order"),
        ({"lambda_residual": 0.0}, "lambda_residual"),
        ({"max_prediction_cycle": 499}, "500"),
    ],
)
def test_config_rejects_invalid_values(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **updates)


def test_nonuniform_prediction_axis_and_invalid_cycles() -> None:
    inputs = _inputs()
    assert inputs.prediction_cycles.tolist() == [21, 22, 60, 500]

    for cycles, exception in (
        (torch.tensor([21, 20], dtype=torch.int64), RuntimeError),
        (torch.tensor([21, 21], dtype=torch.int64), RuntimeError),
        (torch.tensor([21, 501], dtype=torch.int64), RuntimeError),
        (torch.tensor([21.0, 22.0]), ValueError),
    ):
        with pytest.raises(exception, match=r"prediction_cycles|500|increasing"):
            replace(inputs, prediction_cycles=cycles)


def test_structural_output_is_monotone_and_components_nonnegative() -> None:
    torch.manual_seed(20260712)
    output = HybridPatchV2Predictor(_config(8), condition_count=3).eval()(_inputs())

    assert torch.all(output.predicted_soh[:, 1:] <= output.predicted_soh[:, :-1])
    for component in (
        output.sqrt_degradation,
        output.linear_degradation,
        output.knee_degradation,
        output.residual_degradation,
        output.residual_increments,
    ):
        assert torch.all(component >= 0)
    assert torch.all(
        output.residual_degradation[:, 1:] >= output.residual_degradation[:, :-1]
    )
    assert not hasattr(output, "rul") and not hasattr(output, "eol")


def test_masked_early_and_condition_values_do_not_affect_output() -> None:
    torch.manual_seed(20260712)
    model = HybridPatchV2Predictor(_config(8), condition_count=3).eval()
    inputs = _inputs()
    values = inputs.early_batch.values.clone()
    values[~inputs.early_batch.sample_mask] = 9999.0
    conditions = inputs.early_batch.condition_values.clone()
    conditions[~inputs.early_batch.condition_mask] = -9999.0
    altered_batch = replace(
        inputs.early_batch, values=values, condition_values=conditions
    )

    with torch.no_grad():
        expected = model(inputs)
        actual = model(replace(inputs, early_batch=altered_batch))

    assert torch.allclose(expected.predicted_soh, actual.predicted_soh)
    assert torch.allclose(expected.history_reconstruction, actual.history_reconstruction)


def test_condition_adaln_masks_missing_condition_value() -> None:
    torch.manual_seed(20260712)
    layer = ConditionAdaLN(d_model=12, condition_count=3)
    x = torch.randn(2, 4, 12)
    values = torch.randn(2, 3)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    altered = values.clone()
    altered[~mask] = 9999.0

    assert torch.allclose(layer(x, values, mask), layer(x, altered, mask))


def test_loss_masks_supervision_and_rejects_history_leakage() -> None:
    torch.manual_seed(20260712)
    config = _config(0)
    model = HybridPatchV2Predictor(config, condition_count=3)
    inputs = _inputs()
    output = model(inputs)
    targets = _targets(inputs)
    masked_target = targets.target_soh.clone()
    masked_target[:, -1] = float("nan")
    target_mask = targets.target_mask.clone()
    target_mask[:, -1] = False
    masked = replace(targets, target_soh=masked_target, target_mask=target_mask)

    loss = compute_hybridpatch_v2_loss(output, masked, inputs, config)

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.history)
    no_history = replace(
        masked,
        history_soh=torch.full_like(masked.history_soh, float("nan")),
        history_mask=torch.zeros_like(masked.history_mask),
    )
    no_history_loss = compute_hybridpatch_v2_loss(
        output, no_history, inputs, config
    )
    assert torch.equal(
        no_history_loss.history.detach(), torch.zeros_like(no_history_loss.history)
    )
    leaked_history = targets.history_mask.clone()
    leaked_history[:, 2] = True
    leaked_values = targets.history_soh.clone()
    leaked_values[:, 2] = 1.0
    leaked = replace(
        targets, history_mask=leaked_history, history_soh=leaked_values
    )
    with pytest.raises(RuntimeError, match=r"history_mask|early"):
        compute_hybridpatch_v2_loss(output, leaked, inputs, config)

    with pytest.raises(RuntimeError, match=r"target row|valid target"):
        replace(
            targets,
            target_soh=torch.full_like(targets.target_soh, float("nan")),
            target_mask=torch.zeros_like(targets.target_mask),
        )


def test_hot_path_does_not_call_tensor_item(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(20260712)
    raw_inputs = _inputs(batch_size=1)
    raw_targets = _targets(raw_inputs)
    model = HybridPatchV2Predictor(_config(8), condition_count=3)

    def forbidden_item(_tensor: torch.Tensor) -> object:
        raise AssertionError("Tensor.item synchronizes the GPU hot path")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    inputs = replace(raw_inputs)
    targets = replace(raw_targets)
    output = model(inputs)
    loss = compute_hybridpatch_v2_loss(output, targets, inputs, _config(8))
    loss.total.backward()


def test_empty_auxiliary_masks_return_differentiable_zero() -> None:
    torch.manual_seed(20260712)
    base = _inputs(batch_size=1)
    inputs = replace(
        base,
        prediction_cycles=torch.tensor([21], dtype=torch.int64),
    )
    targets = HybridPatchV2Targets(
        history_soh=torch.full((1, 21), float("nan")),
        history_mask=torch.zeros((1, 21), dtype=torch.bool),
        target_soh=torch.tensor([[0.97]]),
        target_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    model = HybridPatchV2Predictor(_config(0), condition_count=3)
    output = model(inputs)

    loss = compute_hybridpatch_v2_loss(output, targets, inputs, _config(0))

    for auxiliary in (loss.history, loss.order, loss.smooth):
        assert torch.equal(auxiliary.detach(), torch.zeros_like(auxiliary))
        assert auxiliary.requires_grad
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_hybridpatch_source_contains_no_host_sync_hot_path() -> None:
    from quanxin_life.models import hybridpatch_v2

    source = inspect.getsource(hybridpatch_v2)
    assert ".item(" not in source
    assert "if torch.any" not in source
    assert "bool(torch" not in source


def test_same_seed_is_deterministic() -> None:
    inputs = _inputs()
    torch.manual_seed(7)
    first = HybridPatchV2Predictor(_config(8), condition_count=3).eval()
    torch.manual_seed(7)
    second = HybridPatchV2Predictor(_config(8), condition_count=3).eval()

    with torch.no_grad():
        first_output = first(inputs).predicted_soh
        second_output = second(inputs).predicted_soh

    assert torch.allclose(first_output, second_output)


def test_tiny_batch_training_reduces_governed_loss() -> None:
    torch.manual_seed(20260712)
    config = _config(0)
    model = HybridPatchV2Predictor(config, condition_count=3)
    inputs = _inputs(batch_size=1)
    targets = _targets(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []

    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_hybridpatch_v2_loss(model(inputs), targets, inputs, config).total
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


def test_knee_is_exactly_zero_before_location_and_quadratic_after() -> None:
    decoder = StructuralMonotoneSOHDecoder(d_model=4, hidden_dim=4)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
    context = torch.zeros(1, 4)
    cycles = torch.tensor([21, 100, 260, 380, 500], dtype=torch.int64)

    output = decoder(
        context,
        torch.tensor([1.0]),
        cycles,
        cutoff_cycle=20,
    )

    knee = output.knee_degradation[0]
    assert torch.count_nonzero(knee[:3]) == 0
    assert knee[3] > 0
    assert knee[4] > knee[3]
    first_distance = (380.0 - 260.0) / (500.0 - 20.0)
    second_distance = (500.0 - 260.0) / (500.0 - 20.0)
    assert (knee[4] / knee[3]).item() == pytest.approx(
        (second_distance / first_distance) ** 2, rel=1e-5
    )


@pytest.mark.parametrize("query_count", [8, 16])
def test_queries_break_symmetry_and_each_receive_gradient(query_count: int) -> None:
    torch.manual_seed(20260712)
    model = HybridPatchV2Predictor(_config(query_count), condition_count=3)
    output = model(_inputs())
    assert output.query_tokens is not None

    assert not torch.allclose(output.query_tokens[:, 0], output.query_tokens[:, 1])
    output.predicted_soh.sum().backward()
    gradients = model.query_encoder.query_tokens.grad
    assert gradients is not None
    assert torch.all(torch.linalg.vector_norm(gradients[0], dim=1) > 0)


def test_smooth_loss_is_masked_nonuniform_second_difference_squared_mean() -> None:
    inputs = _inputs(batch_size=1)
    prediction = torch.tensor([[1.0, 0.9, 0.8, 0.0]])
    zeros = torch.zeros_like(prediction)
    output = HybridPatchV2Output(
        predicted_soh=prediction,
        history_reconstruction=torch.zeros(1, 21),
        sqrt_degradation=zeros,
        linear_degradation=zeros,
        knee_degradation=zeros,
        residual_degradation=zeros,
        residual_increments=zeros,
        query_tokens=None,
        cycle_mask=inputs.early_batch.cycle_mask,
    )
    history_mask = torch.zeros(1, 21, dtype=torch.bool)
    target_mask = torch.tensor([[True, True, True, False]])
    targets = HybridPatchV2Targets(
        history_soh=torch.full((1, 21), float("nan")),
        history_mask=history_mask,
        target_soh=torch.tensor([[1.0, 0.9, 0.8, float("nan")]]),
        target_mask=target_mask,
    )

    loss = compute_hybridpatch_v2_loss(output, targets, inputs, _config(0))

    slope_01 = (0.9 - 1.0) / (22 - 21)
    slope_12 = (0.8 - 0.9) / (60 - 22)
    expected = (slope_12 - slope_01) ** 2
    assert loss.smooth.item() == pytest.approx(expected, rel=1e-6)
