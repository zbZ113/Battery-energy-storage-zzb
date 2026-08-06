from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from quanxin_life.core import LastBatchPolicy
from quanxin_life.training.batching import (
    BatchPlan,
    calculate_effective_batch_size,
    normalize_accumulated_gradients,
    scale_loss_for_accumulation,
    scale_mean_loss_by_sample_count,
)


@pytest.mark.parametrize(
    ("micro_batch_size", "gradient_accumulation_steps", "expected"),
    [(128, 2, 256), (32, 2, 64), (128, 2, 256)],
)
def test_approved_single_gpu_effective_batches(
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    expected: int,
) -> None:
    assert (
        calculate_effective_batch_size(
            micro_batch_size=micro_batch_size,
            visible_gpu_count=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        == expected
    )


def test_batch_plan_rejects_effective_batch_mismatch() -> None:
    with pytest.raises(ValidationError, match="effective_batch"):
        BatchPlan(
            micro_batch_size=32,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=63,
            sample_count=128,
            last_batch_policy=LastBatchPolicy.ERROR,
            optimizer_steps_per_epoch=2,
        )


def test_batch_plan_requires_last_batch_policy() -> None:
    with pytest.raises(ValidationError, match="last_batch_policy"):
        BatchPlan(
            micro_batch_size=32,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=64,
            sample_count=128,
            optimizer_steps_per_epoch=2,
        )


def test_batch_plan_rejects_undeclared_remainder_and_wrong_optimizer_steps() -> None:
    with pytest.raises(ValidationError, match="not divisible"):
        BatchPlan(
            micro_batch_size=32,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=64,
            sample_count=130,
            last_batch_policy=LastBatchPolicy.ERROR,
            optimizer_steps_per_epoch=2,
        )

    with pytest.raises(ValidationError, match="optimizer_steps_per_epoch"):
        BatchPlan(
            micro_batch_size=32,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=64,
            sample_count=130,
            last_batch_policy=LastBatchPolicy.KEEP,
            optimizer_steps_per_epoch=2,
        )


def test_error_policy_rejects_incomplete_accumulation_window() -> None:
    with pytest.raises(ValidationError, match="not divisible"):
        BatchPlan(
            micro_batch_size=32,
            visible_gpu_count=1,
            gradient_accumulation_steps=2,
            effective_batch_size=64,
            sample_count=96,
            last_batch_policy=LastBatchPolicy.ERROR,
            optimizer_steps_per_epoch=1,
        )


def test_loss_is_scaled_before_accumulated_backward() -> None:
    loss = torch.tensor(8.0, requires_grad=True)
    scaled = scale_loss_for_accumulation(loss, gradient_accumulation_steps=4)
    scaled.backward()

    assert scaled.item() == pytest.approx(2.0)
    assert loss.grad is not None
    assert loss.grad.item() == pytest.approx(0.25)


def test_partial_accumulation_window_matches_sample_weighted_full_batch() -> None:
    full = torch.nn.Linear(1, 1, bias=False)
    accumulated = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        full.weight.fill_(0.5)
        accumulated.weight.copy_(full.weight)
    inputs = torch.tensor([[1.0], [2.0], [4.0]])
    targets = torch.tensor([[2.0], [4.0], [8.0]])

    torch.nn.functional.mse_loss(full(inputs), targets).backward()
    for indices in (slice(0, 2), slice(2, 3)):
        loss = torch.nn.functional.mse_loss(
            accumulated(inputs[indices]), targets[indices]
        )
        scale_mean_loss_by_sample_count(
            loss,
            sample_count=len(inputs[indices]),
        ).backward()
    normalize_accumulated_gradients(accumulated.parameters(), sample_count=3)

    assert accumulated.weight.grad is not None
    assert full.weight.grad is not None
    assert accumulated.weight.grad.item() == pytest.approx(full.weight.grad.item())
