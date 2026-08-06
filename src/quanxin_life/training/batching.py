"""Effective-batch and optimizer-step contracts for adapter training."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import overload

import torch
from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import LastBatchPolicy
from quanxin_life.core.schemas import ContractModel


class BatchPlan(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    micro_batch_size: int = Field(gt=0)
    visible_gpu_count: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    effective_batch_size: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    last_batch_policy: LastBatchPolicy
    optimizer_steps_per_epoch: int = Field(gt=0)

    @model_validator(mode="after")
    def sizes_and_steps_are_consistent(self) -> BatchPlan:
        expected_effective = calculate_effective_batch_size(
            micro_batch_size=self.micro_batch_size,
            visible_gpu_count=self.visible_gpu_count,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        if self.effective_batch_size != expected_effective:
            raise ValueError("effective_batch_size does not match the governed formula")
        expected_steps = _optimizer_steps_per_epoch(
            sample_count=self.sample_count,
            global_micro_batch_size=self.micro_batch_size * self.visible_gpu_count,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            last_batch_policy=self.last_batch_policy,
        )
        if self.optimizer_steps_per_epoch != expected_steps:
            raise ValueError(
                "optimizer_steps_per_epoch does not match the batch and accumulation plan"
            )
        return self


def calculate_effective_batch_size(
    *,
    micro_batch_size: int,
    visible_gpu_count: int,
    gradient_accumulation_steps: int,
) -> int:
    values = (micro_batch_size, visible_gpu_count, gradient_accumulation_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("effective batch inputs must be positive integers")
    return micro_batch_size * visible_gpu_count * gradient_accumulation_steps


def _optimizer_steps_per_epoch(
    *,
    sample_count: int,
    global_micro_batch_size: int,
    gradient_accumulation_steps: int,
    last_batch_policy: LastBatchPolicy,
) -> int:
    complete_micro_batches, remainder = divmod(sample_count, global_micro_batch_size)
    if remainder and last_batch_policy is LastBatchPolicy.ERROR:
        raise ValueError("sample_count is not divisible by the global micro batch")
    micro_batches = complete_micro_batches
    if remainder and last_batch_policy is LastBatchPolicy.KEEP:
        micro_batches += 1
    if (
        last_batch_policy is LastBatchPolicy.ERROR
        and micro_batches % gradient_accumulation_steps
    ):
        raise ValueError("micro batch count is not divisible by gradient accumulation steps")
    if last_batch_policy is LastBatchPolicy.KEEP:
        optimizer_steps = math.ceil(micro_batches / gradient_accumulation_steps)
    else:
        optimizer_steps = micro_batches // gradient_accumulation_steps
    if optimizer_steps < 1:
        raise ValueError("batch plan must produce at least one optimizer step")
    return optimizer_steps


@overload
def scale_loss_for_accumulation(
    loss: torch.Tensor,
    *,
    gradient_accumulation_steps: int,
) -> torch.Tensor: ...


@overload
def scale_loss_for_accumulation(
    loss: float,
    *,
    gradient_accumulation_steps: int,
) -> float: ...


def scale_loss_for_accumulation(
    loss: torch.Tensor | float,
    *,
    gradient_accumulation_steps: int,
) -> torch.Tensor | float:
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return loss / gradient_accumulation_steps


def scale_mean_loss_by_sample_count(
    loss: torch.Tensor,
    *,
    sample_count: int,
) -> torch.Tensor:
    """Convert a mean microbatch loss to a sample-sum for exact accumulation."""

    _validate_sample_count(sample_count)
    return loss * sample_count


def normalize_accumulated_gradients(
    parameters: Iterable[torch.nn.Parameter],
    *,
    sample_count: int,
) -> None:
    """Normalize one optimizer window by its actual observed sample count."""

    _validate_sample_count(sample_count)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


def _validate_sample_count(sample_count: int) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")


__all__ = [
    "BatchPlan",
    "calculate_effective_batch_size",
    "normalize_accumulated_gradients",
    "scale_loss_for_accumulation",
    "scale_mean_loss_by_sample_count",
]
