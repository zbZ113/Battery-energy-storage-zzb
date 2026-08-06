from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from quanxin_life.core import SelectionMetricDirection, TrainingReadableSplit
from quanxin_life.data.dataset_bundle import ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.adapters.base import (
    EvalState,
    EvaluationResult,
    PredictionBatch,
    ResolvedTrainingConfig,
    TrainState,
    selection_metric_value,
)
from quanxin_life.training.adapters.registry import (
    TrainingAdapterRegistry,
    build_governed_training_adapter_registry,
)


class _SafeAdapter:
    adapter_version = "safe-adapter-v1"
    selection_metric_name = "validation_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        del resolved_config
        return torch.nn.Linear(1, 1)

    def load_view(self, manifest: ModelViewManifest) -> object:
        return manifest

    def train_epoch(self, state: TrainState) -> EvaluationResult:
        del state
        return EvaluationResult(loss=1.0, metrics={"validation_mae": 1.0})

    def validate(self, state: EvalState) -> EvaluationResult:
        del state
        return EvaluationResult(loss=1.0, metrics={"validation_mae": 1.0})

    def predict(self, state: EvalState) -> PredictionBatch:
        return PredictionBatch(split=state.split, entity_ids=("cell-1",), values=(1.0,))

    def export_best(self, destination: Path) -> ArtifactManifest:
        del destination
        raise NotImplementedError


def test_train_state_rejects_any_non_train_split() -> None:
    with pytest.raises(ValidationError, match="train"):
        TrainState(epoch=1, split=TrainingReadableSplit.TEST)


def test_selection_eval_state_rejects_test_split() -> None:
    with pytest.raises(ValidationError, match="test"):
        EvalState(epoch=5, split=TrainingReadableSplit.TEST, for_selection=True)


def test_selection_metric_must_exist_and_be_finite() -> None:
    adapter = _SafeAdapter()
    with pytest.raises(ValueError, match="validation_mae"):
        selection_metric_value(adapter, EvaluationResult(loss=1.0, metrics={}))

    with pytest.raises(ValidationError, match="finite"):
        EvaluationResult(loss=1.0, metrics={"validation_mae": float("nan")})


def test_registry_accepts_protocol_adapter_and_rejects_duplicate_family() -> None:
    registry = TrainingAdapterRegistry()
    adapter = _SafeAdapter()
    registry.register("cyclepatch_direct", adapter)

    assert registry.get("cyclepatch_direct") is adapter
    assert registry.families == ("cyclepatch_direct",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("cyclepatch_direct", _SafeAdapter())


def test_governed_registry_includes_blocked_external_adapters() -> None:
    registry = build_governed_training_adapter_registry()

    assert registry.families == (
        "batterymformer",
        "battgp",
        "blast_lite",
        "diting_cptransformer",
        "magnet",
        "pbt",
        "smart_feature",
    )
