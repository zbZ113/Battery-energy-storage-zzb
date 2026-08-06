from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from quanxin_life.core import TrainingBlockedReason, TrainingTaskType
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.battgp import (
    BATTGP_LICENSE_STATUS,
    BATTGP_UPSTREAM_COMMIT,
    BattGPAdapter,
    BattGPModel,
    battgp_metrics,
    battgp_online_update,
    compute_battgp_nll,
    validate_battgp_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_battgp_is_field_monitoring_and_data_gated() -> None:
    adapter = BattGPAdapter()
    assert BATTGP_UPSTREAM_COMMIT == "6d5e1db3337f0de5f3a533acbc09eddccc178e9d"
    assert BATTGP_LICENSE_STATUS == "VERIFIED_LICENSE_PRESENT"
    assert adapter.task_type is TrainingTaskType.FIELD_MONITORING
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.readiness(has_field_view=False) is TrainingBlockedReason.BLOCKED_DATA_VIEW


def test_battgp_model_predicts_mean_and_variance_and_nll_is_finite() -> None:
    model = BattGPModel(noise_variance=0.05)
    x_train = torch.tensor([[0.0, 0.0, 50.0, 25.0], [1.0, 1.0, 55.0, 25.0]], dtype=torch.float64)
    y_train = torch.tensor([0.1, 0.9], dtype=torch.float64)
    model.fit(x_train, y_train)
    mean, variance = model.predict(torch.tensor([[0.5, 0.5, 52.0, 25.0]], dtype=torch.float64))
    assert mean.shape == (1,)
    assert variance.shape == (1,)
    assert torch.isfinite(mean).all() and torch.all(variance >= 0)
    assert torch.isfinite(compute_battgp_nll(model, x_train, y_train))


def test_battgp_wiener_kernel_matches_frozen_formula() -> None:
    left = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    observed = BattGPModel.kernel(left, right)
    expected = 0.0099 + 4.23e-13 * (1.0**2 * (1.0 / 3.0 + 1.0 / 2.0))
    assert torch.allclose(observed, torch.tensor([[expected]], dtype=torch.float64))


def test_battgp_online_update_does_not_require_future_labels() -> None:
    model = BattGPModel(noise_variance=0.05)
    x_train = torch.tensor([[0.0, 0.0, 50.0, 25.0]], dtype=torch.float64)
    y_train = torch.tensor([0.1], dtype=torch.float64)
    model.fit(x_train, y_train)
    before = model.predict(torch.tensor([[1.0, 1.0, 55.0, 25.0]], dtype=torch.float64))[1]
    battgp_online_update(
        model, torch.tensor([[1.0, 1.0, 55.0, 25.0]], dtype=torch.float64), torch.tensor([0.9])
    )
    after = model.predict(torch.tensor([[1.0, 1.0, 55.0, 25.0]], dtype=torch.float64))[1]
    assert torch.all(after <= before + 1e-12)


def test_battgp_anomaly_metrics_require_explicit_labels() -> None:
    result = battgp_metrics(residuals=torch.tensor([0.1, 0.2]), anomaly_labels=None)
    assert result["anomaly_status"] == "REQUIRES_EXPLICIT_ANOMALY_LABELS"
    assert result["auroc"] is None and result["auprc"] is None and result["fpr95"] is None


def test_battgp_anomalies_rank_absolute_residuals() -> None:
    result = battgp_metrics(
        residuals=torch.tensor([-10.0, 1.0]),
        anomaly_labels=torch.tensor([True, False]),
    )
    assert result["auroc"] == 1.0


def test_battgp_requires_reviewed_four_field_input() -> None:
    model = BattGPModel()
    with pytest.raises(ValueError, match=r"time.*current.*SOC.*temperature"):
        model.fit(torch.ones((2, 3)), torch.ones(2))


def test_battgp_artifacts_are_structured_and_hashed(tmp_path: Path) -> None:
    params = tmp_path / "hyperparameters.json"
    params.write_text(
        json.dumps({"kernel": "wiener_plus_rbf", "noise_variance": 0.1}), encoding="utf-8"
    )
    evidence = tmp_path / "residuals.json"
    evidence.write_text(json.dumps({"schema_version": "battgp-residuals-v1"}), encoding="utf-8")
    verified = validate_battgp_artifacts(
        {"parameters": (params, _sha(params)), "evidence": (evidence, _sha(evidence))}
    )
    assert set(verified) == {"parameters", "evidence"}
    bad = tmp_path / "parameters.pkl"
    bad.write_bytes(b"unsafe")
    with pytest.raises(ValueError, match=r"unsafe|JSON"):
        validate_battgp_artifacts(
            {"parameters": (bad, _sha(bad)), "evidence": (evidence, _sha(evidence))}
        )


def test_battgp_runtime_has_no_unsafe_or_llm_imports() -> None:
    module = inspect.getmodule(BattGPAdapter)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    roots = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert roots.isdisjoint({"transformers", "qwen", "joblib", "pickle"})


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_battgp_configs_bind_public_task_and_field_gate(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "battgp" / f"{stage}.json").read_text(encoding="utf-8")
    )
    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW"]
    assert payload["task_type"] == TrainingTaskType.FIELD_MONITORING.value
    assert payload["upstream_commit"] == BATTGP_UPSTREAM_COMMIT
    assert payload["optimization"]["effective_batch_size"] == 64
    assert payload["runtime_dependencies"] == ["torch", "gpytorch>=1.11", "botorch"]


def test_battgp_training_matrix_binds_field_monitoring_identity() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "battgp")
    assert entry["task_type"] == TrainingTaskType.FIELD_MONITORING.value
    assert entry["loss_names"] == ["negative_log_marginal_likelihood"]
    assert entry["selection_metric_name"] == "validation_nll"
    assert entry["source_commit"] == BATTGP_UPSTREAM_COMMIT


def test_battgp_missing_optional_gp_dependencies_is_explicitly_blocked() -> None:
    assert (
        BattGPAdapter().readiness(has_field_view=True, gpytorch_available=False)
        is TrainingBlockedReason.BLOCKED_DEPENDENCY
    )
