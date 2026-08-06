from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from quanxin_life.core import TrainingBlockedReason, TrainingTaskType
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.smart_feature import (
    SMART_FEATURE_LICENSE_STATUS,
    SMART_FEATURE_UPSTREAM_COMMIT,
    SmartFeatureAdapter,
    validate_matlab_reference,
)


def test_smart_feature_is_partial_charge_and_license_gated() -> None:
    adapter = SmartFeatureAdapter()
    assert SMART_FEATURE_UPSTREAM_COMMIT == "dc6beea547960cf44d1721734dba93bdcab19a1f"
    assert SMART_FEATURE_LICENSE_STATUS == "RESEARCH_ONLY_LICENSE_UNVERIFIED"
    assert adapter.task_type is TrainingTaskType.PARTIAL_CHARGE_FEATURE
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.readiness(has_matlab_reference=False) is TrainingBlockedReason.BLOCKED_DEPENDENCY


def test_smart_feature_requires_fixed_matlab_export_and_does_not_invent_formula(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="MATLAB reference"):
        validate_matlab_reference(missing)
    reference = tmp_path / "matlab_reference.json"
    reference.write_text(
        json.dumps({"schema_version": "matlab-feature-reference-v1", "cases": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"case|empty"):
        validate_matlab_reference(reference)


def test_smart_feature_runtime_has_no_formula_or_unsafe_loader() -> None:
    module = inspect.getmodule(SmartFeatureAdapter)
    assert module is not None
    source = inspect.getsource(module)
    tree = ast.parse(source)
    roots = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert roots.isdisjoint({"transformers", "qwen", "joblib", "pickle"})
    assert "feature_formula" not in source


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_smart_feature_configs_are_evidence_only_and_blocked(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "smart_feature" / f"{stage}.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW", "BLOCKED_DEPENDENCY"]
    assert payload["promotion_blocked_reasons"] == ["BLOCKED_LICENSE"]
    assert payload["task_type"] == TrainingTaskType.PARTIAL_CHARGE_FEATURE.value
    assert payload["upstream_commit"] == SMART_FEATURE_UPSTREAM_COMMIT


def test_smart_feature_training_matrix_is_evidence_only() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "smart_feature")
    assert entry["task_type"] == TrainingTaskType.PARTIAL_CHARGE_FEATURE.value
    assert entry["optimizer"] == "none"
    assert entry["loss_names"] == ["not_applicable"]
    assert entry["source_commit"] == SMART_FEATURE_UPSTREAM_COMMIT
