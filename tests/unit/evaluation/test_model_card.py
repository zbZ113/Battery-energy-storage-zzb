from __future__ import annotations

import pytest


def _payload() -> dict[str, object]:
    return {
        "task": "RUL",
        "model_version": "demo-v1",
        "target": "MATR_OFFICIAL_CYCLE_LIFE",
        "dataset": {"id": "MATR", "version": "v1", "cell_split": "cell_id"},
        "source_commit": "a" * 40,
        "license": "reviewed",
        "training_parameters": {"seed_count": 5},
        "validation_rules": ["validation-only selection"],
        "metrics_test": {"status": "verified"},
        "seed_statistics": {"status": "complete"},
        "per_cell_tail": {"status": "available"},
        "calibration": {"interval_method": "NOT_AVAILABLE"},
        "ood_supported_domain": {"status": "declared"},
        "failed_experiments": [],
        "inference_environment": {"status": "declared"},
        "rejection_conditions": ["domain mismatch"],
        "artifact_sha256": "b" * 64,
    }


def test_model_card_contains_governance_sections_and_inactive_status() -> None:
    from quanxin_life.evaluation.model_card import render_model_card

    card = render_model_card(_payload())
    assert "VERIFIED_NOT_ACTIVATED" in card
    assert "demo-v1" in card
    for heading in ("Data and Split", "Validation", "Test Metrics", "Calibration", "OOD"):
        assert heading in card
    assert "MATR_OFFICIAL_CYCLE_LIFE" in card


def test_model_card_rejects_active_status_or_missing_evidence() -> None:
    from quanxin_life.evaluation.model_card import render_model_card

    payload = _payload()
    payload["activation_status"] = "ACTIVE"
    with pytest.raises(ValueError, match="activation"):
        render_model_card(payload)

    incomplete = _payload()
    del incomplete["metrics_test"]
    with pytest.raises(ValueError, match="metrics_test"):
        render_model_card(incomplete)
