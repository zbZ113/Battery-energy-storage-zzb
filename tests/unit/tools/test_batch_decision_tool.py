from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import ConformalCalibration, PredictionInterval, ProvenanceRecord, SourceKind
from quanxin_life.data.schemas import DataQualityReport
from quanxin_life.decision import BatchDecisionPolicy
from quanxin_life.tools import ToolRegistry


def _interval() -> PredictionInterval:
    return PredictionInterval(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cutoff_cycle=100,
        point_prediction_cycle=360.0,
        lower_eol_cycle=320.0,
        upper_eol_cycle=400.0,
        calibration=ConformalCalibration(
            alpha=0.10,
            residual_quantile_cycle=40.0,
            calibration_cell_count=4,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            model_version="hybrid-v1",
            data_version="matr-v1",
        ),
    )


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="prediction-tool-result",
            source_kind=SourceKind.PREDICTED,
            uri="tool-result://prediction/fixture",
            sha256="a" * 64,
            description="Prediction ToolResult provenance retained for batch triage",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _input(*, target_domain_calibrated: bool = True):
    from quanxin_life.tools.batch_decision import BatchDecisionToolInput

    return BatchDecisionToolInput(
        prediction_interval=_interval(),
        quality_report=DataQualityReport(dataset_id="MATR"),
        target_domain_calibrated=target_domain_calibrated,
        policy=BatchDecisionPolicy(
            policy_version="batch-policy-v1",
            required_eol_cycle=300.0,
        ),
        upstream_result_ids=(str(uuid4()), str(uuid4())),
        provenance=_provenance(),
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def test_registered_batch_decision_tool_returns_auditable_registry_result() -> None:
    from quanxin_life.tools.batch_decision import register_batch_decision_tool

    registry = ToolRegistry()
    register_batch_decision_tool(registry)
    tool_input = _input()

    result = registry.execute("make_batch_decision", tool_input)

    assert result.tool_name == "make_batch_decision"
    assert result.tool_version == "batch-decision-tool-v1"
    assert result.model_version == "hybrid-v1"
    assert result.data_version == "matr-v1"
    assert result.feature_version == "early-cycle-v1"
    assert result.values["decision"] == "ADMIT"
    assert result.values["upstream_result_ids"] == list(tool_input.upstream_result_ids)
    assert result.provenance == list(_provenance())


def test_batch_decision_tool_preserves_uncalibrated_domain_as_recheck() -> None:
    from quanxin_life.tools.batch_decision import execute_batch_decision_tool

    result = execute_batch_decision_tool(_input(target_domain_calibrated=False))

    assert result.values["decision"] == "RECHECK"
    assert result.values["reason_codes"] == ["TARGET_DOMAIN_UNCALIBRATED"]
    assert "TARGET_DOMAIN_UNCALIBRATED" in result.warnings


def test_batch_decision_input_rejects_duplicate_or_non_uuid_upstream_ids() -> None:
    from quanxin_life.tools.batch_decision import BatchDecisionToolInput

    payload = _input().model_dump(mode="python")
    payload["upstream_result_ids"] = ("not-a-uuid", "not-a-uuid")

    with pytest.raises(ValueError, match="upstream_result_ids"):
        BatchDecisionToolInput.model_validate(payload)
