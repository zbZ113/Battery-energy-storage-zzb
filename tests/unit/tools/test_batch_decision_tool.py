"""Contracts for ledger-bound batch triage decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    NormalizedConformalCalibration,
    NormalizedPredictionInterval,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import DataQualityReport
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
)
from quanxin_life.tools.registry import StandardToolName, ToolRegistry

if TYPE_CHECKING:
    from quanxin_life.tools.batch_decision import (
        BatchDecisionToolInput,
        VerifiedBatchDecisionPolicy,
    )


def _provenance(source_id: str, source_kind: SourceKind) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=source_kind,
        uri=f"trusted-store://batch-decision/{source_id}",
        sha256=sha256_canonical({"fixture": source_id}),
        description=f"Trusted {source_id} fixture evidence for batch-decision contracts",
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _interval(*, target_domain_calibrated: bool = True) -> NormalizedPredictionInterval:
    return NormalizedPredictionInterval(
        dataset_id="synthetic-lfp",
        cell_id="cell-test-1",
        cutoff_cycle=20,
        point_prediction_cycle=360.0,
        lower_eol_cycle=320.0,
        upper_eol_cycle=400.0,
        difficulty_scale_cycle=15.0,
        calibration=NormalizedConformalCalibration(
            alpha=0.10,
            normalized_score_quantile=2.0,
            calibration_cell_count=3,
            scale_version="residual-scale-v1",
            feature_version="early-cycle-v1",
            split_version="split-v1",
            model_version="xgboost-eol80-v1",
            data_version="synthetic-data-v1",
        ),
    )


def _calibration_result() -> ToolResult:
    interval = _interval()
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        tool_version="conformal-calibration-tool-v1",
        model_version=interval.calibration.model_version,
        data_version=interval.calibration.data_version,
        feature_version=interval.calibration.feature_version,
        input_hash=sha256_canonical({"fixture": "normalized-calibration"}),
        values={
            "artifact_type": "quanxin_life.normalized_conformal_calibration.v1",
            "artifact": {
                "calibration_cohort_id": "trusted-calibration-cohort-20260715",
                "calibration_domain_id": "synthetic-lfp",
                "calibration_scope": "in_distribution",
                "calibration": interval.calibration.model_dump(mode="json"),
                "calibration_cell_ids": [
                    "calibration-1",
                    "calibration-2",
                    "calibration-3",
                ],
                "split_manifest_hash": sha256_canonical({"fixture": "split-manifest"}),
                "source_manifest_hash": sha256_canonical({"fixture": "calibration-manifest"}),
            },
        },
        provenance=[_provenance("calibration-evidence", SourceKind.OBSERVED)],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _interval_result(
    calibration_result_id: str, *, target_domain_calibrated: bool = True
) -> ToolResult:
    interval = _interval(target_domain_calibrated=target_domain_calibrated)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        tool_version="conformal-calibration-tool-v1",
        model_version=interval.calibration.model_version,
        data_version=interval.calibration.data_version,
        feature_version=interval.calibration.feature_version,
        input_hash=sha256_canonical({"fixture": "normalized-interval"}),
        values={
            "artifact_type": "quanxin_life.normalized_prediction_interval.v1",
            "artifact": {
                "prediction_interval": interval.model_dump(mode="json"),
                "prediction_result_id": str(uuid4()),
                "target_domain_calibrated": target_domain_calibrated,
                "calibration_result_id": calibration_result_id,
                "calibration_domain_id": "synthetic-lfp",
                "target_domain_id": "synthetic-lfp",
                "difficulty_scale_source_manifest_hash": sha256_canonical(
                    {"fixture": "difficulty-scale-manifest"}
                ),
            },
        },
        provenance=[_provenance("interval-evidence", SourceKind.PREDICTED)],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _quality_result() -> ToolResult:
    quality_report = DataQualityReport(dataset_id="synthetic-lfp")
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
        tool_version=DATA_QUALITY_TOOL_VERSION,
        model_version=DATA_QUALITY_MODEL_VERSION,
        data_version="synthetic-data-v1",
        feature_version="raw-cycle-v1",
        input_hash=sha256_canonical({"fixture": "quality-report"}),
        values={
            "dataset_id": quality_report.dataset_id,
            "blocked": quality_report.blocked,
            "quality_score": quality_report.quality_score,
            "issue_count": 0,
            "issues": [],
        },
        provenance=[_provenance("quality-evidence", SourceKind.OBSERVED)],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _policy() -> VerifiedBatchDecisionPolicy:
    from quanxin_life.decision import BatchDecisionPolicy
    from quanxin_life.tools.batch_decision import VerifiedBatchDecisionPolicy

    return VerifiedBatchDecisionPolicy(
        policy_id="approved-policy-synthetic-v1",
        policy=BatchDecisionPolicy(
            policy_version="batch-policy-v1",
            required_eol_cycle=300.0,
        ),
        policy_domain_id="synthetic-lfp",
        source_manifest_hash=sha256_canonical({"fixture": "approved-policy"}),
        provenance=(_provenance("approved-policy", SourceKind.OBSERVED),),
    )


class _PolicyResolver:
    def __init__(self, policy: VerifiedBatchDecisionPolicy) -> None:
        self._policy = policy
        self.calls: list[str] = []

    def resolve_verified_batch_decision_policy(self, policy_id: str) -> VerifiedBatchDecisionPolicy:
        self.calls.append(policy_id)
        if policy_id != self._policy.policy_id:
            raise ValueError("approved batch-decision policy was not found")
        return self._policy


def _input(
    interval_result_id: str, calibration_result_id: str, quality_result_id: str
) -> BatchDecisionToolInput:
    from quanxin_life.tools.batch_decision import BatchDecisionToolInput

    return BatchDecisionToolInput(
        prediction_interval_result_id=interval_result_id,
        calibration_result_id=calibration_result_id,
        quality_result_id=quality_result_id,
        policy_id="approved-policy-synthetic-v1",
    )


def test_batch_decision_tool_derives_all_numbers_from_registered_evidence_and_policy() -> None:
    from quanxin_life.tools.batch_decision import register_batch_decision_tool

    calibration_result = _calibration_result()
    interval_result = _interval_result(calibration_result.result_id)
    quality_result = _quality_result()
    resolver = _PolicyResolver(_policy())
    registry = ToolRegistry()
    register_batch_decision_tool(
        registry,
        audit_ledger=AuditLedger((interval_result, calibration_result, quality_result)),
        policy_resolver=resolver,
        clock=lambda: datetime(2026, 7, 15, 10, tzinfo=UTC),
    )

    result = registry.execute(
        StandardToolName.MAKE_BATCH_DECISION,
        _input(interval_result.result_id, calibration_result.result_id, quality_result.result_id),
    )

    assert resolver.calls == ["approved-policy-synthetic-v1"]
    assert result.tool_name == StandardToolName.MAKE_BATCH_DECISION.value
    assert result.tool_version == "batch-decision-tool-v2"
    assert result.model_version == "xgboost-eol80-v1"
    assert result.data_version == "synthetic-data-v1"
    assert result.feature_version == "early-cycle-v1"
    assert result.created_at == datetime(2026, 7, 15, 10, tzinfo=UTC)
    assert result.values["decision"] == "ADMIT"
    assert result.values["interval_lower_eol_cycle"] == 320.0
    assert result.values["interval_upper_eol_cycle"] == 400.0
    assert result.values["required_eol_cycle"] == 300.0
    assert result.values["target_domain_calibrated"] is True
    assert result.values["upstream_result_ids"] == [
        interval_result.result_id,
        calibration_result.result_id,
        quality_result.result_id,
    ]
    assert result.values["policy_id"] == "approved-policy-synthetic-v1"
    assert result.provenance == [
        _provenance("interval-evidence", SourceKind.PREDICTED),
        _provenance("quality-evidence", SourceKind.OBSERVED),
        _provenance("approved-policy", SourceKind.OBSERVED),
    ]


def test_batch_decision_input_rejects_caller_intervals_quality_thresholds_and_provenance() -> None:
    from quanxin_life.tools.batch_decision import BatchDecisionToolInput

    with pytest.raises(ValueError, match="Extra inputs"):
        BatchDecisionToolInput.model_validate(
            {
                "prediction_interval_result_id": str(uuid4()),
                "calibration_result_id": str(uuid4()),
                "quality_result_id": str(uuid4()),
                "policy_id": "approved-policy-synthetic-v1",
                "prediction_interval": _interval().model_dump(mode="json"),
                "quality_report": {"dataset_id": "synthetic-lfp"},
                "required_eol_cycle": 1.0,
                "target_domain_calibrated": True,
                "provenance": [],
                "decided_at": datetime(2026, 7, 15, tzinfo=UTC),
            }
        )


def test_batch_decision_tool_rejects_unregistered_or_tampered_upstream_evidence() -> None:
    from quanxin_life.tools.batch_decision import execute_batch_decision_tool

    calibration_result = _calibration_result()
    interval_result = _interval_result(calibration_result.result_id)
    quality_result = _quality_result()
    resolver = _PolicyResolver(_policy())

    with pytest.raises(ValueError, match="not registered"):
        execute_batch_decision_tool(
            _input(str(uuid4()), calibration_result.result_id, quality_result.result_id),
            audit_ledger=AuditLedger((interval_result, calibration_result, quality_result)),
            policy_resolver=resolver,
        )

    wrong_interval = interval_result.model_copy(
        update={"values": {"artifact_type": "legacy.interval.v0", "artifact": {}}}
    )
    with pytest.raises(ValueError, match="artifact_type"):
        execute_batch_decision_tool(
            _input(
                wrong_interval.result_id,
                calibration_result.result_id,
                quality_result.result_id,
            ),
            audit_ledger=AuditLedger((wrong_interval, calibration_result, quality_result)),
            policy_resolver=resolver,
        )

    mismatched_quality = quality_result.model_copy(
        update={"values": {**quality_result.values, "dataset_id": "other-dataset"}}
    )
    with pytest.raises(ValueError, match="dataset_id"):
        execute_batch_decision_tool(
            _input(
                interval_result.result_id,
                calibration_result.result_id,
                mismatched_quality.result_id,
            ),
            audit_ledger=AuditLedger(
                (interval_result, calibration_result, mismatched_quality)
            ),
            policy_resolver=resolver,
        )


def test_batch_decision_tool_uses_interval_domain_status_and_is_not_default_registered() -> None:
    from quanxin_life.tools.batch_decision import execute_batch_decision_tool
    from quanxin_life.tools.bootstrap import create_available_tool_registry

    calibration_result = _calibration_result()
    interval_result = _interval_result(
        calibration_result.result_id,
        target_domain_calibrated=False,
    )
    quality_result = _quality_result()
    result = execute_batch_decision_tool(
        _input(interval_result.result_id, calibration_result.result_id, quality_result.result_id),
        audit_ledger=AuditLedger((interval_result, calibration_result, quality_result)),
        policy_resolver=_PolicyResolver(_policy()),
        clock=lambda: datetime(2026, 7, 15, 10, tzinfo=UTC),
    )

    assert result.values["decision"] == "RECHECK"
    assert result.values["reason_codes"] == ["TARGET_DOMAIN_UNCALIBRATED"]
    assert StandardToolName.MAKE_BATCH_DECISION not in {
        schema.tool_name for schema in create_available_tool_registry().list_schemas()
    }


def test_batch_decision_tool_rejects_policy_identity_substitution_and_simulated_quality() -> None:
    from quanxin_life.tools.batch_decision import execute_batch_decision_tool

    calibration_result = _calibration_result()
    interval_result = _interval_result(calibration_result.result_id)
    quality_result = _quality_result()
    input_value = _input(
        interval_result.result_id,
        calibration_result.result_id,
        quality_result.result_id,
    )
    substituted_policy = _policy().model_copy(update={"policy_id": "other-approved-policy"})

    class _SubstitutingPolicyResolver:
        def resolve_verified_batch_decision_policy(
            self, policy_id: str
        ) -> VerifiedBatchDecisionPolicy:
            del policy_id
            return substituted_policy

    with pytest.raises(ValueError, match="policy_id"):
        execute_batch_decision_tool(
            input_value,
            audit_ledger=AuditLedger((interval_result, calibration_result, quality_result)),
            policy_resolver=_SubstitutingPolicyResolver(),
        )

    simulated_quality = quality_result.model_copy(
        update={"provenance": [_provenance("simulated-quality", SourceKind.SIMULATED)]}
    )
    with pytest.raises(ValueError, match="OBSERVED"):
        execute_batch_decision_tool(
            _input(
                interval_result.result_id,
                calibration_result.result_id,
                simulated_quality.result_id,
            ),
            audit_ledger=AuditLedger((interval_result, calibration_result, simulated_quality)),
            policy_resolver=_PolicyResolver(_policy()),
        )
