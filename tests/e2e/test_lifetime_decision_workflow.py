from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.lifetime_workflow import (
    DECISION_REQUIRED_EOL_PATH,
    INTERVAL_LOWER_EOL_PATH,
    INTERVAL_UPPER_EOL_PATH,
    PREDICTION_POINT_EOL_PATH,
    LifetimeDecisionWorkflowRequest,
    LifetimeDecisionWorkflowResult,
    LifetimeDecisionWorkflowStatus,
    ModelArtifactPolicy,
    run_lifetime_decision_workflow,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    Decision,
    LifePrediction,
    NormalizedConformalCalibration,
    NormalizedPredictionInterval,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.tools.audited_report import (
    AUDITED_REPORT_TOOL_VERSION,
    GenerateAuditedReportToolInput,
    ReportClaimKind,
    execute_generate_audited_report_tool,
)
from quanxin_life.tools.batch_decision import (
    BATCH_DECISION_TOOL_VERSION,
    BatchDecisionToolInput,
)
from quanxin_life.tools.conformal_calibration import (
    CONFORMAL_CALIBRATION_TOOL_VERSION,
    NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
    NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
    CalibratePredictionIntervalToolInput,
)
from quanxin_life.tools.cycle_life_prediction import (
    CYCLE_LIFE_PREDICTION_TOOL_VERSION,
    PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
    PredictCycleLifeToolInput,
)
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_TOOL_VERSION,
    ValidateBatteryDataToolInput,
    execute_validate_battery_data_tool,
    register_validate_battery_data_tool,
)
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
    ExtractEarlyCycleFeaturesToolInput,
    VerifiedEarlyCycleBatch,
)
from quanxin_life.tools.registry import StandardToolName, ToolDefinition, ToolRegistry

NOW = datetime(2026, 7, 15, tzinfo=UTC)
SHA = "a" * 64
POLICY_SHA = "b" * 64
DATASET_ID = "workflow-test"
CELL_ID = "cell-1"
DATA_VERSION = "workflow-test-data-v1"
FEATURE_VERSION = "workflow-test-feature-v1"
SPLIT_VERSION = "workflow-test-split-v1"
MODEL_VERSION = "workflow-test-model-v1"
RECORD_BATCH_ID = "record-batch-1"
CALIBRATION_COHORT_ID = "calibration-cohort-1"
POLICY_ID = "policy-1"


def _provenance(sha256: str = SHA) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            source_id=f"workflow-e2e-fixture-{sha256[:8]}",
            source_kind=SourceKind.OBSERVED,
            uri=f"test://workflow-e2e-fixture/{sha256[:8]}",
            sha256=sha256,
            description="Controlled non-production workflow fixture",
            created_at=NOW,
        )
    ]


def _record(*, dataset_id: str = DATASET_ID, cell_id: str = CELL_ID) -> CycleRecord:
    return CycleRecord(
        dataset_id=dataset_id,
        cell_id=cell_id,
        cycle_index=0,
        sample_index=0,
        time_s=0.0,
        voltage_v=3.2,
        current_a=0.0,
        temperature_c=25.0,
        discharge_capacity_ah=1.0,
        diagnostic=True,
    )


class _BatchResolver:
    def __init__(self, records: tuple[CycleRecord, ...]) -> None:
        self._batch = VerifiedEarlyCycleBatch(
            record_batch_id=RECORD_BATCH_ID,
            records=records,
            metadata={
                "dataset_id": DATASET_ID,
                "cell_id": CELL_ID,
                "chemistry": "LFP/graphite",
                "nominal_capacity_ah": 1.0,
                "source_uri": "test://workflow-e2e-fixture",
                "source_sha256": SHA,
                "schema_version": "test-v1",
            },
            feature_config={"cutoff_cycle": 20, "feature_version": FEATURE_VERSION},
            data_version=DATA_VERSION,
            split_version=SPLIT_VERSION,
            source_manifest_hash=SHA,
            provenance=tuple(_provenance()),
        )

    def resolve_verified_early_cycle_batch(self, record_batch_id: str) -> VerifiedEarlyCycleBatch:
        if record_batch_id != RECORD_BATCH_ID:
            raise KeyError(record_batch_id)
        return self._batch


def _batch_records(*, duplicated: bool = False) -> tuple[CycleRecord, ...]:
    first = _record().model_copy(update={"cycle_index": 0})
    last = _record().model_copy(update={"cycle_index": 20})
    return (first, last, last) if duplicated else (first, last)


def _run_test_workflow(
    service: ToolInvocationService,
    request: LifetimeDecisionWorkflowRequest,
    *,
    records: tuple[CycleRecord, ...] | None = None,
) -> LifetimeDecisionWorkflowResult:
    return run_lifetime_decision_workflow(
        service,
        request,
        batch_resolver=_BatchResolver(records or _batch_records()),
        model_artifact_policy=ModelArtifactPolicy.CONTROLLED_TEST_DOUBLE,
    )


def _result(
    tool_name: StandardToolName,
    tool_version: str,
    input_value: Any,
    *,
    values: dict[str, Any],
    uncertainty: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    model_version: str = MODEL_VERSION,
    provenance: list[ProvenanceRecord] | None = None,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version=tool_version,
        model_version=model_version,
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        input_hash=sha256_canonical(input_value.model_dump(mode="json")),
        values=values,
        uncertainty=uncertainty,
        warnings=warnings or [],
        provenance=provenance or _provenance(),
        created_at=NOW,
    )


def _life_prediction() -> LifePrediction:
    return LifePrediction(
        dataset_id=DATASET_ID,
        cell_id=CELL_ID,
        cutoff_cycle=20,
        predicted_eol_cycle=321.0,
        observed_eol_cycle=None,
        right_censored=True,
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        model_version=MODEL_VERSION,
        data_version=DATA_VERSION,
    )


def _calibration() -> NormalizedConformalCalibration:
    return NormalizedConformalCalibration(
        alpha=0.1,
        normalized_score_quantile=2.0,
        calibration_cell_count=1,
        scale_version="workflow-test-scale-v1",
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        model_version=MODEL_VERSION,
        data_version=DATA_VERSION,
    )


def _completed_service(
    *,
    malformed_feature_artifact: bool = False,
    unrelated_prediction_provenance: bool = False,
    feature_source_manifest_hash: str = SHA,
) -> tuple[ToolInvocationService, list[tuple[StandardToolName, Any]]]:
    registry = ToolRegistry()
    ledger = AuditLedger()
    calls: list[tuple[StandardToolName, Any]] = []

    def register(
        tool_name: StandardToolName,
        tool_version: str,
        input_model: type[Any],
        executor: Any,
    ) -> None:
        def tracked(input_value: Any) -> ToolResult:
            calls.append((tool_name, input_value))
            return cast(ToolResult, executor(input_value))

        registry.register(
            ToolDefinition(
                tool_name=tool_name,
                tool_version=tool_version,
                input_model=input_model,
                executor=tracked,
            )
        )

    register(
        StandardToolName.VALIDATE_BATTERY_DATA,
        DATA_QUALITY_TOOL_VERSION,
        ValidateBatteryDataToolInput,
        execute_validate_battery_data_tool,
    )

    def feature_executor(value: ExtractEarlyCycleFeaturesToolInput) -> ToolResult:
        artifact_type = (
            "malformed.feature.artifact"
            if malformed_feature_artifact
            else EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE
        )
        return _result(
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            EARLY_CYCLE_FEATURE_TOOL_VERSION,
            value,
            model_version=EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
            values={
                "artifact_type": artifact_type,
                "artifact": {
                    "record_batch_id": value.record_batch_id,
                    "dataset_id": DATASET_ID,
                    "cell_id": CELL_ID,
                    "cutoff_cycle": 20,
                    "observed_cycles": [0, 20],
                    "observed_soh": [1.0, 0.99],
                    "reference_capacity_ah": 1.0,
                    "reference_capacity_method": "test-fixture",
                    "feature_values": {"capacity_fade_slope": -0.01},
                    "condition_features": {"nominal_capacity_ah": 1.0},
                    "source_cycles": [0, 20],
                    "feature_warnings": [],
                    "source_manifest_hash": feature_source_manifest_hash,
                    "feature_version": FEATURE_VERSION,
                    "split_version": SPLIT_VERSION,
                    "data_version": DATA_VERSION,
                },
            },
        )

    register(
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        EARLY_CYCLE_FEATURE_TOOL_VERSION,
        ExtractEarlyCycleFeaturesToolInput,
        feature_executor,
    )

    def prediction_executor(value: PredictCycleLifeToolInput) -> ToolResult:
        prediction = _life_prediction()
        return _result(
            StandardToolName.PREDICT_CYCLE_LIFE,
            CYCLE_LIFE_PREDICTION_TOOL_VERSION,
            value,
            values={
                "artifact_type": PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
                "artifact": {
                    "record_batch_id": RECORD_BATCH_ID,
                    "dataset_id": DATASET_ID,
                    "cell_id": CELL_ID,
                    "cutoff_cycle": prediction.cutoff_cycle,
                    "life_prediction": prediction.model_dump(mode="json"),
                    "derived_rul_cycle": prediction.derived_rul,
                    "source_manifest_hash": SHA,
                    "upstream_result_id": value.upstream_result_id,
                    "split_version": SPLIT_VERSION,
                    "used_feature_names": ["capacity_fade_slope"],
                    "model_artifact_status": "CONTROLLED_TEST_DOUBLE",
                },
            },
            provenance=(
                _provenance("c" * 64)
                if unrelated_prediction_provenance
                else _provenance()
            ),
        )

    register(
        StandardToolName.PREDICT_CYCLE_LIFE,
        CYCLE_LIFE_PREDICTION_TOOL_VERSION,
        PredictCycleLifeToolInput,
        prediction_executor,
    )

    def conformal_executor(value: CalibratePredictionIntervalToolInput) -> ToolResult:
        calibration = _calibration()
        if value.calibration_cohort_id is not None:
            return _result(
                StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
                CONFORMAL_CALIBRATION_TOOL_VERSION,
                value,
                values={
                    "artifact_type": NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
                    "artifact": {
                        "calibration_cohort_id": value.calibration_cohort_id,
                        "calibration_domain_id": DATASET_ID,
                        "calibration_scope": "in_distribution",
                        "calibration": calibration.model_dump(mode="json"),
                        "calibration_cell_ids": ["calibration-cell-1"],
                        "split_manifest_hash": POLICY_SHA,
                        "source_manifest_hash": SHA,
                    },
                },
            )
        interval = NormalizedPredictionInterval(
            dataset_id=DATASET_ID,
            cell_id=CELL_ID,
            cutoff_cycle=20,
            point_prediction_cycle=321.0,
            lower_eol_cycle=301.0,
            upper_eol_cycle=341.0,
            difficulty_scale_cycle=10.0,
            calibration=calibration,
        )
        return _result(
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            CONFORMAL_CALIBRATION_TOOL_VERSION,
            value,
            values={
                "artifact_type": NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
                "artifact": {
                    "prediction_interval": interval.model_dump(mode="json"),
                    "prediction_result_id": value.prediction_result_id,
                    "calibration_result_id": value.calibration_result_id,
                    "calibration_domain_id": DATASET_ID,
                    "target_domain_id": DATASET_ID,
                    "target_domain_calibrated": True,
                    "difficulty_scale_source_manifest_hash": SHA,
                },
            },
            uncertainty={
                "coverage_target": interval.coverage_target,
                "difficulty_scale_cycle": interval.difficulty_scale_cycle,
                "lower_eol_cycle": interval.lower_eol_cycle,
                "upper_eol_cycle": interval.upper_eol_cycle,
            },
        )

    register(
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        CONFORMAL_CALIBRATION_TOOL_VERSION,
        CalibratePredictionIntervalToolInput,
        conformal_executor,
    )

    def decision_executor(value: BatchDecisionToolInput) -> ToolResult:
        return _result(
            StandardToolName.MAKE_BATCH_DECISION,
            BATCH_DECISION_TOOL_VERSION,
            value,
            values={
                "cell_id": CELL_ID,
                "dataset_id": DATASET_ID,
                "decision": Decision.RECHECK.value,
                "interval_lower_eol_cycle": 301.0,
                "interval_upper_eol_cycle": 341.0,
                "policy_id": value.policy_id,
                "policy_version": "workflow-test-policy-v1",
                "policy_source_manifest_hash": POLICY_SHA,
                "quality_report_blocked": False,
                "reason_codes": ["INTERVAL_CROSSES_THRESHOLD"],
                "required_eol_cycle": 320.0,
                "target_domain_calibrated": True,
                "upstream_result_ids": [
                    value.prediction_interval_result_id,
                    value.calibration_result_id,
                    value.quality_result_id,
                ],
            },
            provenance=[*_provenance(), *_provenance(POLICY_SHA)],
        )

    register(
        StandardToolName.MAKE_BATCH_DECISION,
        BATCH_DECISION_TOOL_VERSION,
        BatchDecisionToolInput,
        decision_executor,
    )

    def report_executor(input_value: GenerateAuditedReportToolInput) -> ToolResult:
        calls.append((StandardToolName.GENERATE_AUDITED_REPORT, input_value))
        return execute_generate_audited_report_tool(
            input_value,
            audit_ledger=ledger,
            clock=lambda: NOW,
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.GENERATE_AUDITED_REPORT,
            tool_version=AUDITED_REPORT_TOOL_VERSION,
            input_model=GenerateAuditedReportToolInput,
            executor=report_executor,
        )
    )
    return ToolInvocationService(registry=registry, audit_ledger=ledger), calls


def _request() -> LifetimeDecisionWorkflowRequest:
    return LifetimeDecisionWorkflowRequest(
        record_batch_id=RECORD_BATCH_ID,
        calibration_cohort_id=CALIBRATION_COHORT_ID,
        policy_id=POLICY_ID,
    )


def test_completed_workflow_registers_ordered_evidence_chain_and_report() -> None:
    service, calls = _completed_service()

    result = _run_test_workflow(service, _request())

    assert result.status is LifetimeDecisionWorkflowStatus.COMPLETED
    result_ids = (
        result.quality_result_id,
        result.feature_result_id,
        result.prediction_result_id,
        result.calibration_result_id,
        result.interval_result_id,
        result.decision_result_id,
        result.report_result_id,
    )
    assert all(result_id is not None for result_id in result_ids)
    assert all(UUID(result_id) for result_id in result_ids if result_id is not None)
    assert service.audit_ledger is not None
    resolved = tuple(
        service.audit_ledger.resolve_registered_result(result_id)
        for result_id in result_ids
        if result_id is not None
    )
    assert [item.tool_name for item in resolved] == [
        StandardToolName.VALIDATE_BATTERY_DATA.value,
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        StandardToolName.PREDICT_CYCLE_LIFE.value,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        StandardToolName.MAKE_BATCH_DECISION.value,
        StandardToolName.GENERATE_AUDITED_REPORT.value,
    ]
    assert [call[0] for call in calls] == [
        StandardToolName.VALIDATE_BATTERY_DATA,
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        StandardToolName.MAKE_BATCH_DECISION,
        StandardToolName.GENERATE_AUDITED_REPORT,
    ]
    assert calls[2][1].upstream_result_id == result.feature_result_id
    assert calls[4][1].prediction_result_id == result.prediction_result_id
    assert calls[4][1].calibration_result_id == result.calibration_result_id
    assert calls[5][1].prediction_interval_result_id == result.interval_result_id
    assert calls[5][1].calibration_result_id == result.calibration_result_id
    assert calls[5][1].quality_result_id == result.quality_result_id

    report_input = calls[6][1]
    assert [claim.claim_kind for claim in report_input.claims] == [
        ReportClaimKind.LIFETIME_PREDICTION,
        ReportClaimKind.DECISION_POLICY,
    ]
    assert set(report_input.upstream_result_ids) == {
        result.prediction_result_id,
        result.interval_result_id,
        result.decision_result_id,
    }
    evidence_paths = {
        (item.result_id, item.json_path)
        for claim in report_input.claims
        for item in claim.numeric_evidence
    }
    assert evidence_paths == {
        (result.prediction_result_id, PREDICTION_POINT_EOL_PATH),
        (result.interval_result_id, INTERVAL_LOWER_EOL_PATH),
        (result.interval_result_id, INTERVAL_UPPER_EOL_PATH),
        (result.decision_result_id, DECISION_REQUIRED_EOL_PATH),
    }
    assert "MODEL_INFERENCE" in resolved[-1].values["markdown"]
    assert "DOMAIN_KNOWLEDGE" in resolved[-1].values["markdown"]


def test_blocked_quality_returns_without_requiring_downstream_tools() -> None:
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    ledger = AuditLedger()
    service = ToolInvocationService(registry=registry, audit_ledger=ledger)
    result = _run_test_workflow(
        service,
        _request(),
        records=_batch_records(duplicated=True),
    )

    assert result.status is LifetimeDecisionWorkflowStatus.QUALITY_BLOCKED
    assert ledger.resolve_registered_result(result.quality_result_id).values["blocked"] is True
    assert result.feature_result_id is None
    assert result.prediction_result_id is None
    assert result.calibration_result_id is None
    assert result.interval_result_id is None
    assert result.decision_result_id is None
    assert result.report_result_id is None


def test_workflow_rejects_malformed_feature_artifact() -> None:
    service, _ = _completed_service(malformed_feature_artifact=True)

    with pytest.raises(ValueError, match="artifact_type"):
        _run_test_workflow(service, _request())


def test_workflow_result_enforces_status_dependent_result_ids() -> None:
    with pytest.raises(ValidationError, match="completed"):
        LifetimeDecisionWorkflowResult(
            status=LifetimeDecisionWorkflowStatus.COMPLETED,
            quality_result_id=str(uuid4()),
        )
    with pytest.raises(ValidationError, match="quality-blocked"):
        LifetimeDecisionWorkflowResult(
            status=LifetimeDecisionWorkflowStatus.QUALITY_BLOCKED,
            quality_result_id=str(uuid4()),
            feature_result_id=str(uuid4()),
        )


def test_workflow_requires_a_shared_audit_ledger() -> None:
    service, _ = _completed_service()
    service = ToolInvocationService(registry=service.registry)

    with pytest.raises(ValueError, match="audit ledger"):
        _run_test_workflow(service, _request())


def test_default_workflow_rejects_controlled_test_model_artifact() -> None:
    service, _ = _completed_service()

    with pytest.raises(ValueError, match="verified model artifact"):
        run_lifetime_decision_workflow(
            service,
            _request(),
            batch_resolver=_BatchResolver(_batch_records()),
        )


def test_workflow_rejects_prediction_with_unrelated_provenance() -> None:
    service, _ = _completed_service(unrelated_prediction_provenance=True)

    with pytest.raises(ValueError, match="provenance"):
        _run_test_workflow(service, _request())


def test_workflow_rejects_feature_resolver_with_another_source_manifest() -> None:
    service, calls = _completed_service(feature_source_manifest_hash="d" * 64)

    with pytest.raises(ValueError, match="source manifest"):
        _run_test_workflow(service, _request())
    assert [item[0] for item in calls] == [
        StandardToolName.VALIDATE_BATTERY_DATA,
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
    ]
