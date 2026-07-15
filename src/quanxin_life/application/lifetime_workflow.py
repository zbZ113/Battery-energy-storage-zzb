"""Deterministic, ledger-bound lifetime decision workflow.

The workflow only composes registered domain tools.  It never calculates,
repairs, rounds, or otherwise manufactures battery engineering values.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.core import Decision, ToolResult
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.audited_report import (
    AUDITED_REPORT_TOOL_VERSION,
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
)
from quanxin_life.tools.batch_decision import (
    BATCH_DECISION_TOOL_VERSION,
    BatchDecisionToolInput,
    DataQualityEvidence,
    DecisionIntervalEvidence,
)
from quanxin_life.tools.conformal_calibration import (
    CONFORMAL_CALIBRATION_TOOL_VERSION,
    NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
    NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
    CalibratePredictionIntervalToolInput,
    CalibrationOutputEvidence,
    PointPredictionEvidence,
)
from quanxin_life.tools.cycle_life_prediction import (
    CYCLE_LIFE_PREDICTION_TOOL_VERSION,
    PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
    EarlyCycleFeatureEvidence,
    PredictCycleLifeToolInput,
)
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
    ValidateBatteryDataToolInput,
)
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
    ExtractEarlyCycleFeaturesToolInput,
    VerifiedEarlyCycleBatch,
    VerifiedEarlyCycleBatchResolver,
)
from quanxin_life.tools.registry import StandardToolName

PREDICTION_POINT_EOL_PATH = "values.artifact.life_prediction.predicted_eol_cycle"
INTERVAL_LOWER_EOL_PATH = "values.artifact.prediction_interval.lower_eol_cycle"
INTERVAL_UPPER_EOL_PATH = "values.artifact.prediction_interval.upper_eol_cycle"
DECISION_REQUIRED_EOL_PATH = "values.required_eol_cycle"


class LifetimeDecisionWorkflowStatus(StrEnum):
    """Terminal states exposed by the deterministic workflow."""

    COMPLETED = "completed"
    QUALITY_BLOCKED = "quality_blocked"


class ModelArtifactPolicy(StrEnum):
    """Server-side artifact acceptance modes; never supplied by API clients."""

    VERIFIED_ONLY = "VERIFIED_ARTIFACT"
    CONTROLLED_TEST_DOUBLE = "CONTROLLED_TEST_DOUBLE"


class LifetimeDecisionWorkflowRequest(ContractModel):
    """Identifiers and source evidence needed to run one lifetime decision."""

    record_batch_id: str = Field(min_length=1)
    calibration_cohort_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)

    @field_validator("record_batch_id", "calibration_cohort_id", "policy_id")
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("workflow identifiers must not be blank")
        return normalized

class LifetimeDecisionWorkflowResult(ContractModel):
    """Evidence identifiers returned by one terminal workflow execution."""

    status: LifetimeDecisionWorkflowStatus
    quality_result_id: str
    feature_result_id: str | None = None
    prediction_result_id: str | None = None
    calibration_result_id: str | None = None
    interval_result_id: str | None = None
    decision_result_id: str | None = None
    report_result_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "quality_result_id",
        "feature_result_id",
        "prediction_result_id",
        "calibration_result_id",
        "interval_result_id",
        "decision_result_id",
        "report_result_id",
    )
    @classmethod
    def require_uuid_result_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("workflow result IDs must be UUID strings") from exc
        return value

    @model_validator(mode="after")
    def require_terminal_state_result_ids(self) -> LifetimeDecisionWorkflowResult:
        downstream_ids = (
            self.feature_result_id,
            self.prediction_result_id,
            self.calibration_result_id,
            self.interval_result_id,
            self.decision_result_id,
            self.report_result_id,
        )
        if self.status is LifetimeDecisionWorkflowStatus.COMPLETED and any(
            result_id is None for result_id in downstream_ids
        ):
            raise ValueError("completed workflow requires all downstream result IDs")
        if self.status is LifetimeDecisionWorkflowStatus.QUALITY_BLOCKED and any(
            result_id is not None for result_id in downstream_ids
        ):
            raise ValueError("quality-blocked workflow must not contain downstream result IDs")
        return self


class _DecisionOutputEvidence(ContractModel):
    cell_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    decision: Decision
    interval_lower_eol_cycle: float = Field(ge=0, allow_inf_nan=False)
    interval_upper_eol_cycle: float = Field(ge=0, allow_inf_nan=False)
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    policy_source_manifest_hash: Sha256
    quality_report_blocked: bool
    reason_codes: tuple[str, ...] = Field(min_length=1)
    required_eol_cycle: float = Field(gt=0, allow_inf_nan=False)
    target_domain_calibrated: bool
    upstream_result_ids: tuple[str, str, str]

    @field_validator("upstream_result_ids")
    @classmethod
    def require_uuid_upstream_ids(
        cls, value: tuple[str, str, str]
    ) -> tuple[str, str, str]:
        try:
            for result_id in value:
                UUID(result_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("decision upstream result IDs must be UUID strings") from exc
        if len(set(value)) != len(value):
            raise ValueError("decision upstream result IDs must be distinct")
        return value


def _invoke(
    service: ToolInvocationService,
    tool_name: StandardToolName,
    input_value: ContractModel,
) -> ToolResult:
    return service.invoke(
        ToolInvocation(
            tool_name=tool_name,
            input_value=input_value.model_dump(mode="json"),
        )
    )


def _require_artifact(
    result: ToolResult,
    *,
    expected_tool_name: StandardToolName,
    expected_tool_version: str,
    expected_artifact_type: str,
    description: str,
) -> Mapping[str, object]:
    if result.tool_name != expected_tool_name.value:
        raise ValueError(f"{description} result has an unexpected tool_name")
    if result.tool_version != expected_tool_version:
        raise ValueError(f"{description} result has an unsupported tool_version")
    if result.values.get("artifact_type") != expected_artifact_type:
        raise ValueError(f"{description} result has an unexpected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{description} result must contain values.artifact")
    return artifact


def _provenance_hashes(result: ToolResult) -> set[str]:
    return {record.sha256 for record in result.provenance}


def _require_provenance_hashes(
    result: ToolResult,
    required_hashes: set[str],
    *,
    description: str,
) -> None:
    if not required_hashes.issubset(_provenance_hashes(result)):
        raise ValueError(f"{description} provenance does not cover its declared sources")


def _validate_quality_result(
    result: ToolResult,
    *,
    expected_dataset_id: str,
    expected_data_version: str,
    expected_feature_version: str,
    expected_source_manifest_hash: str,
) -> DataQualityEvidence:
    if result.tool_name != StandardToolName.VALIDATE_BATTERY_DATA.value:
        raise ValueError("quality result has an unexpected tool_name")
    if result.tool_version != DATA_QUALITY_TOOL_VERSION:
        raise ValueError("quality result has an unsupported tool_version")
    if result.model_version != DATA_QUALITY_MODEL_VERSION:
        raise ValueError("quality result has an unsupported model_version")
    try:
        evidence = DataQualityEvidence.model_validate(result.values)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("quality result values do not satisfy their contract") from exc
    if evidence.dataset_id != expected_dataset_id:
        raise ValueError("quality result dataset_id must match validation records")
    if result.data_version != expected_data_version:
        raise ValueError("quality result data_version must match the trusted batch")
    if result.feature_version != expected_feature_version:
        raise ValueError("quality result feature_version must match the trusted batch")
    _require_provenance_hashes(
        result,
        {expected_source_manifest_hash},
        description="quality result",
    )
    return evidence


def _validate_feature_result(
    result: ToolResult,
    *,
    request: LifetimeDecisionWorkflowRequest,
    expected_dataset_id: str,
    expected_cell_id: str,
    expected_source_manifest_hash: str,
    quality_result: ToolResult,
) -> EarlyCycleFeatureEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        expected_tool_version=EARLY_CYCLE_FEATURE_TOOL_VERSION,
        expected_artifact_type=EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
        description="early-cycle feature",
    )
    if result.model_version != EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION:
        raise ValueError("early-cycle feature result has an unsupported model_version")
    try:
        evidence = EarlyCycleFeatureEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("early-cycle feature artifact does not satisfy its contract") from exc
    if evidence.record_batch_id != request.record_batch_id:
        raise ValueError("early-cycle feature record_batch_id must match workflow request")
    if (evidence.dataset_id, evidence.cell_id) != (expected_dataset_id, expected_cell_id):
        raise ValueError("early-cycle feature identity must match validation records")
    if evidence.source_manifest_hash != expected_source_manifest_hash:
        raise ValueError("early-cycle feature source manifest must match the trusted batch")
    if (
        result.data_version != evidence.data_version
        or result.feature_version != evidence.feature_version
    ):
        raise ValueError("early-cycle feature result versions must match its artifact")
    quality_hashes = {record.sha256 for record in quality_result.provenance}
    feature_hashes = {record.sha256 for record in result.provenance}
    if not quality_hashes.intersection(feature_hashes):
        raise ValueError("quality and early-cycle feature provenance must share a source hash")
    _require_provenance_hashes(
        result,
        {expected_source_manifest_hash},
        description="early-cycle feature",
    )
    return evidence


def _validate_prediction_result(
    result: ToolResult,
    *,
    feature_result_id: str,
    feature: EarlyCycleFeatureEvidence,
    feature_result: ToolResult,
    model_artifact_policy: ModelArtifactPolicy,
) -> PointPredictionEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        expected_tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
        expected_artifact_type=PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
        description="cycle-life prediction",
    )
    try:
        evidence = PointPredictionEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("cycle-life prediction artifact does not satisfy its contract") from exc
    prediction = evidence.life_prediction
    if evidence.upstream_result_id != feature_result_id:
        raise ValueError("cycle-life prediction must reference the feature result")
    if evidence.record_batch_id != feature.record_batch_id:
        raise ValueError("cycle-life prediction record_batch_id must match feature evidence")
    if (evidence.dataset_id, evidence.cell_id) != (feature.dataset_id, feature.cell_id):
        raise ValueError("cycle-life prediction identity must match feature evidence")
    if evidence.source_manifest_hash != feature.source_manifest_hash:
        raise ValueError("cycle-life prediction source manifest must match feature evidence")
    if evidence.split_version != feature.split_version:
        raise ValueError("cycle-life prediction split_version must match feature evidence")
    if evidence.model_artifact_status != model_artifact_policy.value:
        raise ValueError("cycle-life prediction requires a verified model artifact")
    _require_provenance_hashes(
        result,
        {*_provenance_hashes(feature_result), evidence.source_manifest_hash},
        description="cycle-life prediction",
    )
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(prediction, field_name):
            raise ValueError(f"cycle-life prediction {field_name} must match its artifact")
    return evidence


def _validate_calibration_result(
    result: ToolResult,
    *,
    request: LifetimeDecisionWorkflowRequest,
    prediction: PointPredictionEvidence,
) -> CalibrationOutputEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        expected_tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        expected_artifact_type=NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
        description="conformal calibration",
    )
    try:
        evidence = CalibrationOutputEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("conformal calibration artifact does not satisfy its contract") from exc
    if evidence.calibration_cohort_id != request.calibration_cohort_id:
        raise ValueError("calibration cohort must match workflow request")
    if evidence.calibration_domain_id != prediction.dataset_id:
        raise ValueError("calibration domain must match prediction dataset")
    calibration = evidence.calibration
    life_prediction = prediction.life_prediction
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(calibration, field_name):
            raise ValueError(f"conformal calibration {field_name} must match its artifact")
        if getattr(calibration, field_name) != getattr(life_prediction, field_name):
            raise ValueError(f"conformal calibration {field_name} must match prediction")
    if calibration.split_version != life_prediction.split_version:
        raise ValueError("conformal calibration split_version must match prediction")
    _require_provenance_hashes(
        result,
        {evidence.source_manifest_hash},
        description="conformal calibration",
    )
    return evidence


def _validate_interval_result(
    result: ToolResult,
    *,
    prediction_result_id: str,
    calibration_result_id: str,
    prediction: PointPredictionEvidence,
    calibration: CalibrationOutputEvidence,
    prediction_result: ToolResult,
    calibration_result: ToolResult,
) -> DecisionIntervalEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        expected_tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        expected_artifact_type=NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
        description="prediction interval",
    )
    try:
        evidence = DecisionIntervalEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("prediction interval artifact does not satisfy its contract") from exc
    if evidence.prediction_result_id != prediction_result_id:
        raise ValueError("prediction interval must reference the point prediction")
    if evidence.calibration_result_id != calibration_result_id:
        raise ValueError("prediction interval must reference the calibration result")
    interval = evidence.prediction_interval
    life_prediction = prediction.life_prediction
    if (interval.dataset_id, interval.cell_id, interval.cutoff_cycle) != (
        life_prediction.dataset_id,
        life_prediction.cell_id,
        life_prediction.cutoff_cycle,
    ):
        raise ValueError("prediction interval identity must match point prediction")
    if interval.calibration != calibration.calibration:
        raise ValueError("prediction interval calibration must match calibration evidence")
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(interval.calibration, field_name):
            raise ValueError(f"prediction interval {field_name} must match its artifact")
    _require_provenance_hashes(
        result,
        {
            *_provenance_hashes(prediction_result),
            *_provenance_hashes(calibration_result),
            evidence.difficulty_scale_source_manifest_hash,
        },
        description="prediction interval",
    )
    return evidence


def _validate_decision_result(
    result: ToolResult,
    *,
    request: LifetimeDecisionWorkflowRequest,
    quality_result_id: str,
    calibration_result_id: str,
    interval_result_id: str,
    interval: DecisionIntervalEvidence,
    interval_result: ToolResult,
    quality_result: ToolResult,
) -> _DecisionOutputEvidence:
    if result.tool_name != StandardToolName.MAKE_BATCH_DECISION.value:
        raise ValueError("batch decision result has an unexpected tool_name")
    if result.tool_version != BATCH_DECISION_TOOL_VERSION:
        raise ValueError("batch decision result has an unsupported tool_version")
    try:
        evidence = _DecisionOutputEvidence.model_validate(result.values)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("batch decision result does not satisfy its contract") from exc
    expected_ids = (interval_result_id, calibration_result_id, quality_result_id)
    if evidence.upstream_result_ids != expected_ids:
        raise ValueError("batch decision upstream result IDs do not match workflow evidence")
    if evidence.policy_id != request.policy_id:
        raise ValueError("batch decision policy_id must match workflow request")
    prediction_interval = interval.prediction_interval
    if (evidence.dataset_id, evidence.cell_id) != (
        prediction_interval.dataset_id,
        prediction_interval.cell_id,
    ):
        raise ValueError("batch decision identity must match prediction interval")
    if (
        evidence.interval_lower_eol_cycle != prediction_interval.lower_eol_cycle
        or evidence.interval_upper_eol_cycle != prediction_interval.upper_eol_cycle
    ):
        raise ValueError("batch decision bounds must match prediction interval")
    if evidence.quality_report_blocked:
        raise ValueError("completed batch decision cannot use blocked quality evidence")
    _require_provenance_hashes(
        result,
        {
            *_provenance_hashes(interval_result),
            *_provenance_hashes(quality_result),
            evidence.policy_source_manifest_hash,
        },
        description="batch decision",
    )
    return evidence


def _validate_report_result(
    result: ToolResult,
    upstream_results: tuple[ToolResult, ...],
) -> None:
    if result.tool_name != StandardToolName.GENERATE_AUDITED_REPORT.value:
        raise ValueError("audited report result has an unexpected tool_name")
    if result.tool_version != AUDITED_REPORT_TOOL_VERSION:
        raise ValueError("audited report result has an unsupported tool_version")
    reported_ids = result.values.get("upstream_result_ids")
    upstream_result_ids = tuple(item.result_id for item in upstream_results)
    if reported_ids != list(upstream_result_ids):
        raise ValueError("audited report upstream IDs must match report references")
    if not isinstance(result.values.get("markdown"), str) or not result.values["markdown"]:
        raise ValueError("audited report must contain non-empty markdown")
    _require_provenance_hashes(
        result,
        {sha for upstream in upstream_results for sha in _provenance_hashes(upstream)},
        description="audited report",
    )


def _merge_warnings(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group))


def run_lifetime_decision_workflow(
    service: ToolInvocationService,
    request: LifetimeDecisionWorkflowRequest,
    *,
    batch_resolver: VerifiedEarlyCycleBatchResolver | None = None,
    model_artifact_policy: ModelArtifactPolicy = ModelArtifactPolicy.VERIFIED_ONLY,
) -> LifetimeDecisionWorkflowResult:
    """Run the fixed quality-to-report chain through the shared tool service."""

    if service.audit_ledger is None:
        raise ValueError("lifetime decision workflow requires a shared audit ledger")
    if batch_resolver is None:
        raise ValueError("lifetime decision workflow requires a trusted batch resolver")
    validated_request = LifetimeDecisionWorkflowRequest.model_validate(
        request.model_dump(mode="json")
    )
    resolved_batch = batch_resolver.resolve_verified_early_cycle_batch(
        validated_request.record_batch_id
    )
    batch = VerifiedEarlyCycleBatch.model_validate(resolved_batch.model_dump(mode="json"))
    if batch.record_batch_id != validated_request.record_batch_id:
        raise ValueError("trusted batch resolver returned a mismatched record_batch_id")

    quality = _invoke(
        service,
        StandardToolName.VALIDATE_BATTERY_DATA,
        ValidateBatteryDataToolInput(
            records=batch.records,
            data_version=batch.data_version,
            feature_version=batch.feature_config.feature_version,
            provenance=batch.provenance,
            validated_at=datetime.now(UTC),
        ),
    )
    expected_identity = (batch.metadata.dataset_id, batch.metadata.cell_id)
    quality_evidence = _validate_quality_result(
        quality,
        expected_dataset_id=expected_identity[0],
        expected_data_version=batch.data_version,
        expected_feature_version=batch.feature_config.feature_version,
        expected_source_manifest_hash=batch.source_manifest_hash,
    )
    if quality_evidence.blocked:
        return LifetimeDecisionWorkflowResult(
            status=LifetimeDecisionWorkflowStatus.QUALITY_BLOCKED,
            quality_result_id=quality.result_id,
            warnings=list(quality.warnings),
        )

    features = _invoke(
        service,
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        ExtractEarlyCycleFeaturesToolInput(
            record_batch_id=validated_request.record_batch_id,
        ),
    )
    feature_evidence = _validate_feature_result(
        features,
        request=validated_request,
        expected_dataset_id=expected_identity[0],
        expected_cell_id=expected_identity[1],
        expected_source_manifest_hash=batch.source_manifest_hash,
        quality_result=quality,
    )
    if feature_evidence.cutoff_cycle != batch.feature_config.cutoff_cycle:
        raise ValueError("early-cycle feature cutoff must match the trusted batch")
    prediction = _invoke(
        service,
        StandardToolName.PREDICT_CYCLE_LIFE,
        PredictCycleLifeToolInput(upstream_result_id=features.result_id),
    )
    prediction_evidence = _validate_prediction_result(
        prediction,
        feature_result_id=features.result_id,
        feature=feature_evidence,
        feature_result=features,
        model_artifact_policy=model_artifact_policy,
    )
    calibration = _invoke(
        service,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        CalibratePredictionIntervalToolInput(
            calibration_cohort_id=validated_request.calibration_cohort_id,
        ),
    )
    calibration_evidence = _validate_calibration_result(
        calibration,
        request=validated_request,
        prediction=prediction_evidence,
    )
    interval = _invoke(
        service,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        CalibratePredictionIntervalToolInput(
            prediction_result_id=prediction.result_id,
            calibration_result_id=calibration.result_id,
        ),
    )
    interval_evidence = _validate_interval_result(
        interval,
        prediction_result_id=prediction.result_id,
        calibration_result_id=calibration.result_id,
        prediction=prediction_evidence,
        calibration=calibration_evidence,
        prediction_result=prediction,
        calibration_result=calibration,
    )
    decision = _invoke(
        service,
        StandardToolName.MAKE_BATCH_DECISION,
        BatchDecisionToolInput(
            prediction_interval_result_id=interval.result_id,
            calibration_result_id=calibration.result_id,
            quality_result_id=quality.result_id,
            policy_id=validated_request.policy_id,
        ),
    )
    _validate_decision_result(
        decision,
        request=validated_request,
        quality_result_id=quality.result_id,
        calibration_result_id=calibration.result_id,
        interval_result_id=interval.result_id,
        interval=interval_evidence,
        interval_result=interval,
        quality_result=quality,
    )
    report_upstream_ids = (
        prediction.result_id,
        interval.result_id,
        decision.result_id,
    )
    report = _invoke(
        service,
        StandardToolName.GENERATE_AUDITED_REPORT,
        GenerateAuditedReportToolInput(
            report_kind=ReportKind.LIFETIME_DECISION,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                    numeric_evidence=(
                        NumericEvidenceReference(
                            result_id=prediction.result_id,
                            json_path=PREDICTION_POINT_EOL_PATH,
                        ),
                        NumericEvidenceReference(
                            result_id=interval.result_id,
                            json_path=INTERVAL_LOWER_EOL_PATH,
                        ),
                        NumericEvidenceReference(
                            result_id=interval.result_id,
                            json_path=INTERVAL_UPPER_EOL_PATH,
                        ),
                    ),
                ),
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.DECISION_POLICY,
                    numeric_evidence=(
                        NumericEvidenceReference(
                            result_id=decision.result_id,
                            json_path=DECISION_REQUIRED_EOL_PATH,
                        ),
                    ),
                ),
            ),
            upstream_result_ids=report_upstream_ids,
        ),
    )
    _validate_report_result(report, (prediction, interval, decision))

    return LifetimeDecisionWorkflowResult(
        status=LifetimeDecisionWorkflowStatus.COMPLETED,
        quality_result_id=quality.result_id,
        feature_result_id=features.result_id,
        prediction_result_id=prediction.result_id,
        calibration_result_id=calibration.result_id,
        interval_result_id=interval.result_id,
        decision_result_id=decision.result_id,
        report_result_id=report.result_id,
        warnings=_merge_warnings(
            quality.warnings,
            features.warnings,
            prediction.warnings,
            calibration.warnings,
            interval.warnings,
            decision.warnings,
            report.warnings,
        ),
    )
