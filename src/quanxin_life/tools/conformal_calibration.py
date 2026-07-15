"""Trusted normalized-Conformal calibration and target-interval evidence.

The public tool creates calibration evidence from a trusted cell-disjoint
cohort, or issues a target interval from registered point-prediction and
calibration evidence plus a trusted model difficulty scale.  No caller or LLM
can inject labels, scales, alpha values, residuals or interval bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    LifePrediction,
    NormalizedConformalCalibration,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools.cycle_life_prediction import (
    CYCLE_LIFE_PREDICTION_TOOL_VERSION,
    PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)
from quanxin_life.uncertainty import (
    ScaledLifePrediction,
    calibrate_normalized_conformal,
    make_normalized_prediction_interval,
)

CONFORMAL_CALIBRATION_TOOL_VERSION = "conformal-calibration-tool-v1"
NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE = (
    "quanxin_life.normalized_conformal_calibration.v1"
)
NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE = (
    "quanxin_life.normalized_prediction_interval.v1"
)
COVERAGE_SCOPE_WARNING = "COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT"
TARGET_DOMAIN_RECALIBRATION_WARNING = "TARGET_DOMAIN_RECALIBRATION_APPLIED"
TARGET_DOMAIN_UNCALIBRATED_WARNING = "TARGET_DOMAIN_UNCALIBRATED"
Clock = Callable[[], datetime]
CalibrationScope = Literal["in_distribution", "target_domain_recalibration"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _VerifiedCalibrationModel(ContractModel):
    """Strict transport contract returned only by a trusted calibration store."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class VerifiedNormalizedCalibrationCohort(_VerifiedCalibrationModel):
    """Cell-disjoint observed calibration data with a server-owned policy.

    ``calibration_predictions`` may contain labels only because this class is
    created by the trusted evaluation-data adapter.  It is never a public tool
    input and its individual labels are not copied into the output artifact.
    """

    calibration_cohort_id: str = Field(min_length=1)
    calibration_predictions: tuple[ScaledLifePrediction, ...] = Field(min_length=1)
    split_manifest: SplitManifest
    calibration_domain_id: str = Field(min_length=1)
    calibration_scope: CalibrationScope = "in_distribution"
    source_manifest_hash: Sha256
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("calibration_cohort_id", "calibration_domain_id")
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("calibration identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_observed_source_and_homogeneous_prediction_context(
        self,
    ) -> VerifiedNormalizedCalibrationCohort:
        if not any(item.source_kind is SourceKind.OBSERVED for item in self.provenance):
            raise ValueError("calibration cohort provenance must include an OBSERVED source")

        reference = self.calibration_predictions[0].prediction
        if reference.dataset_id != self.split_manifest.dataset_id:
            raise ValueError("calibration prediction dataset_id must match the split manifest")
        for scaled_prediction in self.calibration_predictions:
            prediction = scaled_prediction.prediction
            if prediction.dataset_id != self.split_manifest.dataset_id:
                raise ValueError("calibration prediction dataset_id must match the split manifest")
            if prediction.right_censored or prediction.observed_eol_cycle is None:
                raise ValueError(
                    "calibration predictions require observed non-censored EOL80 labels"
                )
            for field_name in (
                "cutoff_cycle",
                "target",
                "feature_version",
                "split_version",
                "model_version",
                "data_version",
            ):
                if getattr(prediction, field_name) != getattr(reference, field_name):
                    raise ValueError(f"calibration predictions must share {field_name}")
            if scaled_prediction.scale_version != self.calibration_predictions[0].scale_version:
                raise ValueError("calibration predictions must share scale_version")
        return self


class VerifiedNormalizedCalibrationCohortResolver(Protocol):
    """Server-side lookup for a source-verified calibration cohort."""

    def resolve_verified_normalized_calibration_cohort(
        self, calibration_cohort_id: str
    ) -> VerifiedNormalizedCalibrationCohort: ...


class VerifiedPredictionDifficultyScale(_VerifiedCalibrationModel):
    """Trusted model-produced scale for one registered point prediction."""

    prediction_result_id: str
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    difficulty_scale_cycle: float = Field(gt=0, allow_inf_nan=False)
    scale_version: str = Field(min_length=1)
    target_domain_id: str = Field(min_length=1)
    source_manifest_hash: Sha256
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("prediction_result_id")
    @classmethod
    def require_prediction_result_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("prediction_result_id must be a UUID string") from exc
        return value

    @model_validator(mode="after")
    def require_model_produced_scale_provenance(self) -> VerifiedPredictionDifficultyScale:
        if not any(item.source_kind is SourceKind.PREDICTED for item in self.provenance):
            raise ValueError("difficulty scale provenance must include a PREDICTED source")
        return self


class VerifiedPredictionDifficultyScaleResolver(Protocol):
    """Server-side lookup for a model-produced, identity-bound difficulty scale."""

    def resolve_verified_prediction_difficulty_scale(
        self, prediction_result_id: str
    ) -> VerifiedPredictionDifficultyScale: ...


class CalibratePredictionIntervalToolInput(ContractModel):
    """One trusted calibration cohort or one registered point/calibration pair."""

    calibration_cohort_id: str | None = Field(default=None, min_length=1)
    prediction_result_id: str | None = None
    calibration_result_id: str | None = None

    @field_validator("calibration_cohort_id")
    @classmethod
    def require_nonblank_cohort_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("calibration_cohort_id must not be blank")
        return normalized

    @field_validator("prediction_result_id", "calibration_result_id")
    @classmethod
    def require_result_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result IDs must be UUID strings") from exc
        return value

    @model_validator(mode="after")
    def require_exactly_one_safe_operation_mode(self) -> CalibratePredictionIntervalToolInput:
        calibration_mode = self.calibration_cohort_id is not None
        interval_mode = (
            self.prediction_result_id is not None and self.calibration_result_id is not None
        )
        no_interval_identifiers = (
            self.prediction_result_id is None and self.calibration_result_id is None
        )
        if calibration_mode and no_interval_identifiers:
            return self
        if not calibration_mode and interval_mode:
            return self
        raise ValueError(
            "provide calibration_cohort_id or both prediction_result_id and "
            "calibration_result_id"
        )


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_cohort(
    resolver: VerifiedNormalizedCalibrationCohortResolver,
    calibration_cohort_id: str,
) -> VerifiedNormalizedCalibrationCohort:
    resolved = resolver.resolve_verified_normalized_calibration_cohort(calibration_cohort_id)
    try:
        return VerifiedNormalizedCalibrationCohort.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        message = str(first_error["msg"])
        raise ValueError(f"trusted calibration cohort violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted calibration cohort does not satisfy its public contract") from exc


def _calibration_artifact(
    cohort: VerifiedNormalizedCalibrationCohort,
    calibration: NormalizedConformalCalibration,
) -> dict[str, object]:
    return {
        "calibration_cohort_id": cohort.calibration_cohort_id,
        "calibration_domain_id": cohort.calibration_domain_id,
        "calibration_scope": cohort.calibration_scope,
        "calibration": calibration.model_dump(mode="json"),
        "calibration_cell_ids": [
            scaled_prediction.prediction.cell_id
            for scaled_prediction in cohort.calibration_predictions
        ],
        "split_manifest_hash": sha256_canonical(cohort.split_manifest.model_dump(mode="json")),
        "source_manifest_hash": cohort.source_manifest_hash,
    }


class _IntervalEvidenceModel(ContractModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class CalibrationOutputEvidence(_IntervalEvidenceModel):
    calibration_cohort_id: str = Field(min_length=1)
    calibration_domain_id: str = Field(min_length=1)
    calibration_scope: CalibrationScope
    calibration: NormalizedConformalCalibration
    calibration_cell_ids: tuple[str, ...] = Field(min_length=1)
    split_manifest_hash: Sha256
    source_manifest_hash: Sha256


class PointPredictionEvidence(_IntervalEvidenceModel):
    record_batch_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    life_prediction: LifePrediction
    derived_rul_cycle: float = Field(ge=0, allow_inf_nan=False)
    source_manifest_hash: Sha256
    upstream_result_id: str
    split_version: str = Field(min_length=1)
    used_feature_names: tuple[str, ...] = Field(min_length=1)
    model_artifact_status: str = Field(min_length=1)

    @field_validator("upstream_result_id")
    @classmethod
    def require_upstream_result_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_id must be a UUID string") from exc
        return value


def _require_result_artifact(
    result: ToolResult,
    *,
    expected_tool_name: StandardToolName,
    expected_tool_version: str,
    expected_artifact_type: str,
    description: str,
) -> Mapping[str, object]:
    if result.tool_name != expected_tool_name.value:
        raise ValueError(f"{description} result must come from {expected_tool_name.value}")
    if result.tool_version != expected_tool_version:
        raise ValueError(f"{description} result has an unsupported tool_version")
    if result.values.get("artifact_type") != expected_artifact_type:
        raise ValueError(f"{description} result must contain the expected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{description} result must contain values.artifact")
    return artifact


def _decode_calibration_output(result: ToolResult) -> CalibrationOutputEvidence:
    artifact = _require_result_artifact(
        result,
        expected_tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        expected_tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        expected_artifact_type=NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
        description="calibration",
    )
    try:
        evidence = CalibrationOutputEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("calibration artifact does not satisfy its contract") from exc
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(evidence.calibration, field_name):
            raise ValueError(f"calibration result {field_name} must match its artifact")
    return evidence


def _decode_point_prediction(result: ToolResult) -> PointPredictionEvidence:
    artifact = _require_result_artifact(
        result,
        expected_tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        expected_tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
        expected_artifact_type=PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
        description="point prediction",
    )
    try:
        evidence = PointPredictionEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("point prediction artifact does not satisfy its contract") from exc
    prediction = evidence.life_prediction
    expected_values = {
        "dataset_id": evidence.dataset_id,
        "cell_id": evidence.cell_id,
        "cutoff_cycle": evidence.cutoff_cycle,
        "split_version": evidence.split_version,
    }
    for field_name, expected in expected_values.items():
        if getattr(prediction, field_name) != expected:
            raise ValueError(f"point prediction {field_name} must match its artifact")
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(prediction, field_name):
            raise ValueError(f"point prediction result {field_name} must match its artifact")
    if not prediction.right_censored or prediction.observed_eol_cycle is not None:
        raise ValueError("point prediction must not expose an observed EOL80 label")
    return evidence


def _resolve_verified_difficulty_scale(
    resolver: VerifiedPredictionDifficultyScaleResolver,
    prediction_result_id: str,
) -> VerifiedPredictionDifficultyScale:
    resolved = resolver.resolve_verified_prediction_difficulty_scale(prediction_result_id)
    try:
        return VerifiedPredictionDifficultyScale.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(f"trusted difficulty scale violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted difficulty scale does not satisfy its public contract") from exc


def _validate_interval_context(
    *,
    prediction_result_id: str,
    prediction_evidence: PointPredictionEvidence,
    calibration: CalibrationOutputEvidence,
    scale: VerifiedPredictionDifficultyScale,
) -> None:
    prediction = prediction_evidence.life_prediction
    if scale.prediction_result_id != prediction_result_id:
        raise ValueError(
            "difficulty scale prediction_result_id must match the registered prediction"
        )
    if prediction.cell_id in calibration.calibration_cell_ids:
        raise ValueError("target prediction cell_id must not be a calibration cell")
    for field_name in (
        "dataset_id",
        "cell_id",
        "cutoff_cycle",
        "feature_version",
        "split_version",
        "model_version",
        "data_version",
    ):
        if getattr(scale, field_name) != getattr(prediction, field_name):
            raise ValueError(f"difficulty scale {field_name} must match point prediction")
    for field_name in ("feature_version", "split_version", "model_version", "data_version"):
        if getattr(prediction, field_name) != getattr(calibration.calibration, field_name):
            raise ValueError(f"point prediction {field_name} must match conformal calibration")
    if scale.scale_version != calibration.calibration.scale_version:
        raise ValueError("difficulty scale scale_version must match conformal calibration")
    if scale.source_manifest_hash != prediction_evidence.source_manifest_hash:
        raise ValueError(
            "difficulty scale source_manifest_hash must match point prediction evidence"
        )


def _merge_provenance(*chains: Sequence[ProvenanceRecord]) -> list[ProvenanceRecord]:
    merged: list[ProvenanceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in chains:
        for record in chain:
            key = (record.source_id, record.uri, record.sha256)
            if key not in seen:
                seen.add(key)
                merged.append(record)
    return merged


def _issue_normalized_interval(
    input_value: CalibratePredictionIntervalToolInput,
    *,
    audit_ledger: AuditLedger,
    difficulty_scale_resolver: VerifiedPredictionDifficultyScaleResolver,
    clock: Clock,
) -> ToolResult:
    if input_value.prediction_result_id is None or input_value.calibration_result_id is None:
        raise ValueError("interval issuance requires prediction and calibration result IDs")
    prediction_result = audit_ledger.resolve_registered_result(input_value.prediction_result_id)
    calibration_result = audit_ledger.resolve_registered_result(input_value.calibration_result_id)
    prediction_evidence = _decode_point_prediction(prediction_result)
    calibration_evidence = _decode_calibration_output(calibration_result)
    scale = _resolve_verified_difficulty_scale(
        difficulty_scale_resolver,
        input_value.prediction_result_id,
    )
    _validate_interval_context(
        prediction_result_id=input_value.prediction_result_id,
        prediction_evidence=prediction_evidence,
        calibration=calibration_evidence,
        scale=scale,
    )
    interval = make_normalized_prediction_interval(
        ScaledLifePrediction(
            prediction=prediction_evidence.life_prediction,
            difficulty_scale_cycle=scale.difficulty_scale_cycle,
            scale_version=scale.scale_version,
        ),
        calibration_evidence.calibration,
    )
    target_domain_calibrated = scale.target_domain_id == calibration_evidence.calibration_domain_id
    warnings = [COVERAGE_SCOPE_WARNING]
    if target_domain_calibrated:
        if calibration_evidence.calibration_scope == "target_domain_recalibration":
            warnings.append(TARGET_DOMAIN_RECALIBRATION_WARNING)
    else:
        warnings.append(TARGET_DOMAIN_UNCALIBRATED_WARNING)
    artifact = {
        "prediction_interval": interval.model_dump(mode="json"),
        "prediction_result_id": input_value.prediction_result_id,
        "calibration_result_id": input_value.calibration_result_id,
        "calibration_domain_id": calibration_evidence.calibration_domain_id,
        "target_domain_id": scale.target_domain_id,
        "target_domain_calibrated": target_domain_calibrated,
        "difficulty_scale_source_manifest_hash": scale.source_manifest_hash,
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        model_version=interval.calibration.model_version,
        data_version=interval.calibration.data_version,
        feature_version=interval.calibration.feature_version,
        input_hash=sha256_canonical(input_value.model_dump(mode="json")),
        values={
            "artifact_type": NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
            "artifact": artifact,
        },
        uncertainty={
            "coverage_target": interval.coverage_target,
            "difficulty_scale_cycle": interval.difficulty_scale_cycle,
            "lower_eol_cycle": interval.lower_eol_cycle,
            "upper_eol_cycle": interval.upper_eol_cycle,
        },
        warnings=warnings,
        provenance=_merge_provenance(
            prediction_result.provenance,
            calibration_result.provenance,
            scale.provenance,
        ),
        created_at=_execution_timestamp(clock),
    )


def execute_calibrate_prediction_interval_tool(
    input_value: CalibratePredictionIntervalToolInput,
    *,
    resolver: VerifiedNormalizedCalibrationCohortResolver,
    audit_ledger: AuditLedger | None = None,
    difficulty_scale_resolver: VerifiedPredictionDifficultyScaleResolver | None = None,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Create calibration evidence or issue one interval from trusted dependencies."""

    validated_input = CalibratePredictionIntervalToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    if validated_input.calibration_cohort_id is None:
        if audit_ledger is None or difficulty_scale_resolver is None:
            raise ValueError(
                "interval issuance requires an audit ledger and difficulty scale resolver"
            )
        return _issue_normalized_interval(
            validated_input,
            audit_ledger=audit_ledger,
            difficulty_scale_resolver=difficulty_scale_resolver,
            clock=clock,
        )

    cohort = _resolve_verified_cohort(resolver, validated_input.calibration_cohort_id)
    calibration = calibrate_normalized_conformal(
        cohort.calibration_predictions,
        split_manifest=cohort.split_manifest,
    )
    artifact = _calibration_artifact(cohort, calibration)
    warnings = [COVERAGE_SCOPE_WARNING]
    if cohort.calibration_scope == "target_domain_recalibration":
        warnings.append(TARGET_DOMAIN_RECALIBRATION_WARNING)

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value,
        tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        model_version=calibration.model_version,
        data_version=calibration.data_version,
        feature_version=calibration.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
            "artifact": artifact,
        },
        uncertainty=None,
        warnings=warnings,
        provenance=list(cohort.provenance),
        created_at=_execution_timestamp(clock),
    )


def register_calibrate_prediction_interval_tool(
    registry: ToolRegistry,
    *,
    resolver: VerifiedNormalizedCalibrationCohortResolver,
    audit_ledger: AuditLedger | None = None,
    difficulty_scale_resolver: VerifiedPredictionDifficultyScaleResolver | None = None,
    clock: Clock = _utc_now,
) -> RegisteredTool[CalibratePredictionIntervalToolInput]:
    """Register the trusted normalized-conformal calibration tool for one context."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
            input_model=CalibratePredictionIntervalToolInput,
            executor=lambda input_value: execute_calibrate_prediction_interval_tool(
                input_value,
                resolver=resolver,
                audit_ledger=audit_ledger,
                difficulty_scale_resolver=difficulty_scale_resolver,
                clock=clock,
            ),
        )
    )
