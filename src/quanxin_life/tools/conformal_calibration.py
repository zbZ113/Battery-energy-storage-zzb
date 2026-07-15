"""Trusted normalized-conformal calibration evidence for EOL80 intervals.

The public tool receives only a server-issued calibration cohort identifier.
Observed labels, point predictions, difficulty scales, split membership and the
calibration policy are resolved by a trusted adapter.  This prevents callers or
LLMs from injecting labels, residuals, alpha values, or a fabricated interval.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.core import (
    NormalizedConformalCalibration,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)
from quanxin_life.uncertainty import ScaledLifePrediction, calibrate_normalized_conformal

CONFORMAL_CALIBRATION_TOOL_VERSION = "conformal-calibration-tool-v1"
NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE = (
    "quanxin_life.normalized_conformal_calibration.v1"
)
COVERAGE_SCOPE_WARNING = "COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT"
TARGET_DOMAIN_RECALIBRATION_WARNING = "TARGET_DOMAIN_RECALIBRATION_APPLIED"
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


class CalibratePredictionIntervalToolInput(ContractModel):
    """Public request containing only a trusted calibration cohort identifier."""

    calibration_cohort_id: str = Field(min_length=1)

    @field_validator("calibration_cohort_id")
    @classmethod
    def require_nonblank_cohort_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("calibration_cohort_id must not be blank")
        return normalized


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


def execute_calibrate_prediction_interval_tool(
    input_value: CalibratePredictionIntervalToolInput,
    *,
    resolver: VerifiedNormalizedCalibrationCohortResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Create one normalized conformal calibration from a trusted cohort.

    The method does not construct an interval for a target cell.  A later
    ledger-bound interval-application tool must consume this calibration along
    with a registered point prediction and a matching difficulty scale.
    """

    validated_input = CalibratePredictionIntervalToolInput.model_validate(
        input_value.model_dump(mode="json")
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
                clock=clock,
            ),
        )
    )
