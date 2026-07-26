"""Ledger-bound EOL80 cycle-life prediction from early-cycle evidence.

The public tool accepts one registered early-cycle feature result identifier.
Numerical EOL80 values are produced only by an already-fitted, injected
predictor.  The tool does not deserialize model artefacts and never accepts
caller-provided feature rows, labels, RUL, or uncertainty values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from numbers import Real
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator

from quanxin_life.audit.project_ledger import (
    BoundProjectResultResolver,
    ProjectResultLedger,
    RegisteredResultResolver,
)
from quanxin_life.core import (
    LifePrediction,
    PredictionTarget,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

CYCLE_LIFE_PREDICTION_TOOL_VERSION = "cycle-life-prediction-tool-v1"
PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE = "quanxin_life.predicted_cycle_life.v1"
MODEL_ARTIFACT_UNREGISTERED_WARNING = "MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY"
MODEL_ARTIFACT_VERIFIED_STATUS = "VERIFIED_ARTIFACT"
Clock = Callable[[], datetime]


class CycleLifePredictor(Protocol):
    """The minimal fitted-predictor surface accepted by this service layer."""

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    feature_names: tuple[str, ...]

    def predict(
        self,
        *,
        dataset_id: str,
        cell_id: str,
        features: Mapping[str, float],
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> LifePrediction: ...


class ModelArtifactResolver(Protocol):
    """Structural boundary implemented by the governed model registry."""

    def resolve(
        self,
        artifact_id: str,
        *,
        model_version: str | None = None,
        data_version: str | None = None,
        feature_version: str | None = None,
        split_version: str | None = None,
    ) -> Any: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _EvidenceModel(ContractModel):
    """Strict finite evidence decoded from a registered upstream result only."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class PredictCycleLifeToolInput(ContractModel):
    """Only one immutable early-feature result ID is caller-controlled."""

    upstream_result_id: str

    @field_validator("upstream_result_id")
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_id must be a UUID string") from exc
        return value


class EarlyCycleFeatureEvidence(_EvidenceModel):
    """The subset of the standard early-feature envelope needed for EOL80."""

    record_batch_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    observed_cycles: tuple[int, ...]
    observed_soh: tuple[float, ...]
    reference_capacity_ah: float = Field(gt=0)
    reference_capacity_method: str = Field(min_length=1)
    feature_values: dict[str, float | None]
    condition_features: dict[str, float]
    source_cycles: tuple[int, ...]
    feature_warnings: tuple[str, ...] = ()
    source_manifest_hash: Sha256
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @field_validator("condition_features")
    @classmethod
    def require_finite_conditions(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("condition_features must be non-empty")
        for name, raw_value in value.items():
            if not name:
                raise ValueError("condition_features must not contain blank names")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"condition feature {name} must be finite")
            if not math.isfinite(float(raw_value)):
                raise ValueError(f"condition feature {name} must be finite")
        return value

    @field_validator("feature_values")
    @classmethod
    def require_finite_available_features(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        if not value:
            raise ValueError("feature_values must be non-empty")
        for name, raw_value in value.items():
            if not name:
                raise ValueError("feature_values must not contain blank names")
            if raw_value is None:
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"feature value {name} must be finite when available")
            if not math.isfinite(float(raw_value)):
                raise ValueError(f"feature value {name} must be finite when available")
        return value


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _require_artifact(result: ToolResult) -> Mapping[str, object]:
    if result.values.get("artifact_type") != EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE:
        raise ValueError("registered early feature result must contain the expected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("registered early feature result must contain values.artifact")
    return artifact


def _decode_early_feature_evidence(result: ToolResult) -> EarlyCycleFeatureEvidence:
    if result.tool_name != StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value:
        raise ValueError(
            "upstream_result_id must resolve to an extract_early_cycle_features result"
        )
    if result.tool_version != EARLY_CYCLE_FEATURE_TOOL_VERSION:
        raise ValueError("registered early feature result has an unsupported tool_version")
    if result.model_version != EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION:
        raise ValueError("registered early feature result has an unsupported model_version")
    artifact = _require_artifact(result)
    try:
        evidence = EarlyCycleFeatureEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("registered early feature artifact does not satisfy its contract") from exc
    for field_name in ("data_version", "feature_version"):
        if getattr(result, field_name) != getattr(evidence, field_name):
            raise ValueError(
                f"registered early feature result {field_name} must match its artifact"
            )
    if not any(record.source_kind is SourceKind.OBSERVED for record in result.provenance):
        raise ValueError("registered early feature result lacks OBSERVED provenance")
    return evidence


def _validate_predictor_context(
    predictor: CycleLifePredictor,
    evidence: EarlyCycleFeatureEvidence,
) -> tuple[str, ...]:
    expected_values = {
        "feature_version": evidence.feature_version,
        "split_version": evidence.split_version,
        "data_version": evidence.data_version,
        "cutoff_cycle": evidence.cutoff_cycle,
    }
    for name, expected in expected_values.items():
        if getattr(predictor, name, None) != expected:
            raise ValueError(f"fitted predictor {name} must match early feature evidence")
    if not isinstance(predictor.model_version, str) or not predictor.model_version.strip():
        raise ValueError("fitted predictor model_version must be non-empty")
    raw_feature_names = getattr(predictor, "feature_names", None)
    if not isinstance(raw_feature_names, tuple) or not raw_feature_names:
        raise ValueError("fitted predictor feature_names must be a non-empty tuple")
    if any(not isinstance(name, str) or not name for name in raw_feature_names):
        raise ValueError("fitted predictor feature_names must contain non-empty strings")
    if len(raw_feature_names) != len(set(raw_feature_names)):
        raise ValueError("fitted predictor feature_names must be unique")
    return raw_feature_names


def _select_predictor_features(
    evidence: EarlyCycleFeatureEvidence, feature_names: Sequence[str]
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for name in feature_names:
        if name not in evidence.condition_features:
            raise ValueError(
                f"early feature evidence is missing required predictor feature: {name}"
            )
        value = evidence.condition_features[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"early feature evidence has non-finite predictor feature: {name}")
        selected[name] = float(value)
    return selected


def _validate_prediction(
    prediction: object,
    *,
    evidence: EarlyCycleFeatureEvidence,
    predictor: CycleLifePredictor,
) -> LifePrediction:
    if not isinstance(prediction, LifePrediction):
        raise TypeError("fitted predictor must return a LifePrediction")
    validated = LifePrediction.model_validate(prediction.model_dump(mode="json"))
    expected_values = {
        "dataset_id": evidence.dataset_id,
        "cell_id": evidence.cell_id,
        "cutoff_cycle": evidence.cutoff_cycle,
        "feature_version": evidence.feature_version,
        "split_version": evidence.split_version,
        "data_version": evidence.data_version,
        "model_version": predictor.model_version,
    }
    for name, expected in expected_values.items():
        if getattr(validated, name) != expected:
            raise ValueError(f"predictor output {name} must match verified prediction context")
    if validated.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("predictor output target must be EOL80_CYCLE")
    if not validated.right_censored or validated.observed_eol_cycle is not None:
        raise ValueError("predictor output must not expose an observed EOL80 label")
    return validated


def execute_predict_cycle_life_tool(
    input_value: PredictCycleLifeToolInput,
    *,
    predictor: CycleLifePredictor,
    audit_ledger: RegisteredResultResolver,
    model_artifact_registry: ModelArtifactResolver | None = None,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Predict EOL80 from one trusted early-feature record without loading artefacts."""

    validated_input = PredictCycleLifeToolInput.model_validate(input_value.model_dump(mode="json"))
    upstream_result = audit_ledger.resolve_registered_result(validated_input.upstream_result_id)
    evidence = _decode_early_feature_evidence(upstream_result)
    feature_names = _validate_predictor_context(predictor, evidence)
    features = _select_predictor_features(evidence, feature_names)
    prediction = _validate_prediction(
        predictor.predict(
            dataset_id=evidence.dataset_id,
            cell_id=evidence.cell_id,
            features=features,
            cutoff_cycle=evidence.cutoff_cycle,
            feature_version=evidence.feature_version,
            split_version=evidence.split_version,
            data_version=evidence.data_version,
        ),
        evidence=evidence,
        predictor=predictor,
    )
    artifact_status, artifact_id, artifact_sha256, artifact_provenance, warnings = (
        _resolve_model_artifact(
            predictor=predictor,
            registry=model_artifact_registry,
        )
    )
    prediction_payload = prediction.model_dump(mode="json")
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
        feature_version=prediction.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": evidence.record_batch_id,
                "dataset_id": evidence.dataset_id,
                "cell_id": evidence.cell_id,
                "cutoff_cycle": evidence.cutoff_cycle,
                "life_prediction": prediction_payload,
                "derived_rul_cycle": prediction.derived_rul,
                "source_manifest_hash": evidence.source_manifest_hash,
                "upstream_result_id": validated_input.upstream_result_id,
                "split_version": evidence.split_version,
                "used_feature_names": list(feature_names),
                "model_artifact_status": artifact_status,
                "model_artifact_id": artifact_id,
                "model_artifact_sha256": artifact_sha256,
            },
        },
        uncertainty=None,
        warnings=warnings,
        provenance=[*upstream_result.provenance, *artifact_provenance],
        created_at=_execution_timestamp(clock),
    )


def register_predict_cycle_life_tool(
    registry: ToolRegistry,
    *,
    predictor: CycleLifePredictor,
    audit_ledger: RegisteredResultResolver,
    model_artifact_registry: ModelArtifactResolver | None = None,
    clock: Clock = _utc_now,
) -> RegisteredTool[PredictCycleLifeToolInput]:
    """Register the sole evidence-bound EOL80 predictor for this service context."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
            input_model=PredictCycleLifeToolInput,
            executor=lambda input_value: execute_predict_cycle_life_tool(
                input_value,
                predictor=predictor,
                audit_ledger=audit_ledger,
                model_artifact_registry=model_artifact_registry,
                clock=clock,
            ),
        )
    )


def register_project_predict_cycle_life_tool(
    registry: ToolRegistry,
    *,
    predictor: CycleLifePredictor,
    project_audit_ledger: ProjectResultLedger,
    model_artifact_registry: ModelArtifactResolver | None = None,
    clock: Clock = _utc_now,
) -> RegisteredTool[PredictCycleLifeToolInput]:
    """Register the same numerical predictor behind project-bound evidence."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
            input_model=PredictCycleLifeToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: execute_predict_cycle_life_tool(
                input_value,
                predictor=predictor,
                audit_ledger=BoundProjectResultResolver(
                    project_audit_ledger,
                    context,
                ),
                model_artifact_registry=model_artifact_registry,
                clock=clock,
            ),
        )
    )


def _resolve_model_artifact(
    *,
    predictor: CycleLifePredictor,
    registry: ModelArtifactResolver | None,
) -> tuple[str, str | None, str | None, list[ProvenanceRecord], list[str]]:
    artifact_id = getattr(predictor, "model_artifact_id", None)
    artifact_sha256 = getattr(predictor, "model_artifact_sha256", None)
    if registry is None or artifact_id is None or artifact_sha256 is None:
        return (
            "UNREGISTERED_IN_MEMORY",
            None,
            None,
            [],
            [MODEL_ARTIFACT_UNREGISTERED_WARNING],
        )
    verified = registry.resolve(
        artifact_id,
        model_version=predictor.model_version,
        data_version=predictor.data_version,
        feature_version=predictor.feature_version,
        split_version=predictor.split_version,
    )
    manifest = verified.manifest
    if manifest.sha256 != artifact_sha256:
        raise ValueError("predictor artifact SHA-256 does not match the verified registry")
    if manifest.cutoff_cycle != predictor.cutoff_cycle:
        raise ValueError("predictor cutoff_cycle does not match the verified artifact")
    if manifest.feature_names != predictor.feature_names:
        raise ValueError("predictor feature_names do not match the verified artifact")
    return (
        MODEL_ARTIFACT_VERIFIED_STATUS,
        manifest.artifact_id,
        manifest.sha256,
        [
            ProvenanceRecord(
                source_id=f"model-artifact-{manifest.artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://model/{manifest.artifact_id}",
                sha256=manifest.sha256,
                description="SHA-256 verified native model artifact",
                created_at=manifest.created_at,
            )
        ],
        [],
    )
