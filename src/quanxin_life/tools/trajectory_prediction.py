"""Ledger-bound finite-horizon SOH trajectory prediction.

The tool accepts only one registered early-cycle feature result ID.  It rejects
the former untyped ``values["trajectory_input"]`` payload and consumes the
versioned evidence envelope emitted by ``extract_early_cycle_features``.  The
only numerical producer remains an already fitted in-memory
``HybridDegradationPredictor``; no serialised artefact is loaded here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from itertools import pairwise
from numbers import Real
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.models.hybrid_degradation import (
    EOL80_THRESHOLD,
    HybridDegradationPredictor,
    SOHTrajectoryPrediction,
)
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

TRAJECTORY_PREDICTION_TOOL_VERSION = "trajectory-prediction-tool-v1"
PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE = "quanxin_life.predicted_soh_trajectory.v1"


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite real number")
    return number


class PredictSOHTrajectoryToolInput(ContractModel):
    """Minimal caller input: one immutable, registered early-cycle result ID."""

    upstream_result_id: str

    @field_validator("upstream_result_id")
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_id must be a UUID string") from exc
        return value


class TrajectoryPredictionEvidence(ContractModel):
    """Strict, versioned early-cycle evidence emitted by the feature tool only."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

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

    @field_validator("observed_soh")
    @classmethod
    def require_permitted_observed_soh(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        validated: list[float] = []
        for soh in value:
            numeric_soh = _finite_real(soh, field_name="observed SOH")
            if numeric_soh <= 0 or numeric_soh > 1.5:
                raise ValueError("observed SOH values must be in (0, 1.5]")
            validated.append(numeric_soh)
        return tuple(validated)

    @field_validator("condition_features")
    @classmethod
    def require_finite_conditions(cls, value: dict[str, float]) -> dict[str, float]:
        validated: dict[str, float] = {}
        for name, raw_value in value.items():
            if not name:
                raise ValueError("condition feature names must be non-empty")
            validated[name] = _finite_real(raw_value, field_name=f"condition feature {name}")
        return validated

    @field_validator("feature_values")
    @classmethod
    def require_finite_or_explicitly_missing_features(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        validated: dict[str, float | None] = {}
        for name, raw_value in value.items():
            if not name:
                raise ValueError("feature value names must be non-empty")
            validated[name] = (
                None
                if raw_value is None
                else _finite_real(raw_value, field_name=f"feature value {name}")
            )
        return validated

    @model_validator(mode="after")
    def require_cutoff_safe_observations(self) -> TrajectoryPredictionEvidence:
        if len(self.observed_cycles) != len(self.observed_soh) or not self.observed_cycles:
            raise ValueError("observed_cycles and observed_soh must be non-empty and align")
        if any(cycle < 0 for cycle in self.observed_cycles):
            raise ValueError("observed_cycles must be non-negative")
        if any(current <= previous for previous, current in pairwise(self.observed_cycles)):
            raise ValueError("observed_cycles must be strictly increasing")
        if any(cycle > self.cutoff_cycle for cycle in self.observed_cycles):
            raise ValueError("observed_cycles cannot exceed cutoff_cycle")
        if self.observed_cycles[-1] != self.cutoff_cycle:
            raise ValueError("the last observed cycle must equal cutoff_cycle")
        if self.observed_soh[-1] <= EOL80_THRESHOLD:
            raise ValueError("observed SOH at cutoff already reached EOL80")
        if not self.condition_features:
            raise ValueError("condition_features must be non-empty")
        if not self.source_cycles:
            raise ValueError("source_cycles must be non-empty")
        if any(cycle < 0 or cycle > self.cutoff_cycle for cycle in self.source_cycles):
            raise ValueError("source_cycles must remain within the feature cutoff")
        if any(current <= previous for previous, current in pairwise(self.source_cycles)):
            raise ValueError("source_cycles must be strictly increasing")
        return self


def _resolve_feature_evidence(
    input_value: PredictSOHTrajectoryToolInput,
    *,
    audit_ledger: AuditLedger,
) -> tuple[ToolResult, TrajectoryPredictionEvidence]:
    """Resolve the only accepted numerical payload from the public audit ledger."""

    upstream_result = audit_ledger.resolve_registered_result(input_value.upstream_result_id)
    if upstream_result.tool_name != StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value:
        raise ValueError("upstream_result_id must resolve to extract_early_cycle_features")
    if upstream_result.tool_version != EARLY_CYCLE_FEATURE_TOOL_VERSION:
        raise ValueError("registered feature result has an unsupported tool_version")
    if upstream_result.model_version != EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION:
        raise ValueError("registered feature result has an unsupported model_version")
    if upstream_result.values.get("artifact_type") != EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE:
        raise ValueError("registered feature result must contain the expected artifact_type")
    raw_payload = upstream_result.values.get("artifact")
    if not isinstance(raw_payload, Mapping):
        raise ValueError("registered feature result must contain values.artifact")
    try:
        evidence = TrajectoryPredictionEvidence.model_validate(raw_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"registered early-cycle artifact evidence is invalid: {exc}") from exc
    for field_name in ("data_version", "feature_version"):
        if getattr(upstream_result, field_name) != getattr(evidence, field_name):
            raise ValueError(f"registered feature result {field_name} must match its artifact")
    if not any(item.source_kind.value == "OBSERVED" for item in upstream_result.provenance):
        raise ValueError("registered feature result provenance must include OBSERVED evidence")
    return upstream_result, evidence


def _require_exact_predictor_context(
    evidence: TrajectoryPredictionEvidence,
    predictor: HybridDegradationPredictor,
) -> None:
    for name, received, expected in (
        ("cutoff_cycle", evidence.cutoff_cycle, predictor.cutoff_cycle),
        ("feature_version", evidence.feature_version, predictor.feature_version),
        ("split_version", evidence.split_version, predictor.split_version),
        ("data_version", evidence.data_version, predictor.data_version),
    ):
        if received != expected:
            raise ValueError(f"{name} must match the fitted hybrid-model context")


def _select_predictor_condition_features(
    evidence: TrajectoryPredictionEvidence,
    predictor: HybridDegradationPredictor,
) -> dict[str, float]:
    """Project source evidence onto the fitted model's explicit input schema.

    The feature tool deliberately preserves every available finite condition.
    A fitted model, however, must consume precisely the condition names it was
    trained with.  This function selects that declared subset without filling
    missing fields or altering values, and fails closed when evidence is
    insufficient for the model contract.
    """

    missing = [
        name
        for name in predictor.condition_feature_names
        if name not in evidence.condition_features
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"early-cycle artifact is missing required condition features: {joined}")
    return {name: evidence.condition_features[name] for name in predictor.condition_feature_names}


def _validated_prediction(
    *,
    evidence: TrajectoryPredictionEvidence,
    predictor: HybridDegradationPredictor,
) -> SOHTrajectoryPrediction:
    """Invoke the fitted model once and revalidate its public numerical contract."""

    _require_exact_predictor_context(evidence, predictor)
    model_conditions = _select_predictor_condition_features(evidence, predictor)
    raw_prediction = predictor.predict(
        dataset_id=evidence.dataset_id,
        cell_id=evidence.cell_id,
        observed_cycles=evidence.observed_cycles,
        observed_soh=evidence.observed_soh,
        condition_features=model_conditions,
        cutoff_cycle=evidence.cutoff_cycle,
        feature_version=evidence.feature_version,
        split_version=evidence.split_version,
        data_version=evidence.data_version,
    )
    prediction = SOHTrajectoryPrediction.model_validate(raw_prediction.model_dump(mode="json"))
    for name, received, expected in (
        ("dataset_id", prediction.dataset_id, evidence.dataset_id),
        ("cell_id", prediction.cell_id, evidence.cell_id),
        ("cutoff_cycle", prediction.cutoff_cycle, evidence.cutoff_cycle),
        ("feature_version", prediction.feature_version, predictor.feature_version),
        ("split_version", prediction.split_version, predictor.split_version),
        ("data_version", prediction.data_version, predictor.data_version),
        ("model_version", prediction.model_version, predictor.model_version),
    ):
        if received != expected:
            raise ValueError(f"predictor output {name} does not match the declared context")
    return prediction


def _execution_time_utc(clock: Callable[[], datetime] | None) -> datetime:
    """Read the execution clock and reject a non-UTC-aware test or service clock."""

    timestamp = datetime.now(UTC) if clock is None else clock()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return timestamp.astimezone(UTC)


def execute_predict_soh_trajectory_tool(
    input_value: PredictSOHTrajectoryToolInput,
    *,
    predictor: HybridDegradationPredictor,
    audit_ledger: AuditLedger,
    clock: Callable[[], datetime] | None = None,
) -> ToolResult:
    """Expose finite-horizon model evidence from one ledger-bound feature result."""

    validated_input = PredictSOHTrajectoryToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    upstream_result, evidence = _resolve_feature_evidence(
        validated_input,
        audit_ledger=audit_ledger,
    )
    prediction = _validated_prediction(evidence=evidence, predictor=predictor)
    crossing_payload = prediction.eol80_crossing.model_dump(mode="json")
    # This context-bound tool accepts a fitted in-memory predictor.  Until the
    # model-artifact registry is injected by the service factory, no artifact
    # manifest SHA-256 exists to support a formal-report claim.  Make that
    # degraded provenance state explicit instead of implying registration.
    warnings: list[str] = ["MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY"]
    if prediction.eol80_crossing.eol80_cycle is None:
        warnings.append("NO_EOL80_CROSSING_IN_FINITE_HORIZON")

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
        tool_version=TRAJECTORY_PREDICTION_TOOL_VERSION,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
        feature_version=prediction.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": evidence.record_batch_id,
                "dataset_id": prediction.dataset_id,
                "cell_id": prediction.cell_id,
                "cutoff_cycle": prediction.cutoff_cycle,
                "prediction_cycles": list(prediction.prediction_cycles),
                "predicted_soh": list(prediction.predicted_soh),
                "eol80_crossing": crossing_payload,
                "derived_rul_cycle": prediction.derived_rul_cycle,
                "model_version": prediction.model_version,
                "feature_version": prediction.feature_version,
                "split_version": prediction.split_version,
                "data_version": prediction.data_version,
                "upstream_result_id": upstream_result.result_id,
                "model_condition_feature_names": list(predictor.condition_feature_names),
                "model_artifact_status": "UNREGISTERED_IN_MEMORY",
            },
        },
        uncertainty={
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        },
        warnings=warnings,
        provenance=upstream_result.provenance,
        created_at=_execution_time_utc(clock),
    )


def register_predict_soh_trajectory_tool(
    registry: ToolRegistry,
    *,
    predictor: HybridDegradationPredictor,
    audit_ledger: AuditLedger,
    clock: Callable[[], datetime] | None = None,
) -> RegisteredTool[PredictSOHTrajectoryToolInput]:
    """Register the only in-memory, ledger-bound trajectory-tool implementation."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
            tool_version=TRAJECTORY_PREDICTION_TOOL_VERSION,
            input_model=PredictSOHTrajectoryToolInput,
            executor=lambda input_value: execute_predict_soh_trajectory_tool(
                input_value,
                predictor=predictor,
                audit_ledger=audit_ledger,
                clock=clock,
            ),
        )
    )
