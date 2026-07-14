"""Auditable online individual-parameter calibration from standard evidence.

Callers submit only two audit-ledger result IDs and a calibration-config-bound
version.  The tool rejects the legacy bare trajectory and observation payloads;
all numbers are decoded from the versioned evidence envelopes emitted by the
trajectory predictor and newly observed SOH ingestion tool.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.online.individual_calibration import (
    FrozenGlobalTrajectory,
    IndividualCalibrationOutcome,
    IndividualTrajectoryCalibrator,
    NewlyObservedSOH,
)
from quanxin_life.tools.observed_soh_ingestion import (
    NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
    OBSERVED_SOH_INGESTION_MODEL_VERSION,
    OBSERVED_SOH_INGESTION_TOOL_VERSION,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)
from quanxin_life.tools.trajectory_prediction import (
    PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE,
    TRAJECTORY_PREDICTION_TOOL_VERSION,
)

ONLINE_UPDATE_TOOL_VERSION = "online-update-tool-v1"
ONLINE_CALIBRATION_EVIDENCE_TYPE = "quanxin_life.online_individual_calibration.v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _EvidenceModel(ContractModel):
    """Strict, finite contracts for revalidated upstream evidence envelopes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class PredictedTrajectoryArtifact(_EvidenceModel):
    """The trajectory envelope shape eligible for online calibration only."""

    record_batch_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    prediction_cycles: tuple[int, ...]
    predicted_soh: tuple[float, ...]
    eol80_crossing: dict[str, object]
    derived_rul_cycle: int | None = Field(default=None, ge=0)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    upstream_result_id: str
    model_condition_feature_names: tuple[str, ...]
    model_artifact_status: str = Field(min_length=1)

    @field_validator("upstream_result_id")
    @classmethod
    def require_uuid_upstream_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_id must be a UUID string") from exc
        return value

    @model_validator(mode="after")
    def require_valid_frozen_trajectory(self) -> PredictedTrajectoryArtifact:
        FrozenGlobalTrajectory(
            dataset_id=self.dataset_id,
            cell_id=self.cell_id,
            cutoff_cycle=self.cutoff_cycle,
            cycles=self.prediction_cycles,
            soh=self.predicted_soh,
            model_version=self.model_version,
            feature_version=self.feature_version,
            split_version=self.split_version,
            data_version=self.data_version,
        )
        if not self.model_condition_feature_names:
            raise ValueError("model_condition_feature_names must be non-empty")
        if any(not name for name in self.model_condition_feature_names):
            raise ValueError("model_condition_feature_names must not contain blanks")
        return self


class NewlyObservedMeasurementArtifact(_EvidenceModel):
    """One trusted new diagnostic measurement already converted to SOH."""

    measurement_id: str
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    soh: float = Field(ge=0.0, le=1.5)
    source_kind: SourceKind
    measured_at: datetime
    source_record_hash: Sha256

    @field_validator("measurement_id")
    @classmethod
    def require_uuid_measurement_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("measurement_id must be a UUID string") from exc
        return value

    @field_validator("measured_at")
    @classmethod
    def normalize_measurement_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_newly_observed_source_kind(self) -> NewlyObservedMeasurementArtifact:
        if self.source_kind is not SourceKind.NEWLY_OBSERVED:
            raise ValueError("observation artifacts must be NEWLY_OBSERVED")
        return self


class NewlyObservedSOHArtifact(_EvidenceModel):
    """The source-tool evidence envelope consumed by online calibration only."""

    measurement_batch_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    reference_capacity_ah: float = Field(gt=0)
    reference_capacity_method: str = Field(min_length=1)
    observations: tuple[NewlyObservedMeasurementArtifact, ...] = Field(min_length=1)
    observation_count: int = Field(ge=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_measurement_batch(self) -> NewlyObservedSOHArtifact:
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must equal the observation artifact length")
        identities = {(item.dataset_id, item.cell_id) for item in self.observations}
        if identities != {(self.dataset_id, self.cell_id)}:
            raise ValueError("observation artifact identity must match every measurement")
        cycles = tuple(item.cycle for item in self.observations)
        if any(current <= previous for previous, current in pairwise(cycles)):
            raise ValueError("observation artifact cycles must be strictly increasing")
        measurement_ids = tuple(item.measurement_id for item in self.observations)
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("observation artifact measurement IDs must be unique")
        return self


class UpdateCellParametersToolInput(ContractModel):
    """Only ledger IDs and a config-bound update version may enter this tool."""

    trajectory_result_id: str
    observation_result_id: str
    update_version: str = Field(min_length=1)

    @field_validator("trajectory_result_id", "observation_result_id")
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            message = "trajectory_result_id and observation_result_id must be UUID strings"
            raise ValueError(message) from exc
        return value

    @field_validator("update_version")
    @classmethod
    def require_nonblank_update_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("update_version must not be blank")
        return value

    @model_validator(mode="after")
    def require_distinct_upstream_results(self) -> UpdateCellParametersToolInput:
        if self.trajectory_result_id == self.observation_result_id:
            raise ValueError("trajectory_result_id and observation_result_id must differ")
        return self


def _calibration_config_evidence(
    calibrator: IndividualTrajectoryCalibrator,
) -> tuple[dict[str, object], str, str]:
    normalized_config = calibrator.config.model_dump(mode="json")
    config_hash = sha256_canonical(normalized_config)
    config_version = calibrator.config.config_version
    expected_update_version = f"{config_version}:{config_hash}"
    return normalized_config, config_hash, expected_update_version


def _require_matching_update_version(
    *,
    update_version: str,
    expected_update_version: str,
) -> None:
    if update_version != expected_update_version:
        raise ValueError(
            "update_version must exactly match the active calibration config version and hash"
        )


def _require_artifact(
    result: ToolResult,
    *,
    expected_type: str,
    context: str,
) -> Mapping[str, object]:
    if result.values.get("artifact_type") != expected_type:
        raise ValueError(f"registered {context} result must contain the expected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"registered {context} result must contain values.artifact")
    return artifact


def _decode_frozen_global_trajectory(
    result: ToolResult,
) -> tuple[FrozenGlobalTrajectory, PredictedTrajectoryArtifact]:
    if result.tool_name != StandardToolName.PREDICT_SOH_TRAJECTORY.value:
        raise ValueError("trajectory_result_id must resolve to a predict_soh_trajectory result")
    if result.tool_version != TRAJECTORY_PREDICTION_TOOL_VERSION:
        raise ValueError("registered trajectory result has an unsupported tool_version")
    artifact = _require_artifact(
        result,
        expected_type=PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE,
        context="trajectory",
    )
    try:
        evidence = PredictedTrajectoryArtifact.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("registered trajectory artifact does not satisfy its contract") from exc
    for field_name in ("model_version", "data_version", "feature_version"):
        if getattr(result, field_name) != getattr(evidence, field_name):
            raise ValueError(f"registered trajectory result {field_name} must match its artifact")
    trajectory = FrozenGlobalTrajectory(
        dataset_id=evidence.dataset_id,
        cell_id=evidence.cell_id,
        cutoff_cycle=evidence.cutoff_cycle,
        cycles=evidence.prediction_cycles,
        soh=evidence.predicted_soh,
        model_version=evidence.model_version,
        feature_version=evidence.feature_version,
        split_version=evidence.split_version,
        data_version=evidence.data_version,
    )
    return trajectory, evidence


def _decode_new_observations(
    *,
    result: ToolResult,
    global_trajectory: FrozenGlobalTrajectory,
) -> tuple[NewlyObservedSOH, ...]:
    if result.tool_name != StandardToolName.INGEST_NEWLY_OBSERVED_SOH.value:
        raise ValueError(
            "observation_result_id must resolve to an ingest_newly_observed_soh result"
        )
    if result.tool_version != OBSERVED_SOH_INGESTION_TOOL_VERSION:
        raise ValueError("registered observation result has an unsupported tool_version")
    if result.model_version != OBSERVED_SOH_INGESTION_MODEL_VERSION:
        raise ValueError("registered observation result has an unsupported model_version")
    artifact = _require_artifact(
        result,
        expected_type=NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
        context="observation",
    )
    try:
        evidence = NewlyObservedSOHArtifact.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("registered observation artifact does not satisfy its contract") from exc
    for field_name in ("data_version", "feature_version"):
        if getattr(result, field_name) != getattr(evidence, field_name):
            raise ValueError(f"registered observation result {field_name} must match its artifact")
    if (
        evidence.data_version != global_trajectory.data_version
        or evidence.feature_version != global_trajectory.feature_version
        or evidence.split_version != global_trajectory.split_version
    ):
        raise ValueError("registered observation artifact does not match trajectory versions")
    if (
        evidence.dataset_id != global_trajectory.dataset_id
        or evidence.cell_id != global_trajectory.cell_id
    ):
        raise ValueError(
            "registered observation artifact does not match global trajectory identity"
        )
    if not any(record.source_kind is SourceKind.NEWLY_OBSERVED for record in result.provenance):
        raise ValueError("registered observation result lacks NEWLY_OBSERVED provenance")

    observations = tuple(
        NewlyObservedSOH(
            dataset_id=item.dataset_id,
            cell_id=item.cell_id,
            cycle=item.cycle,
            soh=item.soh,
            source_kind=item.source_kind,
        )
        for item in evidence.observations
    )
    return observations


def _combine_provenance(results: Sequence[ToolResult]) -> list[ProvenanceRecord]:
    combined: list[ProvenanceRecord] = []
    seen: set[str] = set()
    for result in results:
        for record in result.provenance:
            fingerprint = sha256_canonical(record.model_dump(mode="json"))
            if fingerprint not in seen:
                seen.add(fingerprint)
                combined.append(record)
    return combined


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def execute_update_cell_parameters_tool(
    input_value: UpdateCellParametersToolInput,
    *,
    audit_ledger: AuditLedger,
    calibrator: IndividualTrajectoryCalibrator | None = None,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Calibrate bounded individual parameters from ledger-derived evidence only."""

    validated_input = UpdateCellParametersToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    numerical_calibrator = calibrator or IndividualTrajectoryCalibrator()
    normalized_config, config_hash, expected_update_version = _calibration_config_evidence(
        numerical_calibrator
    )
    _require_matching_update_version(
        update_version=validated_input.update_version,
        expected_update_version=expected_update_version,
    )
    trajectory_result = audit_ledger.resolve_registered_result(
        validated_input.trajectory_result_id
    )
    observation_result = audit_ledger.resolve_registered_result(
        validated_input.observation_result_id
    )
    global_trajectory, trajectory_evidence = _decode_frozen_global_trajectory(trajectory_result)
    observations = _decode_new_observations(
        result=observation_result,
        global_trajectory=global_trajectory,
    )
    outcome = IndividualCalibrationOutcome.model_validate(
        numerical_calibrator.calibrate(
            global_trajectory=global_trajectory,
            observations=observations,
            update_version=validated_input.update_version,
        ).model_dump(mode="json")
    )
    outcome_payload = outcome.model_dump(mode="json")
    warnings = list(outcome.warnings)
    if outcome.reason_code != "ADAPTED":
        warnings.append(outcome.reason_code)
    if trajectory_evidence.model_artifact_status != "VERIFIED_ARTIFACT":
        warnings.append("UPSTREAM_MODEL_ARTIFACT_UNREGISTERED")

    calibration_config = {
        "config_version": numerical_calibrator.config.config_version,
        "config_hash": config_hash,
        "normalized_config": normalized_config,
    }
    audit_payload = dict(outcome_payload["audit"])
    audit_payload["calibration_config"] = calibration_config
    upstream_results = (trajectory_result, observation_result)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.UPDATE_CELL_PARAMETERS.value,
        tool_version=ONLINE_UPDATE_TOOL_VERSION,
        model_version=global_trajectory.model_version,
        data_version=global_trajectory.data_version,
        feature_version=global_trajectory.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": ONLINE_CALIBRATION_EVIDENCE_TYPE,
            "artifact": {
                "status": outcome.status.value,
                "reason_code": outcome.reason_code,
                "trajectory": {
                    "cycles": outcome_payload["trajectory_cycles"],
                    "adapted_soh": outcome_payload["adapted_soh"],
                },
                "audit": audit_payload,
                "calibration_config": calibration_config,
                "warnings": outcome_payload["warnings"],
                "upstream_result_ids": [
                    validated_input.trajectory_result_id,
                    validated_input.observation_result_id,
                ],
                "split_version": global_trajectory.split_version,
                "model_artifact_status": trajectory_evidence.model_artifact_status,
            },
        },
        uncertainty=None,
        warnings=list(dict.fromkeys(warnings)),
        provenance=_combine_provenance(upstream_results),
        created_at=_execution_timestamp(clock),
    )


def register_update_cell_parameters_tool(
    registry: ToolRegistry,
    *,
    audit_ledger: AuditLedger,
    calibrator: IndividualTrajectoryCalibrator | None = None,
    clock: Clock = _utc_now,
) -> RegisteredTool[UpdateCellParametersToolInput]:
    """Register the sole ledger-bound frozen-horizon individual update tool."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.UPDATE_CELL_PARAMETERS,
            tool_version=ONLINE_UPDATE_TOOL_VERSION,
            input_model=UpdateCellParametersToolInput,
            executor=lambda input_value: execute_update_cell_parameters_tool(
                input_value,
                audit_ledger=audit_ledger,
                calibrator=calibrator,
                clock=clock,
            ),
        )
    )
