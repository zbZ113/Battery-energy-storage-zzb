"""Trusted ingestion of newly observed SOH evidence for online calibration.

The public tool contract deliberately accepts only a server-issued measurement
batch identifier.  The raw diagnostic capacities, versions and provenance are
resolved by an injected trusted-store adapter; no API, Agent, MCP client or UI
caller can submit them directly.  This source tool derives SOH deterministically
from that trusted batch and emits the standard evidence envelope consumed by
the online-calibration tool.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

OBSERVED_SOH_INGESTION_TOOL_VERSION = "observed-soh-ingestion-v1"
OBSERVED_SOH_INGESTION_MODEL_VERSION = "observed-soh-ingestion-rule-engine-v1"
NEWLY_OBSERVED_SOH_EVIDENCE_TYPE = "quanxin_life.newly_observed_soh_evidence.v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _VerifiedMeasurementModel(ContractModel):
    """Internal-store record contract with finite numerical fields only."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class VerifiedNewlyObservedMeasurement(_VerifiedMeasurementModel):
    """A trusted diagnostic discharge record; SOH is never supplied by a caller."""

    measurement_id: str
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cycle: int = Field(ge=0, strict=True)
    discharge_capacity_ah: float = Field(gt=0, strict=True, allow_inf_nan=False)
    reference_capacity_ah: float = Field(gt=0, strict=True, allow_inf_nan=False)
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
    def require_utc_measured_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at must include a timezone")
        return value.astimezone(UTC)


class VerifiedObservationBatch(_VerifiedMeasurementModel):
    """One verified, single-cell measurement batch returned by a trusted resolver."""

    measurement_batch_id: str = Field(min_length=1)
    measurements: tuple[VerifiedNewlyObservedMeasurement, ...] = Field(min_length=1)
    reference_capacity_method: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_consistent_trusted_measurements(self) -> VerifiedObservationBatch:
        identities = {(item.dataset_id, item.cell_id) for item in self.measurements}
        if len(identities) != 1:
            raise ValueError("verified measurements must contain one dataset and one cell")
        cycles = tuple(item.cycle for item in self.measurements)
        if any(current <= previous for previous, current in pairwise(cycles)):
            raise ValueError("verified measurement cycles must be strictly increasing")
        timestamps = tuple(item.measured_at for item in self.measurements)
        if any(current < previous for previous, current in pairwise(timestamps)):
            raise ValueError("verified measurement timestamps must not move backward")
        reference_capacities = {item.reference_capacity_ah for item in self.measurements}
        if len(reference_capacities) != 1:
            raise ValueError("verified measurements must use one reference_capacity_ah")
        measurement_ids = tuple(item.measurement_id for item in self.measurements)
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("verified measurement_ids must be unique")
        if not any(item.source_kind is SourceKind.NEWLY_OBSERVED for item in self.provenance):
            raise ValueError("verified provenance must include a NEWLY_OBSERVED source record")
        return self


class VerifiedMeasurementResolver(Protocol):
    """Server-side adapter for source-verified BMS, file or lab-test records."""

    def resolve_verified_observation_batch(
        self, measurement_batch_id: str
    ) -> VerifiedObservationBatch: ...


class NewlyObservedSOHIngestionInput(ContractModel):
    """Public request containing only a trusted-store batch reference."""

    measurement_batch_id: str = Field(min_length=1)

    @field_validator("measurement_batch_id")
    @classmethod
    def require_nonblank_batch_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("measurement_batch_id must not be blank")
        return normalized


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_batch(
    resolver: VerifiedMeasurementResolver,
    measurement_batch_id: str,
) -> VerifiedObservationBatch:
    """Detach and revalidate a resolver return before it enters numerical code."""

    resolved = resolver.resolve_verified_observation_batch(measurement_batch_id)
    if resolved.measurement_batch_id != measurement_batch_id:
        raise ValueError("trusted measurement resolver returned a mismatched batch identifier")
    return VerifiedObservationBatch.model_validate(resolved.model_dump(mode="json"))


def _measurement_payload(measurement: VerifiedNewlyObservedMeasurement) -> dict[str, object]:
    soh = measurement.discharge_capacity_ah / measurement.reference_capacity_ah
    if soh > 1.5:
        raise ValueError("derived SOH must be no greater than 1.5")
    return {
        "measurement_id": measurement.measurement_id,
        "dataset_id": measurement.dataset_id,
        "cell_id": measurement.cell_id,
        "cycle": measurement.cycle,
        "soh": soh,
        "source_kind": SourceKind.NEWLY_OBSERVED.value,
        "measured_at": measurement.measured_at.isoformat(),
        "source_record_hash": measurement.source_record_hash,
    }


def execute_ingest_newly_observed_soh_tool(
    input_value: NewlyObservedSOHIngestionInput,
    *,
    resolver: VerifiedMeasurementResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Resolve a trusted batch and derive standard online-calibration evidence."""

    validated_input = NewlyObservedSOHIngestionInput.model_validate(
        input_value.model_dump(mode="json")
    )
    verified_batch = _resolve_verified_batch(resolver, validated_input.measurement_batch_id)
    created_at = _execution_timestamp(clock)
    if any(item.measured_at > created_at for item in verified_batch.measurements):
        raise ValueError("measurement timestamps cannot be later than execution time")

    first = verified_batch.measurements[0]
    observations = [_measurement_payload(item) for item in verified_batch.measurements]
    artifact = {
        "measurement_batch_id": verified_batch.measurement_batch_id,
        "dataset_id": first.dataset_id,
        "cell_id": first.cell_id,
        "reference_capacity_ah": first.reference_capacity_ah,
        "reference_capacity_method": verified_batch.reference_capacity_method,
        "observations": observations,
        "observation_count": len(observations),
        "data_version": verified_batch.data_version,
        "feature_version": verified_batch.feature_version,
        "split_version": verified_batch.split_version,
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.INGEST_NEWLY_OBSERVED_SOH.value,
        tool_version=OBSERVED_SOH_INGESTION_TOOL_VERSION,
        model_version=OBSERVED_SOH_INGESTION_MODEL_VERSION,
        data_version=verified_batch.data_version,
        feature_version=verified_batch.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
            "artifact": artifact,
        },
        uncertainty=None,
        warnings=[],
        provenance=list(verified_batch.provenance),
        created_at=created_at,
    )


def register_ingest_newly_observed_soh_tool(
    registry: ToolRegistry,
    *,
    resolver: VerifiedMeasurementResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[NewlyObservedSOHIngestionInput]:
    """Register the context-bound source tool without global service initialization."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.INGEST_NEWLY_OBSERVED_SOH,
            tool_version=OBSERVED_SOH_INGESTION_TOOL_VERSION,
            input_model=NewlyObservedSOHIngestionInput,
            executor=lambda input_value: execute_ingest_newly_observed_soh_tool(
                input_value,
                resolver=resolver,
                clock=clock,
            ),
        )
    )
