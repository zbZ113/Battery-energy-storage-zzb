"""Trusted, cutoff-safe early-cycle trajectory evidence extraction.

The public contract carries only a server-issued record-batch ID.  Canonical
records, metadata, data versions and provenance are resolved by an injected
trusted-store adapter, revalidated, and then deterministically transformed into
the standard evidence envelope consumed by trajectory prediction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from statistics import median
from typing import Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import (
    CellMetadata,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig, extract_early_cycle_features
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

EARLY_CYCLE_FEATURE_TOOL_VERSION = "early-cycle-feature-tool-v1"
EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION = "early-cycle-feature-rule-engine-v1"
EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE = "quanxin_life.early_cycle_trajectory_evidence.v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _VerifiedBatchModel(ContractModel):
    """Internal trusted-store record contract with nonfinite values disabled."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class VerifiedEarlyCycleBatch(_VerifiedBatchModel):
    """A source-verified canonical record batch for exactly one cell and cutoff."""

    record_batch_id: str = Field(min_length=1)
    records: tuple[CycleRecord, ...] = Field(min_length=2)
    metadata: CellMetadata
    feature_config: EarlyCycleFeatureConfig
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    source_manifest_hash: Sha256
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_resolved_batch(self) -> VerifiedEarlyCycleBatch:
        identities = {(record.dataset_id, record.cell_id) for record in self.records}
        expected_identity = (self.metadata.dataset_id, self.metadata.cell_id)
        if identities != {expected_identity}:
            raise ValueError("metadata identity must match every canonical record")
        if any(record.cycle_index > self.feature_config.cutoff_cycle for record in self.records):
            raise ValueError("records must not include cycles after the feature cutoff")
        has_cutoff_diagnostic_capacity = any(
            record.cycle_index == self.feature_config.cutoff_cycle
            and record.valid
            and record.diagnostic
            and record.discharge_capacity_ah is not None
            and record.discharge_capacity_ah > 0
            for record in self.records
        )
        if not has_cutoff_diagnostic_capacity:
            raise ValueError("cutoff cycle requires a valid diagnostic discharge capacity")
        if not any(item.source_kind is SourceKind.OBSERVED for item in self.provenance):
            raise ValueError("provenance must include an OBSERVED source record")
        if not math.isfinite(self.metadata.nominal_capacity_ah):
            raise ValueError("metadata nominal_capacity_ah must be finite")
        if (
            self.metadata.reference_capacity_ah is not None
            and not math.isfinite(self.metadata.reference_capacity_ah)
        ):
            raise ValueError("metadata reference_capacity_ah must be finite")
        return self


class VerifiedEarlyCycleBatchResolver(Protocol):
    """Server-side adapter for source-verified canonical Parquet and metadata."""

    def resolve_verified_early_cycle_batch(
        self, record_batch_id: str
    ) -> VerifiedEarlyCycleBatch: ...


class ExtractEarlyCycleFeaturesToolInput(ContractModel):
    """Public request containing only a trusted early-cycle batch identifier."""

    record_batch_id: str = Field(min_length=1)

    @field_validator("record_batch_id")
    @classmethod
    def require_nonblank_record_batch_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("record_batch_id must not be blank")
        return normalized


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_batch(
    resolver: VerifiedEarlyCycleBatchResolver,
    record_batch_id: str,
) -> VerifiedEarlyCycleBatch:
    """Detach and revalidate the trusted-store return before numerical processing."""

    resolved = resolver.resolve_verified_early_cycle_batch(record_batch_id)
    if resolved.record_batch_id != record_batch_id:
        raise ValueError("trusted early-cycle resolver returned a mismatched batch identifier")
    return VerifiedEarlyCycleBatch.model_validate(resolved.model_dump(mode="json"))


def _diagnostic_capacity_by_cycle(records: Sequence[CycleRecord]) -> tuple[tuple[int, float], ...]:
    by_cycle: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record.valid and record.diagnostic and record.discharge_capacity_ah is not None:
            capacity = float(record.discharge_capacity_ah)
            if math.isfinite(capacity) and capacity > 0:
                by_cycle[record.cycle_index].append(capacity)
    capacities = tuple(
        (cycle, max(values)) for cycle, values in sorted(by_cycle.items()) if values
    )
    if len(capacities) < 2:
        raise ValueError(
            "trusted records require at least two diagnostic discharge capacity cycles"
        )
    return capacities


def _reference_capacity(
    metadata: CellMetadata,
    diagnostic_capacities: Sequence[tuple[int, float]],
) -> tuple[float, str]:
    if metadata.reference_capacity_ah is not None:
        return float(metadata.reference_capacity_ah), "metadata_reference_capacity"

    derived = float(median(capacity for _, capacity in diagnostic_capacities))
    if not math.isfinite(derived) or derived <= 0:
        raise ValueError(
            "first valid diagnostic median reference capacity must be finite and positive"
        )
    return derived, "first_valid_diagnostic_median"


def _observed_soh_trajectory(
    diagnostic_capacities: Sequence[tuple[int, float]],
    reference_capacity_ah: float,
) -> tuple[list[int], list[float]]:
    cycles = [cycle for cycle, _ in diagnostic_capacities]
    observed_soh = [capacity / reference_capacity_ah for _, capacity in diagnostic_capacities]
    if any(not math.isfinite(value) or value <= 0 or value > 1.5 for value in observed_soh):
        raise ValueError("derived observed SOH must remain within (0, 1.5]")
    return cycles, observed_soh


def execute_extract_early_cycle_features_tool(
    input_value: ExtractEarlyCycleFeaturesToolInput,
    *,
    resolver: VerifiedEarlyCycleBatchResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Create deterministic early-cycle evidence from a resolver-owned record batch."""

    validated_input = ExtractEarlyCycleFeaturesToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    batch = _resolve_verified_batch(resolver, validated_input.record_batch_id)
    feature_set = extract_early_cycle_features(batch.records, config=batch.feature_config)
    diagnostic_capacities = _diagnostic_capacity_by_cycle(batch.records)
    reference_capacity_ah, reference_capacity_method = _reference_capacity(
        batch.metadata,
        diagnostic_capacities,
    )
    observed_cycles, observed_soh = _observed_soh_trajectory(
        diagnostic_capacities,
        reference_capacity_ah,
    )
    if observed_cycles[-1] != batch.feature_config.cutoff_cycle:
        raise ValueError("last valid diagnostic observation must equal the feature cutoff cycle")

    # Preserve unavailable feature signals explicitly in ``feature_values``.
    # ``condition_features`` is a separate finite-only view for the fitted
    # predictor, so missing data is never fabricated as zero or another proxy.
    feature_values = dict(feature_set.values)
    condition_features = {
        "nominal_capacity_ah": batch.metadata.nominal_capacity_ah,
        **{
            name: value
            for name, value in feature_values.items()
            if value is not None and math.isfinite(value)
        },
    }
    artifact = {
        "record_batch_id": batch.record_batch_id,
        "dataset_id": batch.metadata.dataset_id,
        "cell_id": batch.metadata.cell_id,
        "cutoff_cycle": batch.feature_config.cutoff_cycle,
        "observed_cycles": observed_cycles,
        "observed_soh": observed_soh,
        "reference_capacity_ah": reference_capacity_ah,
        "reference_capacity_method": reference_capacity_method,
        "feature_values": feature_values,
        "condition_features": condition_features,
        "source_cycles": list(feature_set.source_cycles),
        "feature_warnings": list(feature_set.warnings),
        "source_manifest_hash": batch.source_manifest_hash,
        "feature_version": feature_set.feature_version,
        "split_version": batch.split_version,
        "data_version": batch.data_version,
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=EARLY_CYCLE_FEATURE_TOOL_VERSION,
        model_version=EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
        data_version=batch.data_version,
        feature_version=feature_set.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "artifact_type": EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
            "artifact": artifact,
        },
        uncertainty=None,
        warnings=list(feature_set.warnings),
        provenance=list(batch.provenance),
        created_at=_execution_timestamp(clock),
    )


def register_extract_early_cycle_features_tool(
    registry: ToolRegistry,
    *,
    resolver: VerifiedEarlyCycleBatchResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[ExtractEarlyCycleFeaturesToolInput]:
    """Register only in a service context that owns a trusted batch resolver."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            tool_version=EARLY_CYCLE_FEATURE_TOOL_VERSION,
            input_model=ExtractEarlyCycleFeaturesToolInput,
            executor=lambda input_value: execute_extract_early_cycle_features_tool(
                input_value,
                resolver=resolver,
                clock=clock,
            ),
        )
    )
