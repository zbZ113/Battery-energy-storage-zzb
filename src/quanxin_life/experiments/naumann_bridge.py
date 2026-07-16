"""Provenance-preserving bridge from reviewed Naumann conditions to GP inputs.

The bridge is deliberately one-to-one: it transforms an already reviewed
condition-level source measurement into an :class:`ExperimentObservation`
without loading source files, inferring condition fields from strings, or
aggregating measurements.  It is not a ToolResult factory; a governed tool
layer must wrap these transparent intermediate records later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.naumann_calendar import CalendarCapacityObservation
from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.experiments.gp_active import ExperimentObservation, OperatingCondition

NAUMANN_GP_BRIDGE_VERSION = "naumann-gp-bridge-v1"
NaumannConditionObservation: TypeAlias = CycleMatrixObservation | CalendarCapacityObservation


class TargetTransformMode(StrEnum):
    """Reviewed metric transformations allowed by the bridge."""

    DIRECT_METRIC_VALUE = "direct_metric_value"
    RELATIVE_CAPACITY_LOSS = "relative_capacity_loss"
    RELATIVE_CAPACITY_LOSS_RATE = "relative_capacity_loss_rate"
    RELATIVE_RESISTANCE_GROWTH = "relative_resistance_growth"
    RELATIVE_RESISTANCE_GROWTH_RATE = "relative_resistance_growth_rate"


class RateDenominatorBasis(StrEnum):
    """Explicit units allowed for a relative per-axis GP target."""

    EQUIVALENT_FULL_CYCLES = "equivalent_full_cycles"
    TIME_H = "time_h"


class ReferenceMetricSourceKind(StrEnum):
    """The two auditable ways a relative-transform baseline may be reviewed."""

    SOURCE_OBSERVATION = "source_observation"
    REVIEWED_CONFIGURATION = "reviewed_configuration"


class ReviewedReferenceMetric(ContractModel):
    """A provenance-bearing reference required by relative metric transforms."""

    metric_value: float = Field(gt=0, allow_inf_nan=False)
    source_kind: ReferenceMetricSourceKind
    source_observation_id: str | None = Field(default=None, min_length=1)
    source_sha256: Sha256 | None = None
    source_uri: str | None = Field(default=None, min_length=1)
    review_statement: str | None = Field(default=None, min_length=1)
    evidence_reference: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def reference_has_declared_evidence(self) -> ReviewedReferenceMetric:
        if self.source_kind is ReferenceMetricSourceKind.SOURCE_OBSERVATION:
            if self.source_observation_id is None:
                raise ValueError("source_observation_id is required for source observations")
            if self.source_sha256 is None:
                raise ValueError("source_sha256 is required for source observations")
            if self.source_uri is None:
                raise ValueError("source_uri is required for source observations")
        elif self.review_statement is None or self.evidence_reference is None:
            raise ValueError(
                "review_statement and evidence_reference are required for reviewed configuration"
            )
        return self


class MetricTargetTransform(ContractModel):
    """An explicit source-metric-to-GP-target transformation declaration."""

    source_metric_name: Literal["capacity_ah", "relative_capacity_ratio", "resistance_ohm"]
    target_name: str = Field(min_length=1)
    mode: TargetTransformMode
    reference_metric: ReviewedReferenceMetric | None = None
    rate_denominator_basis: RateDenominatorBasis | None = None

    @model_validator(mode="after")
    def transform_is_metric_compatible(self) -> MetricTargetTransform:
        capacity_modes = {
            TargetTransformMode.RELATIVE_CAPACITY_LOSS,
            TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
        }
        resistance_modes = {
            TargetTransformMode.RELATIVE_RESISTANCE_GROWTH,
            TargetTransformMode.RELATIVE_RESISTANCE_GROWTH_RATE,
        }
        if self.mode in capacity_modes and self.source_metric_name not in {
            "capacity_ah",
            "relative_capacity_ratio",
        }:
            raise ValueError(
                "capacity-loss transforms require a capacity-valued source metric"
            )
        if self.mode in resistance_modes and self.source_metric_name != "resistance_ohm":
            raise ValueError(
                "resistance-growth transforms require source_metric_name=resistance_ohm"
            )
        rate_modes = {
            TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
            TargetTransformMode.RELATIVE_RESISTANCE_GROWTH_RATE,
        }
        if self.mode is TargetTransformMode.DIRECT_METRIC_VALUE:
            if self.reference_metric is not None:
                raise ValueError("direct_metric_value must not declare reference_metric")
            if self.rate_denominator_basis is not None:
                raise ValueError("direct_metric_value must not declare rate_denominator_basis")
        elif self.reference_metric is None:
            raise ValueError("reference_metric is required for relative transforms")
        elif self.mode in rate_modes and self.rate_denominator_basis is None:
            raise ValueError("rate_denominator_basis is required for rate transforms")
        elif self.mode not in rate_modes and self.rate_denominator_basis is not None:
            raise ValueError("rate_denominator_basis is only valid for rate transforms")
        return self


class CalendarCyclingConditionMapping(ContractModel):
    """Manual declaration required before calendar data can enter cycling GP input."""

    condition_id: str = Field(min_length=1)
    dod: float = Field(gt=0, le=1, allow_inf_nan=False)
    charge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    discharge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    scientific_use_statement: str = Field(min_length=1)
    evidence_reference: str = Field(min_length=1)


class ReviewedExperimentResources(ContractModel):
    """Reviewed resource metadata required by the active-experiment interface.

    Naumann source adapters preserve condition measurements, not a lab booking
    ledger.  The active GP interface nevertheless requires a duration and an
    equipment cost for replay-aware acquisition.  Requiring both values and
    their review evidence here prevents the bridge from silently inventing
    placeholders.
    """

    duration_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost: float = Field(gt=0, allow_inf_nan=False)
    review_statement: str = Field(min_length=1)
    evidence_reference: str = Field(min_length=1)


class NaumannGpBridgeMapping(ContractModel):
    """Reviewed configuration; it never guesses values from source text."""

    mapping_version: str = Field(min_length=1)
    resources: ReviewedExperimentResources | None = None
    target_transform: MetricTargetTransform
    calendar_conditions: tuple[CalendarCyclingConditionMapping, ...] = ()

    @model_validator(mode="after")
    def calendar_condition_ids_are_unique(self) -> NaumannGpBridgeMapping:
        condition_ids = [item.condition_id for item in self.calendar_conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("calendar condition mappings must use unique condition_id values")
        return self


@dataclass(frozen=True)
class NaumannGpProvenance:
    """Source lineage carried with a bridged GP observation, not a ToolResult."""

    source_dataset_id: str
    source_observation_id: str
    source_condition_id: str
    source_sha256: Sha256
    source_file: str
    layout_version: str
    adapter_version: str
    bridge_version: str
    mapping_version: str
    scientific_use_statement: str | None
    evidence_reference: str | None
    resource_review_statement: str | None
    resource_evidence_reference: str | None
    reference_metric: ReviewedReferenceMetric | None


@dataclass(frozen=True)
class BridgedExperimentObservation:
    """One unaggregated GP input and the source lineage required to audit it."""

    experiment_observation: ExperimentObservation
    provenance: NaumannGpProvenance


def bridge_naumann_observations(
    observations: tuple[NaumannConditionObservation, ...],
    *,
    mapping: NaumannGpBridgeMapping,
) -> tuple[BridgedExperimentObservation, ...]:
    """Bridge reviewed Naumann condition observations one-to-one.

    Missing calendar cycling fields are accepted only through a per-condition
    mapping with an explicit scientific-use statement and evidence reference.
    The function intentionally has no aggregation, interpolation, default
    condition values or free-text parsing path.
    """

    if not observations:
        raise ValueError("at least one reviewed Naumann observation is required")
    mapping = NaumannGpBridgeMapping.model_validate(mapping.model_dump(mode="json"))
    calendar_mapping_by_id = {
        item.condition_id: item for item in mapping.calendar_conditions
    }
    bridged: list[BridgedExperimentObservation] = []
    seen_observation_ids: set[str] = set()
    for source_observation in observations:
        source_metric_name = _source_metric_name(source_observation)
        if source_metric_name != mapping.target_transform.source_metric_name:
            raise ValueError(
                "target transform source_metric_name does not match source observation metric"
            )
        observation_id = _source_observation_id(source_observation)
        if observation_id in seen_observation_ids:
            raise ValueError(
                "bridge input contains duplicate source observation identity: " + observation_id
            )
        seen_observation_ids.add(observation_id)
        condition, scientific_use_statement, evidence_reference = _operating_condition(
            source_observation,
            calendar_mapping_by_id,
        )
        source_metric_value = _source_metric_value(source_observation)
        observed_target = _transform_target(
            source_metric_value,
            _source_axis_value(source_observation),
            _source_axis_basis(source_observation),
            mapping.target_transform,
        )
        if not math.isfinite(observed_target):
            raise RuntimeError("reviewed metric transform produced a non-finite target")
        resources = mapping.resources
        experiment_observation = ExperimentObservation(
            observation_id=observation_id,
            condition=condition,
            target_name=mapping.target_transform.target_name,
            observed_target=observed_target,
            duration_hours=resources.duration_hours if resources is not None else None,
            equipment_cost=resources.equipment_cost if resources is not None else None,
        )
        bridged.append(
            BridgedExperimentObservation(
                experiment_observation=experiment_observation,
                provenance=NaumannGpProvenance(
                    source_dataset_id=source_observation.dataset_id,
                    source_observation_id=observation_id,
                    source_condition_id=source_observation.condition_id,
                    source_sha256=source_observation.source_sha256,
                    source_file=source_observation.source_file,
                    layout_version=source_observation.layout_version,
                    adapter_version=source_observation.adapter_version,
                    bridge_version=NAUMANN_GP_BRIDGE_VERSION,
                    mapping_version=mapping.mapping_version,
                    scientific_use_statement=scientific_use_statement,
                    evidence_reference=evidence_reference,
                    resource_review_statement=(
                        resources.review_statement if resources is not None else None
                    ),
                    resource_evidence_reference=(
                        resources.evidence_reference if resources is not None else None
                    ),
                    reference_metric=mapping.target_transform.reference_metric,
                ),
            )
        )
    return tuple(bridged)


def _source_metric_name(observation: NaumannConditionObservation) -> str:
    if isinstance(observation, CycleMatrixObservation):
        return observation.metric_name
    return "capacity_ah"


def _source_metric_value(observation: NaumannConditionObservation) -> float:
    if isinstance(observation, CycleMatrixObservation):
        return observation.metric_value
    return observation.capacity_ah


def _source_axis_value(observation: NaumannConditionObservation) -> float:
    if isinstance(observation, CycleMatrixObservation):
        return observation.observation_value
    return observation.storage_time_h


def _source_observation_id(observation: NaumannConditionObservation) -> str:
    """Create a deterministic identity from already explicit source fields."""

    axis_name: str
    axis_value: float
    if isinstance(observation, CycleMatrixObservation):
        axis_name = observation.observation_axis
        axis_value = observation.observation_value
    else:
        axis_name = "storage_time_h"
        axis_value = observation.storage_time_h
    identity_hash = sha256_canonical(
        {
            "dataset_id": observation.dataset_id,
            "condition_id": observation.condition_id,
            "source_sha256": observation.source_sha256,
            "layout_version": observation.layout_version,
            "adapter_version": observation.adapter_version,
            "axis_name": axis_name,
            "axis_value": axis_value,
            "metric_name": _source_metric_name(observation),
            "metric_value": _source_metric_value(observation),
        }
    )
    return f"{observation.dataset_id}:{observation.condition_id}:{identity_hash[:16]}"


def _source_axis_basis(observation: NaumannConditionObservation) -> RateDenominatorBasis:
    if isinstance(observation, CalendarCapacityObservation):
        return RateDenominatorBasis.TIME_H
    if observation.observation_axis == "equivalent_full_cycles":
        return RateDenominatorBasis.EQUIVALENT_FULL_CYCLES
    return RateDenominatorBasis.TIME_H


def _operating_condition(
    observation: NaumannConditionObservation,
    calendar_mapping_by_id: dict[str, CalendarCyclingConditionMapping],
) -> tuple[OperatingCondition, str | None, str | None]:
    if isinstance(observation, CycleMatrixObservation):
        return (
            OperatingCondition(
                temperature_c=observation.temperature_c,
                mean_soc=observation.mean_soc,
                dod=observation.dod,
                charge_c_rate=observation.charge_c_rate,
                discharge_c_rate=observation.discharge_c_rate,
            ),
            None,
            None,
        )
    calendar_mapping = calendar_mapping_by_id.get(observation.condition_id)
    if calendar_mapping is None:
        raise ValueError(
            "calendar condition mapping with reviewed cycling fields and scientific use is required"
        )
    return (
        OperatingCondition(
            temperature_c=observation.temperature_c,
            mean_soc=observation.mean_soc,
            dod=calendar_mapping.dod,
            charge_c_rate=calendar_mapping.charge_c_rate,
            discharge_c_rate=calendar_mapping.discharge_c_rate,
        ),
        calendar_mapping.scientific_use_statement,
        calendar_mapping.evidence_reference,
    )


def _transform_target(
    metric_value: float,
    axis_value: float,
    source_axis_basis: RateDenominatorBasis,
    transform: MetricTargetTransform,
) -> float:
    if transform.mode is TargetTransformMode.DIRECT_METRIC_VALUE:
        return metric_value
    reference_metric = transform.reference_metric
    reference = reference_metric.metric_value if reference_metric is not None else None
    if reference is None:  # Defensive; Pydantic mapping validation already guards this.
        raise RuntimeError("relative target transform is missing reference_metric_value")
    if transform.mode in {
        TargetTransformMode.RELATIVE_CAPACITY_LOSS,
        TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
    }:
        relative_change = (reference - metric_value) / reference
    else:
        relative_change = (metric_value - reference) / reference
    if transform.mode in {
        TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
        TargetTransformMode.RELATIVE_RESISTANCE_GROWTH_RATE,
    }:
        if transform.rate_denominator_basis is not source_axis_basis:
            raise ValueError(
                "rate denominator basis does not match the reviewed source observation axis"
            )
        if axis_value <= 0:
            raise ValueError("rate transform requires a strictly positive source observation axis")
        return relative_change / axis_value
    return relative_change
