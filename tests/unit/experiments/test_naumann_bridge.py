"""Contracts for provenance-preserving Naumann-to-GP input bridging."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quanxin_life.data.adapters.naumann_calendar import CalendarCapacityObservation
from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.experiments.naumann_bridge import (
    CalendarCyclingConditionMapping,
    MetricTargetTransform,
    NaumannGpBridgeMapping,
    RateDenominatorBasis,
    ReferenceMetricSourceKind,
    ReviewedExperimentResources,
    ReviewedReferenceMetric,
    TargetTransformMode,
    bridge_naumann_observations,
)

_SHA256 = "a" * 64


def _cycle_observation(
    *,
    observation_value: float = 100.0,
    metric_value: float = 9.0,
) -> CycleMatrixObservation:
    return CycleMatrixObservation(
        condition_id="cycle-condition-01",
        observation_axis="equivalent_full_cycles",
        observation_value=observation_value,
        temperature_c=25.0,
        mean_soc=0.5,
        dod=0.6,
        charge_c_rate=0.5,
        discharge_c_rate=0.5,
        metric_name="capacity_ah",
        metric_value=metric_value,
        source_file="cycle/source.mat",
        source_sha256=_SHA256,
        layout_version="cycle-layout-v1",
    )


def _calendar_observation() -> CalendarCapacityObservation:
    return CalendarCapacityObservation(
        condition_id="calendar-condition-01",
        storage_time_h=100.0,
        temperature_c=35.0,
        mean_soc=0.6,
        capacity_ah=9.0,
        source_file="calendar/source.xlsx",
        source_sha256=_SHA256,
        layout_version="calendar-layout-v1",
    )


def _direct_capacity_mapping() -> NaumannGpBridgeMapping:
    return NaumannGpBridgeMapping(
        mapping_version="naumann-gp-map-v1",
        resources=ReviewedExperimentResources(
            duration_hours=24.0,
            equipment_cost=3.5,
            review_statement="Reviewed equipment-time allocation for the replay cohort.",
            evidence_reference="experiment-resource-protocol-v1",
        ),
        target_transform=MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="capacity_ah",
            mode=TargetTransformMode.DIRECT_METRIC_VALUE,
        ),
    )


def _reviewed_capacity_reference() -> ReviewedReferenceMetric:
    return ReviewedReferenceMetric(
        metric_value=10.0,
        source_kind=ReferenceMetricSourceKind.REVIEWED_CONFIGURATION,
        review_statement="Reviewed baseline capacity for the Naumann replay protocol.",
        evidence_reference="naumann-reference-metric-protocol-v1",
    )


def test_bridge_mapping_can_explicitly_omit_unavailable_resource_metadata() -> None:
    mapping = NaumannGpBridgeMapping(
        mapping_version="naumann-gp-map-v1",
        resources=None,
        target_transform=MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="capacity_ah",
            mode=TargetTransformMode.DIRECT_METRIC_VALUE,
        ),
    )

    assert mapping.resources is None


def test_cycle_observation_bridges_one_to_one_with_source_provenance() -> None:
    observation = _cycle_observation()

    bridged = bridge_naumann_observations((observation,), mapping=_direct_capacity_mapping())

    assert len(bridged) == 1
    result = bridged[0]
    assert result.experiment_observation.condition.temperature_c == 25.0
    assert result.experiment_observation.condition.dod == 0.6
    assert result.experiment_observation.observed_target == 9.0
    assert result.experiment_observation.target_name == "capacity_ah"
    assert result.experiment_observation.duration_hours == 24.0
    assert result.experiment_observation.equipment_cost == 3.5
    assert result.provenance.source_sha256 == _SHA256
    assert result.provenance.layout_version == "cycle-layout-v1"
    assert result.provenance.source_condition_id == "cycle-condition-01"
    assert result.provenance.source_observation_id == result.experiment_observation.observation_id


def test_bridge_preserves_relative_capacity_ratio_without_claiming_ah() -> None:
    observation = _cycle_observation(metric_value=0.91).model_copy(
        update={"metric_name": "relative_capacity_ratio"}
    )
    mapping = NaumannGpBridgeMapping(
        mapping_version="naumann-relative-capacity-map-v1",
        resources=None,
        target_transform=MetricTargetTransform(
            source_metric_name="relative_capacity_ratio",
            target_name="relative_capacity_ratio",
            mode=TargetTransformMode.DIRECT_METRIC_VALUE,
        ),
    )

    result = bridge_naumann_observations((observation,), mapping=mapping)[0]

    assert result.experiment_observation.observed_target == 0.91
    assert result.experiment_observation.duration_hours is None
    assert result.experiment_observation.equipment_cost is None
    assert result.provenance.resource_review_statement is None
    assert result.provenance.resource_evidence_reference is None


def test_bridge_never_aggregates_multiple_source_points() -> None:
    first = _cycle_observation(observation_value=100.0)
    second = _cycle_observation(observation_value=200.0)

    bridged = bridge_naumann_observations((first, second), mapping=_direct_capacity_mapping())

    assert len(bridged) == 2
    assert [item.experiment_observation.observation_id for item in bridged] == [
        item.provenance.source_observation_id for item in bridged
    ]
    assert len({item.provenance.source_observation_id for item in bridged}) == 2


def test_source_identity_preserves_distinct_metric_measurements_at_same_axis() -> None:
    first = _cycle_observation(metric_value=9.0)
    second = _cycle_observation(metric_value=8.5)

    bridged = bridge_naumann_observations((first, second), mapping=_direct_capacity_mapping())

    assert len(bridged) == 2
    assert len({item.provenance.source_observation_id for item in bridged}) == 2


def test_calendar_observation_requires_reviewed_cycling_field_declaration() -> None:
    with pytest.raises(ValueError, match="calendar condition mapping"):
        bridge_naumann_observations(
            (_calendar_observation(),), mapping=_direct_capacity_mapping()
        )


def test_calendar_mapping_supplies_explicit_cycling_fields_and_target_transform() -> None:
    mapping = NaumannGpBridgeMapping(
        mapping_version="naumann-gp-map-v1",
        resources=ReviewedExperimentResources(
            duration_hours=48.0,
            equipment_cost=5.0,
            review_statement="Reviewed calendar comparison resource allocation.",
            evidence_reference="calendar-resource-protocol-v1",
        ),
        target_transform=MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="relative_capacity_loss_rate",
            mode=TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
            reference_metric=_reviewed_capacity_reference(),
            rate_denominator_basis=RateDenominatorBasis.TIME_H,
        ),
        calendar_conditions=(
            CalendarCyclingConditionMapping(
                condition_id="calendar-condition-01",
                dod=0.4,
                charge_c_rate=0.25,
                discharge_c_rate=0.25,
                scientific_use_statement="Reviewed calendar-to-GP comparison condition.",
                evidence_reference="internal-reviewed-protocol-v1",
            ),
        ),
    )

    result = bridge_naumann_observations((_calendar_observation(),), mapping=mapping)[0]

    assert result.experiment_observation.condition.dod == 0.4
    assert result.experiment_observation.condition.charge_c_rate == 0.25
    assert result.experiment_observation.condition.discharge_c_rate == 0.25
    assert result.experiment_observation.observed_target == pytest.approx(0.001)
    assert result.provenance.scientific_use_statement == (
        "Reviewed calendar-to-GP comparison condition."
    )


def test_metric_transform_rejects_missing_explicit_reference_for_loss_rate() -> None:
    with pytest.raises(ValidationError, match="reference_metric"):
        MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="relative_capacity_loss_rate",
            mode=TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
            rate_denominator_basis=RateDenominatorBasis.TIME_H,
        )


def test_relative_reference_requires_auditable_source_or_reviewed_evidence() -> None:
    with pytest.raises(ValidationError, match="source_uri"):
        ReviewedReferenceMetric(
            metric_value=10.0,
            source_kind=ReferenceMetricSourceKind.SOURCE_OBSERVATION,
            source_observation_id="NAUMANN_CYCLE:condition:source",
            source_sha256=_SHA256,
        )


def test_rate_transform_rejects_cycle_axis_with_incompatible_declared_basis() -> None:
    mapping = NaumannGpBridgeMapping(
        mapping_version="naumann-gp-map-v1",
        resources=ReviewedExperimentResources(
            duration_hours=24.0,
            equipment_cost=3.5,
            review_statement="Reviewed equipment-time allocation for the replay cohort.",
            evidence_reference="experiment-resource-protocol-v1",
        ),
        target_transform=MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="relative_capacity_loss_rate_per_h",
            mode=TargetTransformMode.RELATIVE_CAPACITY_LOSS_RATE,
            reference_metric=_reviewed_capacity_reference(),
            rate_denominator_basis=RateDenominatorBasis.TIME_H,
        ),
    )

    with pytest.raises(ValueError, match="rate denominator basis"):
        bridge_naumann_observations((_cycle_observation(),), mapping=mapping)
