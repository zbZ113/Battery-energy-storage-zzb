"""Canonical condition-level bundle mapping for reviewed Naumann observations."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import (
    CanonicalTableType,
    CanonicalUnit,
    DataQualityStatus,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.adapters.naumann_calendar import CalendarCapacityObservation
from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.data.canonical import CanonicalNumericValue
from quanxin_life.data.processing import DatasetBuildSpec, RawDatasetFile

NaumannObservation = CycleMatrixObservation | CalendarCapacityObservation


class NaumannConditionBundle(ContractModel):
    """Canonical values that remain explicitly condition-level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["NAUMANN_CYCLE", "NAUMANN_CALENDAR"]
    dataset_version: str = Field(min_length=1)
    identity_level: Literal["condition"] = "condition"
    cell_level_split_supported: Literal[False] = False
    values: tuple[CanonicalNumericValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def values_share_dataset_context(self) -> NaumannConditionBundle:
        expected_prefix = f"{self.dataset_id}:"
        if any(not value.record_id.startswith(expected_prefix) for value in self.values):
            raise ValueError("Naumann canonical values must share the bundle dataset")
        return self


def build_naumann_condition_bundle(
    *,
    dataset_version: str,
    observations: tuple[NaumannObservation, ...],
) -> NaumannConditionBundle:
    """Map reviewed direct observations without inventing cell or lifetime labels."""

    if not observations:
        raise ValueError("Naumann condition bundle requires reviewed observations")
    dataset_ids = {item.dataset_id for item in observations}
    if len(dataset_ids) != 1:
        raise ValueError("Naumann condition bundle cannot mix datasets")
    values: list[CanonicalNumericValue] = []
    for index, observation in enumerate(observations):
        prefix = f"{observation.dataset_id}:{observation.condition_id}:{index}"
        pairs: tuple[tuple[str, float, CanonicalUnit], ...]
        if isinstance(observation, CycleMatrixObservation):
            axis_unit = (
                CanonicalUnit.FEC
                if observation.observation_axis == "equivalent_full_cycles"
                else CanonicalUnit.HOUR
            )
            metric_unit = {
                "capacity_ah": CanonicalUnit.AMPERE_HOUR,
                "relative_capacity_ratio": CanonicalUnit.RATIO,
                "resistance_ohm": CanonicalUnit.OHM,
            }[observation.metric_name]
            pairs = (
                (observation.observation_axis, observation.observation_value, axis_unit),
                (observation.metric_name, observation.metric_value, metric_unit),
                ("temperature", observation.temperature_c, CanonicalUnit.CELSIUS),
                ("mean_soc", observation.mean_soc, CanonicalUnit.RATIO),
                ("dod", observation.dod, CanonicalUnit.RATIO),
                ("charge_c_rate", observation.charge_c_rate, CanonicalUnit.C_RATE),
                ("discharge_c_rate", observation.discharge_c_rate, CanonicalUnit.C_RATE),
            )
        else:
            pairs = (
                ("storage_time", observation.storage_time_h, CanonicalUnit.HOUR),
                ("capacity", observation.capacity_ah, CanonicalUnit.AMPERE_HOUR),
                ("temperature", observation.temperature_c, CanonicalUnit.CELSIUS),
                ("mean_soc", observation.mean_soc, CanonicalUnit.RATIO),
            )
        values.extend(
            CanonicalNumericValue(
                record_id=f"{prefix}:{quantity_name}",
                quantity_name=quantity_name,
                value=value,
                unit=unit,
                table_type=CanonicalTableType.CONDITION_OBSERVATIONS,
                quality_status=DataQualityStatus.VALID,
                source_file=observation.source_file,
                source_sha256=observation.source_sha256,
                adapter_version=observation.adapter_version,
            )
            for quantity_name, value, unit in pairs
        )
    return NaumannConditionBundle(
        dataset_id=next(iter(dataset_ids)),
        dataset_version=dataset_version,
        values=tuple(values),
    )


def to_naumann_build_spec(
    bundle: NaumannConditionBundle,
    *,
    raw_files: tuple[RawDatasetFile, ...],
) -> DatasetBuildSpec:
    """Bind a reviewed Naumann bundle to immutable raw artifacts for publication."""

    validated = NaumannConditionBundle.model_validate(bundle.model_dump(mode="json"))
    return DatasetBuildSpec(
        dataset_id=validated.dataset_id,
        dataset_version=validated.dataset_version,
        adapter_version="naumann-condition-canonical-v1",
        raw_files=raw_files,
        values=validated.values,
    )


__all__ = [
    "NaumannConditionBundle",
    "build_naumann_condition_bundle",
    "to_naumann_build_spec",
]
