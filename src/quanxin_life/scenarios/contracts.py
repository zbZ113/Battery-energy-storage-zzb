"""Versioned operating-scenario contracts for numerical lifetime tools."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord
from quanxin_life.core.schemas import ContractModel


class ScenarioCellDescriptor(ContractModel):
    """Cell identity declared by a caller and matched to trusted server context."""

    chemistry: str = Field(min_length=1, max_length=100)
    nominal_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    cell_format: Literal["cylindrical", "prismatic"]

    @field_validator("chemistry")
    @classmethod
    def chemistry_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("chemistry must not be blank")
        return normalized


class VerifiedScenarioContext(ContractModel):
    """Server-resolved cell metadata and provenance for scenario execution."""

    scenario_context_id: str = Field(min_length=1, max_length=200)
    cell: ScenarioCellDescriptor
    trusted_reference_use: bool = False
    data_version: str = Field(min_length=1, max_length=200)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("scenario_context_id", "data_version")
    @classmethod
    def context_identifiers_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scenario context identifiers must not be blank")
        return normalized


class ScenarioSegment(ContractModel):
    """One contiguous, constant-condition stage in a lifetime scenario."""

    segment_id: str = Field(min_length=1, max_length=100)
    start_year: int = Field(ge=0, le=24)
    end_year: int = Field(ge=1, le=25)
    temperature_c: float = Field(allow_inf_nan=False)
    charge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    discharge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    soc_lower_bound: float = Field(ge=0, lt=1, allow_inf_nan=False)
    soc_upper_bound: float = Field(gt=0, le=1, allow_inf_nan=False)
    dod: float = Field(gt=0, le=1, allow_inf_nan=False)
    equivalent_full_cycles_per_year: float = Field(ge=0, allow_inf_nan=False)
    rest_duration_hours: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("segment_id")
    @classmethod
    def segment_id_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("segment_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def segment_bounds_are_consistent(self) -> Self:
        if self.start_year >= self.end_year:
            raise ValueError("segment start_year must be before end_year")
        if self.soc_lower_bound >= self.soc_upper_bound:
            raise ValueError("SOC lower bound must be below upper bound")
        declared_window = self.soc_upper_bound - self.soc_lower_bound
        if not math.isclose(self.dod, declared_window, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("DoD must equal the declared SOC window")
        return self


class OperationScenario(ContractModel):
    """A complete 1--25 year scenario with no temporal gaps or overlaps."""

    scenario_id: str = Field(min_length=1, max_length=100)
    scenario_version: str = Field(min_length=1, max_length=100)
    horizon_years: int = Field(ge=1, le=25)
    eol_threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    segments: tuple[ScenarioSegment, ...] = Field(min_length=1)

    @field_validator("scenario_id", "scenario_version")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scenario identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def segments_are_contiguous_and_complete(self) -> Self:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario segment_id values must be unique")
        if self.segments[0].start_year != 0:
            raise ValueError("scenario segments must start at year zero")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.end_year != current.start_year:
                raise ValueError("scenario segments must be contiguous without gaps or overlaps")
        if self.segments[-1].end_year != self.horizon_years:
            raise ValueError("scenario segments must cover the full horizon")
        return self


__all__ = [
    "OperationScenario",
    "ScenarioCellDescriptor",
    "ScenarioSegment",
    "VerifiedScenarioContext",
]
