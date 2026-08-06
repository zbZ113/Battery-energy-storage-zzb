"""Physical, provenance-bound records for canonical battery datasets."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core import CanonicalTableType, CanonicalUnit, DataQualityStatus
from quanxin_life.core.schemas import ContractModel, Sha256


class CanonicalNumericValue(ContractModel):
    """One numeric observation with explicit source, unit, and quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    table_type: CanonicalTableType
    quantity_name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: CanonicalUnit
    quality_status: DataQualityStatus
    source_file: str = Field(min_length=1)
    source_sha256: Sha256
    adapter_version: str = Field(min_length=1)

    @field_validator("source_file")
    @classmethod
    def source_file_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("source_file must be a confined portable path")
        return value


__all__ = ["CanonicalNumericValue"]
