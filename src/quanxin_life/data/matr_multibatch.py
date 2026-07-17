"""Contracts and validation for governed multi-batch MATR preparation."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel


class MatrTrajectoryExclusion(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    observed_cycle_count: int = Field(gt=0)
    reason: Literal["INSUFFICIENT_REAL_TRAJECTORY_500"] = (
        "INSUFFICIENT_REAL_TRAJECTORY_500"
    )


class MatrTrajectoryEligibilityAudit(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-trajectory-eligibility-v1"] = (
        "matr-trajectory-eligibility-v1"
    )
    batch_index: int = Field(ge=1)
    horizon_cycle: int = Field(ge=5)
    eligible_cell_ids: tuple[str, ...]
    excluded: tuple[MatrTrajectoryExclusion, ...]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def inventory_is_closed(self) -> MatrTrajectoryEligibilityAudit:
        eligible = set(self.eligible_cell_ids)
        excluded = {item.cell_id for item in self.excluded}
        if len(eligible) != len(self.eligible_cell_ids) or len(excluded) != len(self.excluded):
            raise ValueError("trajectory eligibility cell identifiers must be unique")
        if eligible & excluded:
            raise ValueError("trajectory eligibility inventories must be disjoint")
        if not eligible:
            raise ValueError("trajectory eligibility requires at least one eligible cell")
        return self


def audit_matr_supervision_eligibility(
    *,
    batch_index: int,
    observed_cycle_counts: dict[str, int],
    horizon_cycle: int,
    created_at: datetime,
) -> MatrTrajectoryEligibilityAudit:
    """Separate complete real trajectories from shorter cells without imputation."""

    if horizon_cycle < 5:
        raise ValueError("trajectory horizon must include at least cycles 1 through 5")
    if not observed_cycle_counts:
        raise ValueError("trajectory eligibility requires observed cycle counts")
    pattern = re.compile(rf"MATR_b{batch_index}c\d+\Z")
    if any(pattern.fullmatch(cell_id) is None for cell_id in observed_cycle_counts):
        raise ValueError("trajectory eligibility cell_id does not match batch_index")
    if any(count <= 0 for count in observed_cycle_counts.values()):
        raise ValueError("observed cycle counts must be positive")
    eligible = tuple(
        sorted(
            cell_id
            for cell_id, count in observed_cycle_counts.items()
            if count > horizon_cycle
        )
    )
    excluded = tuple(
        MatrTrajectoryExclusion(
            cell_id=cell_id,
            observed_cycle_count=count,
        )
        for cell_id, count in sorted(observed_cycle_counts.items())
        if count <= horizon_cycle
    )
    return MatrTrajectoryEligibilityAudit(
        batch_index=batch_index,
        horizon_cycle=horizon_cycle,
        eligible_cell_ids=eligible,
        excluded=excluded,
        created_at=created_at,
    )
