"""Per-cell and trajectory prediction evidence contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from quanxin_life.core.schemas import ContractModel


class CellPredictionRecord(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    seed: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    cutoff: int | None = Field(default=None, ge=0)
    target: str = Field(min_length=1)
    true_value: float | None = None
    predicted_value: float | None = None
    error: float | None = None
    right_censored: bool = False
    ood: bool = False
    evidence: str = Field(min_length=1)


class TrajectoryPredictionRecord(CellPredictionRecord):
    cycle: int = Field(ge=0)
    timestamp_utc: datetime | None = None
    horizon: int = Field(ge=0)
    true_soh: float | None = None
    predicted_soh: float | None = None
    lower: float | None = None
    upper: float | None = None
    interval_method: str = "NOT_AVAILABLE"


def validate_test_cohort(records: list[CellPredictionRecord]) -> None:
    if not records:
        raise ValueError("prediction cohort cannot be empty")
    keys = [(record.seed, record.cell_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("prediction cohort contains duplicate seed/cell rows")
    by_seed = {
        seed: {item.cell_id for item in records if item.seed == seed}
        for seed in {item.seed for item in records}
    }
    reference = next(iter(by_seed.values()))
    if any(cells != reference for cells in by_seed.values()):
        raise ValueError("seed cell cohorts must be identical")


__all__ = ["CellPredictionRecord", "TrajectoryPredictionRecord", "validate_test_cohort"]
