"""Contracts and validation for governed multi-batch MATR preparation."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import SplitManifest


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


class MatrBatchArtifactReference(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_index: int = Field(ge=1, le=3)
    batch_date: date
    raw_relative_path: str = Field(min_length=1)
    raw_sha256: Sha256
    processed_root: str = Field(min_length=1)
    conversion_report: str = Field(min_length=1)
    conversion_report_sha256: Sha256
    split_manifest: str = Field(min_length=1)
    split_manifest_sha256: Sha256
    supervision_root: str = Field(min_length=1)
    supervision_report: str = Field(min_length=1)
    supervision_report_sha256: Sha256
    eligibility_report: str = Field(min_length=1)
    eligibility_report_sha256: Sha256
    cell_count: int = Field(gt=0)
    scalar_label_count: int = Field(ge=0)
    hybrid_eligible_count: int = Field(ge=0)
    hybrid_excluded_count: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> MatrBatchArtifactReference:
        if self.scalar_label_count > self.cell_count:
            raise ValueError("scalar label count cannot exceed batch cell count")
        if self.hybrid_eligible_count + self.hybrid_excluded_count != self.cell_count:
            raise ValueError("Hybrid eligibility counts must cover the batch")
        return self


class MatrThreeBatchManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-three-batch-manifest-v1"] = (
        "matr-three-batch-manifest-v1"
    )
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    combined_split_manifest: str = Field(min_length=1)
    combined_split_sha256: Sha256
    batches: tuple[MatrBatchArtifactReference, ...] = Field(min_length=3, max_length=3)
    total_cell_count: int = Field(gt=0)
    scalar_label_count: int = Field(gt=0)
    hybrid_eligible_count: int = Field(gt=0)
    hybrid_excluded_count: int = Field(ge=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def manifest_created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def components_and_counts_are_consistent(self) -> MatrThreeBatchManifest:
        expected_dates = {
            1: date(2017, 5, 12),
            2: date(2017, 6, 30),
            3: date(2018, 4, 12),
        }
        if [batch.batch_index for batch in self.batches] != [1, 2, 3]:
            raise ValueError("three-batch manifest components must be ordered 1, 2, 3")
        if any(
            batch.batch_date != expected_dates[batch.batch_index]
            for batch in self.batches
        ):
            raise ValueError("three-batch manifest contains an unexpected batch date")
        if self.total_cell_count != sum(batch.cell_count for batch in self.batches):
            raise ValueError("total cell count does not match batch components")
        if self.scalar_label_count != sum(
            batch.scalar_label_count for batch in self.batches
        ):
            raise ValueError("scalar label count does not match batch components")
        if self.hybrid_eligible_count != sum(
            batch.hybrid_eligible_count for batch in self.batches
        ):
            raise ValueError("Hybrid eligible count does not match batch components")
        if self.hybrid_excluded_count != sum(
            batch.hybrid_excluded_count for batch in self.batches
        ):
            raise ValueError("Hybrid excluded count does not match batch components")
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


def combine_matr_batch_splits(
    splits: tuple[SplitManifest, ...],
) -> SplitManifest:
    """Merge three batch-local splits without changing any cell assignment."""

    if len(splits) != 3:
        raise ValueError("three-batch MATR split requires exactly three batch splits")
    if any(split.dataset_id != "MATR" for split in splits):
        raise ValueError("three-batch split requires MATR component splits")
    if len({split.seed for split in splits}) != 1:
        raise ValueError("three-batch split seeds must agree")
    indexed: dict[int, SplitManifest] = {}
    for split in splits:
        if not split.all_cells:
            raise ValueError("component MATR split must contain cells")
        batch_indices: set[int] = set()
        for cell_id in split.all_cells:
            match = re.fullmatch(r"MATR_b(?P<batch>\d+)c\d+", cell_id)
            if match is None:
                raise ValueError("component MATR split contains an invalid cell_id")
            batch_indices.add(int(match.group("batch")))
        if len(batch_indices) != 1:
            raise ValueError("component MATR split mixes batch identifiers")
        batch_index = next(iter(batch_indices))
        if batch_index in indexed:
            raise ValueError("duplicate MATR batch split")
        if any(
            not partition
            for partition in (
                split.train,
                split.validation,
                split.calibration,
                split.test,
            )
        ):
            raise ValueError("every component split partition must be nonempty")
        indexed[batch_index] = split
    if set(indexed) != {1, 2, 3}:
        raise ValueError("three-batch split requires batch indices 1, 2 and 3")
    ordered = tuple(indexed[index] for index in (1, 2, 3))
    return SplitManifest(
        dataset_id="MATR",
        seed=ordered[0].seed,
        train=tuple(cell for split in ordered for cell in split.train),
        validation=tuple(cell for split in ordered for cell in split.validation),
        calibration=tuple(cell for split in ordered for cell in split.calibration),
        test=tuple(cell for split in ordered for cell in split.test),
    )
