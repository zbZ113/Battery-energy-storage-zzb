"""Reviewed outer-archive index for the 40/280 Ah multi-DoD dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.zip_safety import audit_zip_archive

_MEMBER = re.compile(
    r"^(?P<vendor>CATL|EVE)/(?P=vendor)-0\.5C-(?P<dod>20|60|100)% "
    r"Depth of Discharge/(?P<cell>\d+)\.zip$"
)


class DodArchiveLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    vendor: Literal["CATL", "EVE"]
    archive_sha256: Sha256
    expected_members: tuple[str, ...] = Field(min_length=1)
    task_role: Literal["calibration", "external_test"] = "external_test"


class DodCellSource(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["LFP_280AH_DOD"] = "LFP_280AH_DOD"
    vendor: Literal["CATL", "EVE"]
    cell_id: str = Field(min_length=1)
    dod_fraction: float = Field(gt=0, le=1)
    c_rate: float = 0.5
    source_member: str = Field(min_length=1)
    archive_sha256: Sha256
    task_role: Literal["calibration", "external_test"]


def load_dod_sources(path: Path, *, layout: DodArchiveLayout) -> tuple[DodCellSource, ...]:
    validated = DodArchiveLayout.model_validate(layout.model_dump(mode="json"))
    infos = audit_zip_archive(
        path,
        archive_sha256=validated.archive_sha256,
        expected_members=validated.expected_members,
        allowed_suffixes=frozenset({".zip"}),
    )
    records: list[DodCellSource] = []
    for info in infos:
        match = _MEMBER.fullmatch(info.filename)
        if match is None or match.group("vendor") != validated.vendor:
            raise ValueError("DoD ZIP member does not match the reviewed layout")
        records.append(
            DodCellSource(
                vendor=validated.vendor,
                cell_id=f"{validated.vendor}-{match.group('cell')}",
                dod_fraction=int(match.group("dod")) / 100.0,
                source_member=info.filename,
                archive_sha256=validated.archive_sha256,
                task_role=validated.task_role,
            )
        )
    return tuple(records)


__all__ = ["DodArchiveLayout", "DodCellSource", "load_dod_sources"]
