"""Cell/round/phase index for the realistic forklift ageing dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.zip_safety import audit_zip_archive

_MEMBER = re.compile(r"^.+/Cell(?P<cell>[123])/Round(?P<round>\d{2})/(?P<kind>RPT|Ageing)\.csv$")


class ForkliftLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    archive_sha256: Sha256
    expected_members: tuple[str, ...] = ()
    expected_member_count: int | None = Field(default=None, gt=0)
    member_inventory_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def inventory_is_bound(self) -> ForkliftLayout:
        if not self.expected_members and (
            self.expected_member_count is None or self.member_inventory_sha256 is None
        ):
            raise ValueError("forklift layout requires exact members or inventory SHA")
        return self


class ForkliftMeasurementSource(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["LFP_180AH_FORKLIFT"] = "LFP_180AH_FORKLIFT"
    cell_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    measurement_type: Literal["RPT", "AGEING"]
    source_member: str = Field(min_length=1)
    archive_sha256: Sha256
    split_group: str = Field(min_length=1)


def load_forklift_sources(
    path: Path,
    *,
    layout: ForkliftLayout,
) -> tuple[ForkliftMeasurementSource, ...]:
    validated = ForkliftLayout.model_validate(layout.model_dump(mode="json"))
    infos = audit_zip_archive(
        path,
        archive_sha256=validated.archive_sha256,
        expected_members=(validated.expected_members or None),
        allowed_suffixes=frozenset({".csv"}),
    )
    names = tuple(sorted(info.filename for info in infos))
    if (
        validated.expected_member_count is not None
        and len(names) != validated.expected_member_count
    ):
        raise ValueError("forklift member count differs from reviewed inventory")
    if (
        validated.member_inventory_sha256 is not None
        and sha256_canonical(names) != validated.member_inventory_sha256
    ):
        raise ValueError("forklift member inventory SHA mismatch")
    records: list[ForkliftMeasurementSource] = []
    for info in infos:
        match = _MEMBER.fullmatch(info.filename)
        if match is None:
            raise ValueError("forklift member path does not match reviewed cell/round layout")
        cell_id = f"FORKLIFT-CELL-{match.group('cell')}"
        records.append(
            ForkliftMeasurementSource(
                cell_id=cell_id,
                round_index=int(match.group("round")),
                measurement_type=("RPT" if match.group("kind") == "RPT" else "AGEING"),
                source_member=info.filename,
                archive_sha256=validated.archive_sha256,
                split_group=cell_id,
            )
        )
    return tuple(records)


__all__ = ["ForkliftLayout", "ForkliftMeasurementSource", "load_forklift_sources"]
