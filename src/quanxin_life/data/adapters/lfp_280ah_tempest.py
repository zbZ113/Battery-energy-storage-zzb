"""Single-cell identity index for the TEMPEST 280 Ah dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.zip_safety import audit_zip_archive


class TempestArchive(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["aging", "characterization"]
    archive_sha256: Sha256
    expected_members: tuple[str, ...] = Field(min_length=1)


class TempestLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    archives: tuple[TempestArchive, TempestArchive]

    @model_validator(mode="after")
    def roles_are_complete(self) -> TempestLayout:
        if {item.role for item in self.archives} != {"aging", "characterization"}:
            raise ValueError("TEMPEST layout requires aging and characterization archives")
        return self


class TempestDatasetIndex(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["LFP_280AH_TEMPEST"] = "LFP_280AH_TEMPEST"
    cell_id: str = Field(min_length=1)
    aging_members: tuple[str, ...] = Field(min_length=1)
    characterization_members: tuple[str, ...] = Field(min_length=1)
    temperatures_c: tuple[int, ...] = Field(min_length=1)
    split_policy: Literal["single_cell_no_temperature_split"] = (
        "single_cell_no_temperature_split"
    )


def load_tempest_index(
    paths: tuple[Path, Path],
    *,
    layout: TempestLayout,
) -> TempestDatasetIndex:
    validated = TempestLayout.model_validate(layout.model_dump(mode="json"))
    by_role: dict[str, tuple[str, ...]] = {}
    for path, archive_layout in zip(paths, validated.archives, strict=True):
        infos = audit_zip_archive(
            path,
            archive_sha256=archive_layout.archive_sha256,
            expected_members=archive_layout.expected_members,
            allowed_suffixes=frozenset({".csv", ".txt"}),
        )
        by_role[archive_layout.role] = tuple(info.filename for info in infos)
    temperatures = sorted(
        {
            int(match.group(1))
            for member in by_role["characterization"]
            if (match := re.search(r"/(\d+)C_", member)) is not None
        }
    )
    if not temperatures:
        raise ValueError("TEMPEST characterization archive has no reviewed temperatures")
    return TempestDatasetIndex(
        cell_id=validated.cell_id,
        aging_members=by_role["aging"],
        characterization_members=by_role["characterization"],
        temperatures_c=tuple(temperatures),
    )


__all__ = [
    "TempestArchive",
    "TempestDatasetIndex",
    "TempestLayout",
    "load_tempest_index",
]
