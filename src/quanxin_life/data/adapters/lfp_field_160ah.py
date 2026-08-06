"""System-level source index for the 160 Ah field telemetry archive."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.zip_safety import audit_zip_archive

_MEMBER = re.compile(r"^field_data/data_sys_(?P<system>\d+)\.csv$")


class FieldLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    archive_sha256: Sha256
    expected_members: tuple[str, ...] = Field(min_length=1)
    ignored_members: tuple[str, ...] = ()

    @model_validator(mode="after")
    def member_sets_do_not_overlap(self) -> FieldLayout:
        if set(self.expected_members) & set(self.ignored_members):
            raise ValueError("field expected and ignored members must not overlap")
        return self


class FieldSystemSource(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["LFP_FIELD_160AH"] = "LFP_FIELD_160AH"
    system_id: str = Field(min_length=1)
    source_member: str = Field(min_length=1)
    archive_sha256: Sha256
    split_group: str = Field(min_length=1)
    timestamp_semantics: Literal["LOCAL_TIME_TIMEZONE_UNRESOLVED"] = (
        "LOCAL_TIME_TIMEZONE_UNRESOLVED"
    )


def load_field_sources(
    path: Path,
    *,
    layout: FieldLayout,
) -> tuple[FieldSystemSource, ...]:
    validated = FieldLayout.model_validate(layout.model_dump(mode="json"))
    infos = audit_zip_archive(
        path,
        archive_sha256=validated.archive_sha256,
        expected_members=validated.expected_members,
        ignored_members=validated.ignored_members,
        allowed_suffixes=frozenset({".csv", ".ds_store"}),
    )
    records: list[FieldSystemSource] = []
    for info in infos:
        match = _MEMBER.fullmatch(info.filename)
        if match is None:
            raise ValueError("field member path does not match reviewed system layout")
        system_id = f"FIELD-SYSTEM-{int(match.group('system'))}"
        records.append(
            FieldSystemSource(
                system_id=system_id,
                source_member=info.filename,
                archive_sha256=validated.archive_sha256,
                split_group=system_id,
            )
        )
    if len({item.system_id for item in records}) != len(records):
        raise ValueError("field system identities must be unique")
    return tuple(records)


__all__ = ["FieldLayout", "FieldSystemSource", "load_field_sources"]
