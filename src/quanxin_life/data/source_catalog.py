import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import Sha256


class IngestionMode(StrEnum):
    HDF5 = "hdf5"
    MATLAB = "matlab"
    TABULAR = "tabular"
    ARCHIVE_TABULAR = "archive_tabular"
    QUARANTINE_CONVERSION = "quarantine_conversion"


class SourceCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    paper_uri: str = Field(min_length=1)
    license_status: str = Field(min_length=1)
    ingestion_mode: IngestionMode
    expected_suffixes: tuple[str, ...] = Field(min_length=1)
    prohibited_direct_suffixes: tuple[str, ...] = ()
    downloaded_at: datetime | None = None
    artifact_paths: tuple[str, ...] = ()
    artifact_sha256: tuple[Sha256, ...] = ()
    notes: str | None = None

    @field_validator("downloaded_at")
    @classmethod
    def downloaded_at_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("downloaded_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("artifact_paths")
    @classmethod
    def artifact_paths_are_versioned_and_confined(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        for item in value:
            path = PurePosixPath(item)
            if (
                not item.startswith("data/raw/")
                or "\\" in item
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("artifact_paths must use versioned data/raw paths")
        if len(value) != len(set(value)):
            raise ValueError("artifact_paths must be unique")
        return value

    @model_validator(mode="after")
    def quarantine_has_explicit_boundary(self) -> "SourceCatalogEntry":
        if (
            self.ingestion_mode == IngestionMode.QUARANTINE_CONVERSION
            and not self.prohibited_direct_suffixes
        ):
            raise ValueError("quarantine sources require prohibited_direct_suffixes")
        if len(self.artifact_paths) != len(self.artifact_sha256):
            raise ValueError("artifact_paths and artifact_sha256 must align")
        return self


class SourceCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[SourceCatalogEntry, ...]

    @classmethod
    def load(cls, path: Path) -> "SourceCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(SourceCatalogEntry.model_validate(item) for item in payload)
        for entry in entries:
            if (
                entry.downloaded_at is None
                or not entry.artifact_paths
                or not entry.artifact_sha256
            ):
                raise ValueError(
                    "source catalog entries require audited download and artifact metadata"
                )
        identifiers = [entry.dataset_id for entry in entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate dataset_id in source catalog")
        return cls(entries=entries)

    def require(self, dataset_id: str) -> SourceCatalogEntry:
        for entry in self.entries:
            if entry.dataset_id == dataset_id:
                return entry
        raise KeyError(f"dataset source is not registered: {dataset_id}")
