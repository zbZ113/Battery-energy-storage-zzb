import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IngestionMode(StrEnum):
    HDF5 = "hdf5"
    TABULAR = "tabular"
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
    notes: str | None = None

    @model_validator(mode="after")
    def quarantine_has_explicit_boundary(self) -> "SourceCatalogEntry":
        if (
            self.ingestion_mode == IngestionMode.QUARANTINE_CONVERSION
            and not self.prohibited_direct_suffixes
        ):
            raise ValueError("quarantine sources require prohibited_direct_suffixes")
        return self


class SourceCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[SourceCatalogEntry, ...]

    @classmethod
    def load(cls, path: Path) -> "SourceCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(SourceCatalogEntry.model_validate(item) for item in payload)
        identifiers = [entry.dataset_id for entry in entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate dataset_id in source catalog")
        return cls(entries=entries)

    def require(self, dataset_id: str) -> SourceCatalogEntry:
        for entry in self.entries:
            if entry.dataset_id == dataset_id:
                return entry
        raise KeyError(f"dataset source is not registered: {dataset_id}")
