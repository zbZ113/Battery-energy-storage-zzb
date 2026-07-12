import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from quanxin_life.data.security import assert_safe_external_data_file

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RawFileManifest(BaseModel):
    """Provenance record required before a raw file can enter processing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    source_uri: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_uri: str | None = None
    paper_doi: str | None = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("downloaded_at")
    @classmethod
    def normalize_download_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("downloaded_at must include a timezone")
        return value.astimezone(UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_file(path: Path, manifest: RawFileManifest) -> str:
    """Reject executable serialization and verify immutable raw-file provenance."""
    assert_safe_external_data_file(path)
    if not path.is_file():
        raise ValueError(f"raw data file does not exist: {path}")
    actual = _sha256_file(path)
    if actual != manifest.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {manifest.relative_path}: "
            f"expected {manifest.sha256}, got {actual}"
        )
    return actual
