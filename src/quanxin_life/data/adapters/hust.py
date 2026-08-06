"""Main-process verification boundary for isolated HUST conversion outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256


class HustFieldMapping(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_member: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    canonical_field: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    unit_evidence: str = Field(min_length=1)
    conversion_rule: str = Field(min_length=1)
    status: Literal["REVIEWED", "UNRESOLVED"]


class HustReviewedLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    dataset_id: Literal["HUST"] = "HUST"
    inventory_version: str = Field(min_length=1)
    inventory_sha256: Sha256
    archive_sha256: Sha256
    member_count: Literal[77] = 77
    review_status: Literal["APPROVED", "BLOCKED_REVIEW"]
    fields: tuple[HustFieldMapping, ...]
    unresolved_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def approval_requires_resolved_fields(self) -> HustReviewedLayout:
        if self.review_status == "APPROVED" and (
            not self.fields or any(field.status != "REVIEWED" for field in self.fields)
        ):
            raise ValueError("APPROVED HUST layout requires reviewed fields")
        if self.review_status == "BLOCKED_REVIEW" and not self.unresolved_reasons:
            raise ValueError("BLOCKED_REVIEW requires unresolved reasons")
        return self


def load_hust_layout(path: Path) -> HustReviewedLayout:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HUST layout must be valid UTF-8 JSON") from exc
    return HustReviewedLayout.model_validate(payload)


def verify_hust_output(
    output_root: Path,
    *,
    layout: HustReviewedLayout,
) -> tuple[str, ...]:
    """Verify only isolated JSON/Parquet outputs; raw pickle objects stay inaccessible."""

    validated = HustReviewedLayout.model_validate(layout.model_dump(mode="json"))
    if validated.review_status != "APPROVED":
        raise ValueError("BLOCKED_REVIEW: HUST semantic layout is not approved")
    root = Path(output_root).resolve(strict=True)
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("inventory_sha256") != validated.inventory_sha256:
        raise ValueError("HUST output inventory SHA mismatch")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("HUST output manifest requires files")
    verified: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("HUST output file entry must be an object")
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("HUST output file entry is incomplete")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix.lower() not in {".json", ".parquet"}
        ):
            raise ValueError("HUST output contains an unsafe artifact path")
        candidate = root.joinpath(*path.parts).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError("HUST output artifact escapes its root")
        actual = _sha256_file(candidate)
        if actual != expected:
            raise ValueError("HUST output artifact SHA mismatch")
        verified.append(actual)
    return tuple(verified)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "HustFieldMapping",
    "HustReviewedLayout",
    "load_hust_layout",
    "verify_hust_output",
]
