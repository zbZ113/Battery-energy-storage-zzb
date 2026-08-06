"""Evidence-bound Markdown model cards for inactive offline artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

_REQUIRED_FIELDS = (
    "task",
    "target",
    "dataset",
    "source_commit",
    "license",
    "training_parameters",
    "validation_rules",
    "metrics_test",
    "seed_statistics",
    "per_cell_tail",
    "calibration",
    "ood_supported_domain",
    "failed_experiments",
    "inference_environment",
    "rejection_conditions",
    "artifact_sha256",
)
_OPTIONAL_FIELDS = ("model_version",)


class ModelCardData(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    model_version: str = Field(default="unspecified", min_length=1)
    target: str = Field(min_length=1)
    dataset: dict[str, Any]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    license: str = Field(min_length=1)
    training_parameters: dict[str, Any]
    validation_rules: list[Any]
    metrics_test: dict[str, Any]
    seed_statistics: dict[str, Any]
    per_cell_tail: dict[str, Any]
    calibration: dict[str, Any]
    ood_supported_domain: dict[str, Any]
    failed_experiments: list[Any]
    inference_environment: dict[str, Any]
    rejection_conditions: list[Any]
    artifact_sha256: Sha256

    @field_validator(
        "dataset",
        "training_parameters",
        "metrics_test",
        "seed_statistics",
        "per_cell_tail",
        "calibration",
        "ood_supported_domain",
        "inference_environment",
    )
    @classmethod
    def mappings_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        sha256_canonical(value)
        return value

    @field_validator("validation_rules", "failed_experiments", "rejection_conditions")
    @classmethod
    def lists_are_json(cls, value: list[Any]) -> list[Any]:
        sha256_canonical(value)
        return value


def render_model_card(payload: Mapping[str, Any] | ModelCardData) -> str:
    """Render a UTF-8 model card from supplied evidence without inventing metrics."""

    if isinstance(payload, ModelCardData):
        data = payload
    else:
        unknown = set(payload) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS) - {
            "activation_status"
        }
        if unknown:
            raise ValueError(f"unknown model card fields: {sorted(unknown)}")
        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"model card is missing required evidence: {missing}")
        status = payload.get("activation_status", "VERIFIED_NOT_ACTIVATED")
        if status != "VERIFIED_NOT_ACTIVATED":
            raise ValueError("model card activation status must be VERIFIED_NOT_ACTIVATED")
        data = ModelCardData.model_validate(
            {
                field: payload[field]
                for field in (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS)
                if field in payload
            }
        )
    lines = [
        f"# {data.task} model card",
        "",
        "## Status",
        "",
        "- Activation status: `VERIFIED_NOT_ACTIVATED`",
        f"- Model version: `{data.model_version}`",
        f"- Source commit: `{data.source_commit}`",
        f"- License: `{data.license}`",
        "",
        "## Data and Split",
        "",
        _json_block(data.dataset),
        "",
        "## Target",
        "",
        f"`{data.target}`",
        "",
        "## Training Parameters",
        "",
        _json_block(data.training_parameters),
        "",
        "## Validation",
        "",
        _json_block(data.validation_rules),
        "",
        "## Test Metrics",
        "",
        _json_block(data.metrics_test),
        "",
        "## Five-Seed Statistics",
        "",
        _json_block(data.seed_statistics),
        "",
        "## Per-Cell Tail",
        "",
        _json_block(data.per_cell_tail),
        "",
        "## Calibration",
        "",
        _json_block(data.calibration),
        "",
        "## OOD Supported Domain",
        "",
        _json_block(data.ood_supported_domain),
        "",
        "## Failed Experiments",
        "",
        _json_block(data.failed_experiments),
        "",
        "## Inference Environment",
        "",
        _json_block(data.inference_environment),
        "",
        "## Rejection Conditions",
        "",
        _json_block(data.rejection_conditions),
        "",
        f"Artifact SHA-256: `{data.artifact_sha256}`",
        "",
    ]
    return "\n".join(lines)


def build_model_card(payload: Mapping[str, Any] | ModelCardData) -> str:
    return render_model_card(payload)


def write_model_card(path: Any, payload: Mapping[str, Any] | ModelCardData) -> None:
    path.write_text(render_model_card(payload), encoding="utf-8")


def _json_block(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("model card evidence must be finite JSON") from exc
    if any(isinstance(item, float) and not math.isfinite(item) for item in _walk(value)):
        raise ValueError("model card evidence must be finite JSON")
    return f"```json\n{encoded}\n```"


def _walk(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


__all__ = ["ModelCardData", "build_model_card", "render_model_card", "write_model_card"]
