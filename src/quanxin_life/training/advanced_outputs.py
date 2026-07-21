"""Byte-level indexes for complete advanced A100 training stages."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.training.outputs import TrainingOutputFile, _collect_output_files

AdvancedOutputMode = Literal["smoke", "select", "final"]
_INDEX_NAME = "output_index.json"
_COMPLETED_STATUSES = frozenset({"COMPLETED", "EARLY_STOPPED"})


class AdvancedTrainingOutputIndex(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-training-output-index-v1"] = (
        "advanced-training-output-index-v1"
    )
    mode: AdvancedOutputMode
    created_at: datetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    config_sha256: Sha256
    operation_count: int = Field(gt=0)
    run_directory_count: int = Field(gt=0)
    files: tuple[TrainingOutputFile, ...] = Field(min_length=1)
    output_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


def build_advanced_training_output_index(
    output_root: Path,
    *,
    mode: AdvancedOutputMode,
    source_commit: str,
    config_sha256: str,
    created_at: datetime,
) -> AdvancedTrainingOutputIndex:
    root = output_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("advanced output root must be a regular directory")
    files = _collect_output_files(
        root, excluded_relative_paths=frozenset({_INDEX_NAME})
    )
    relative_paths = {item.relative_path for item in files}
    status_paths = sorted(path for path in relative_paths if path.endswith("/run_status.json"))
    if not status_paths:
        raise ValueError("advanced output contains no completed run directories")
    accepted_statuses = (
        _COMPLETED_STATUSES | {"PAUSED_STAGE"}
        if mode == "select"
        else _COMPLETED_STATUSES
    )
    for relative in status_paths:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        if payload.get("status") not in accepted_statuses:
            raise ValueError(f"advanced run is not complete: {relative}")
        parent = Path(relative).parent.as_posix()
        required = {
            f"{parent}/training_log.jsonl",
            f"{parent}/metrics_validation.csv",
        }
        if not required <= relative_paths:
            raise ValueError(f"advanced run evidence is incomplete: {parent}")
        if not any(
            path.startswith(f"{parent}/checkpoints/")
            and path.endswith("/model.safetensors")
            for path in relative_paths
        ):
            raise ValueError(f"advanced run checkpoint is missing: {parent}")

    operation_count = _validate_mode_evidence(
        root=root,
        mode=mode,
        relative_paths=relative_paths,
        run_directory_count=len(status_paths),
    )
    payload = {
        "schema_version": "advanced-training-output-index-v1",
        "mode": mode,
        "created_at": created_at.astimezone(UTC).isoformat(),
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "operation_count": operation_count,
        "run_directory_count": len(status_paths),
        "files": [item.model_dump(mode="json") for item in files],
    }
    return AdvancedTrainingOutputIndex(
        **payload,
        output_sha256=sha256_canonical(payload),
    )


def verify_advanced_training_output_index(
    output_root: Path, index: AdvancedTrainingOutputIndex
) -> None:
    actual = build_advanced_training_output_index(
        output_root,
        mode=index.mode,
        source_commit=index.source_commit,
        config_sha256=index.config_sha256,
        created_at=index.created_at,
    )
    if actual != index:
        raise ValueError("advanced training output differs from its index")


def load_advanced_training_output_index(path: Path) -> AdvancedTrainingOutputIndex:
    if path.is_symlink() or not path.is_file():
        raise ValueError("advanced training output index must be a regular file")
    return AdvancedTrainingOutputIndex.model_validate_json(path.read_bytes())


def write_advanced_training_output_index(
    output_root: Path,
    *,
    mode: AdvancedOutputMode,
    source_commit: str,
    config_sha256: str,
    created_at: datetime,
) -> AdvancedTrainingOutputIndex:
    root = output_root.resolve(strict=True)
    index = build_advanced_training_output_index(
        root,
        mode=mode,
        source_commit=source_commit,
        config_sha256=config_sha256,
        created_at=created_at,
    )
    destination = root / _INDEX_NAME
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return index


def _validate_mode_evidence(
    *,
    root: Path,
    mode: AdvancedOutputMode,
    relative_paths: set[str],
    run_directory_count: int,
) -> int:
    if mode == "smoke":
        if run_directory_count != 4 or "aggregate_metrics.json" not in relative_paths:
            raise ValueError("advanced Smoke requires exactly four completed runs")
        return 4
    if mode == "final":
        test_metrics = [path for path in relative_paths if path.endswith("/metrics_test.json")]
        if (
            run_directory_count != 80
            or len(test_metrics) != 80
            or "aggregate_metrics.json" not in relative_paths
        ):
            raise ValueError("advanced Final requires 80 completed test-evaluated runs")
        return 80

    required = {
        "selection_manifest.json",
        "selection_trace.json",
        "final_config_resolved.json",
    }
    if not required <= relative_paths:
        raise ValueError("advanced Select evidence is incomplete")
    trace = json.loads((root / "selection_trace.json").read_text(encoding="utf-8"))
    operation_count = sum(
        len(trace[name])
        for name in ("stage1_evidence", "stage2_evidence", "recheck_evidence")
    )
    if operation_count != 104:
        raise ValueError("advanced Select requires 104 recorded training operations")
    return operation_count
