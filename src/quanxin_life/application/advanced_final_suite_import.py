"""Closed-world registration of one mixed-task Advanced Final evidence tree."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.training.advanced_outputs import (
    AdvancedTrainingOutputIndex,
    load_advanced_training_output_index,
    verify_advanced_training_output_index,
)
from quanxin_life.training.checkpoint import AdvancedTrainingCheckpointManifest
from quanxin_life.training.engine import TrainingRunResult, TrainingRunStatus

AdvancedFinalTask = Literal["RUL", "SOH"]
AdvancedFinalOutputTarget = Literal["matr_official_cycle_life", "soh_trajectory"]
AdvancedFinalFamily = Literal[
    "cyclepatch_direct",
    "cyclepatch_batlinet",
    "current_hybrid",
    "hybridpatch_v2",
]

_CUTOFFS = (20, 50, 100, 150)
_SEEDS = (38, 39, 40, 41, 42)
_FAMILY_CANDIDATES: dict[str, str] = {
    "current_hybrid": "current-hybrid-reference",
    "cyclepatch_batlinet": "cpb-d128-p50-r10-n32-a50",
    "cyclepatch_direct": "cpd-d128-l2-h4-p05",
    "hybridpatch_v2": "hpv2-d256-q16-reg-full",
}
_FAMILY_TASKS: dict[str, tuple[AdvancedFinalTask, AdvancedFinalOutputTarget]] = {
    "cyclepatch_direct": ("RUL", "matr_official_cycle_life"),
    "cyclepatch_batlinet": ("RUL", "matr_official_cycle_life"),
    "current_hybrid": ("SOH", "soh_trajectory"),
    "hybridpatch_v2": ("SOH", "soh_trajectory"),
}
_FINAL_PREFIX = "runs/a100/matr-three-batch/advanced/final"


class ImportedAdvancedFinalTaskRecord(ContractModel):
    """One verified Advanced Final task identity without copied business metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["imported-advanced-final-task-v1"] = (
        "imported-advanced-final-task-v1"
    )
    task: AdvancedFinalTask
    output_target: AdvancedFinalOutputTarget
    checkpoint_target: PredictionTarget
    family: AdvancedFinalFamily
    candidate_id: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(gt=0)
    seed: int = Field(gt=0)
    run_id: str = Field(min_length=1, max_length=200)
    status: Literal["COMPLETED", "EARLY_STOPPED"]
    best_epoch: int = Field(ge=0)
    last_epoch: int = Field(ge=0)
    dataset_id: Literal["MATR"] = "MATR"
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    selection_manifest_sha256: Sha256
    candidate_config_sha256: Sha256
    model_architecture_sha256: Sha256
    normalization_sha256: Sha256
    reference_library_sha256: Sha256 | None = None
    checkpoint_context_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    checkpoint_manifest_file_sha256: Sha256
    checkpoint_model_sha256: Sha256
    task_relative_root: str = Field(min_length=1)
    finished_at: datetime

    @field_validator("task_relative_root")
    @classmethod
    def task_root_is_safe(cls, value: str) -> str:
        return _safe_relative(value, "task_relative_root")

    @field_validator("finished_at")
    @classmethod
    def finished_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, "finished_at")

    @model_validator(mode="after")
    def task_semantics_are_consistent(self) -> ImportedAdvancedFinalTaskRecord:
        expected = _FAMILY_TASKS[self.family]
        if (self.task, self.output_target) != expected:
            raise ValueError("Advanced Final family task semantics do not match")
        is_batlinet = self.family == "cyclepatch_batlinet"
        if is_batlinet != (self.reference_library_sha256 is not None):
            raise ValueError("BatLiNet reference provenance does not match its family")
        return self


class ImportedAdvancedFinalSuiteRecord(ContractModel):
    """Immutable trust-anchor record for an externally retained Advanced Final tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["imported-advanced-final-suite-v1"] = (
        "imported-advanced-final-suite-v1"
    )
    import_id: Sha256
    task_scope: Literal["MIXED_RUL_SOH"] = "MIXED_RUL_SOH"
    mode: Literal["final"] = "final"
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    config_sha256: Sha256
    output_sha256: Sha256
    output_index_file_sha256: Sha256
    transfer_sha256: Sha256
    training_input_bundle_sha256: Sha256
    local_reconstructed_input_bundle_sha256: Sha256
    input_bundle_hashes_match: bool
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    selection_manifest_sha256: Sha256
    task_count: Literal[80] = 80
    rul_task_count: Literal[40] = 40
    soh_task_count: Literal[40] = 40
    completed_task_count: int = Field(ge=0, le=80)
    early_stopped_task_count: int = Field(ge=0, le=80)
    file_count: int = Field(gt=0)
    source_relative_root: str = Field(min_length=1)
    transfer_relative_path: str = Field(min_length=1)
    output_created_at: datetime
    registered_at: datetime
    record_sha256: Sha256

    @field_validator("source_relative_root", "transfer_relative_path")
    @classmethod
    def evidence_path_is_safe(cls, value: str, info: Any) -> str:
        return _safe_relative(value, info.field_name)

    @field_validator("output_created_at", "registered_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def counts_hashes_and_record_are_consistent(
        self,
    ) -> ImportedAdvancedFinalSuiteRecord:
        if self.completed_task_count + self.early_stopped_task_count != 80:
            raise ValueError("Advanced Final status counts must total 80")
        hashes_match = (
            self.training_input_bundle_sha256
            == self.local_reconstructed_input_bundle_sha256
        )
        if self.input_bundle_hashes_match != hashes_match:
            raise ValueError("input bundle comparison does not match its digests")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if self.record_sha256 != sha256_canonical(payload):
            raise ValueError("Advanced Final registry record SHA-256 does not match")
        return self


@dataclass(frozen=True, slots=True)
class RegisteredAdvancedFinalSuite:
    """A verified external evidence tree paired with its 80 task records."""

    record: ImportedAdvancedFinalSuiteRecord
    output_root: Path
    tasks: tuple[ImportedAdvancedFinalTaskRecord, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedFinal:
    index: AdvancedTrainingOutputIndex
    tasks: tuple[ImportedAdvancedFinalTaskRecord, ...]
    training_input_bundle_sha256: str
    data_version: str
    split_version: str
    feature_version: str
    selection_manifest_sha256: str


class AdvancedFinalSuiteImporter:
    """Register and freshly reverify one exact 80-run Advanced Final suite."""

    def __init__(self, registry_root: Path, *, evidence_root: Path) -> None:
        registry_root.mkdir(parents=True, exist_ok=True)
        self._root = _regular_directory(registry_root, "Advanced Final registry root")
        self._records = self._root / "records"
        self._records.mkdir(exist_ok=True)
        self._records = _regular_directory(self._records, "Advanced Final records")
        self._evidence_root = _regular_directory(evidence_root, "evidence root")

    def register(
        self,
        output_root: Path,
        *,
        expected_output_sha256: str,
        expected_output_index_file_sha256: str,
        transfer_archive: Path,
        transfer_sha256: str,
        local_reconstructed_input_bundle_sha256: str,
        registered_at: datetime,
    ) -> ImportedAdvancedFinalSuiteRecord:
        """Register metadata for exact external bytes without duplicating the 15 GB tree."""

        self._verify_registry_directories()
        output = _inside_root(self._evidence_root, output_root, "Advanced Final output")
        transfer = _inside_file(
            self._evidence_root,
            transfer_archive,
            "Advanced Final transfer archive",
        )
        if (
            output.is_relative_to(self._root)
            or self._root.is_relative_to(output)
            or transfer.is_relative_to(self._root)
        ):
            raise ValueError("Advanced Final evidence must not overlap its registry")
        expected_output = _sha256(expected_output_sha256, "expected output SHA-256")
        expected_index_file = _sha256(
            expected_output_index_file_sha256,
            "expected index file SHA-256",
        )
        expected_transfer = _sha256(transfer_sha256, "transfer archive SHA-256")
        local_input = _sha256(
            local_reconstructed_input_bundle_sha256,
            "local reconstructed input bundle SHA-256",
        )
        if _sha256_file(transfer) != expected_transfer:
            raise ValueError("transfer archive SHA-256 does not match")
        verified = _verify_final(
            output,
            expected_output_sha256=expected_output,
            expected_output_index_file_sha256=expected_index_file,
        )
        record = _suite_record(
            verified,
            expected_output_index_file_sha256=expected_index_file,
            transfer_sha256=expected_transfer,
            local_reconstructed_input_bundle_sha256=local_input,
            source_relative_root=output.relative_to(self._evidence_root).as_posix(),
            transfer_relative_path=transfer.relative_to(self._evidence_root).as_posix(),
            registered_at=registered_at,
        )
        path = self._records / f"{record.import_id}.json"
        self._verify_registry_directories()
        if path.exists() or path.is_symlink():
            existing = _load_record(path)
            _assert_reimport_context(existing, record)
            self.resolve(existing.import_id)
            return existing
        created = _write_record_atomic(
            path,
            record,
            root=self._root,
            before_link=self._verify_registry_directories,
        )
        if not created:
            self._verify_registry_directories()
            existing = _load_record(path)
            _assert_reimport_context(existing, record)
            self.resolve(existing.import_id)
            return existing
        return record

    def resolve(self, import_id: str) -> RegisteredAdvancedFinalSuite:
        """Resolve a registration only after rechecking all indexed bytes."""

        self._verify_registry_directories()
        normalized = _sha256(import_id, "import_id")
        path = self._records / f"{normalized}.json"
        if _is_reparse_point(path) or not path.is_file():
            raise KeyError(f"unknown Advanced Final import: {normalized}")
        record = _load_record(path)
        if record.import_id != normalized:
            raise ValueError("Advanced Final record identity does not match")
        output = _inside_root(
            self._evidence_root,
            self._evidence_root / record.source_relative_root,
            "registered Advanced Final output",
        )
        transfer = _inside_file(
            self._evidence_root,
            self._evidence_root / record.transfer_relative_path,
            "registered transfer archive",
        )
        if _sha256_file(transfer) != record.transfer_sha256:
            raise ValueError("registered transfer archive SHA-256 does not match")
        verified = _verify_final(
            output,
            expected_output_sha256=record.output_sha256,
            expected_output_index_file_sha256=record.output_index_file_sha256,
        )
        expected = _suite_record(
            verified,
            expected_output_index_file_sha256=record.output_index_file_sha256,
            transfer_sha256=record.transfer_sha256,
            local_reconstructed_input_bundle_sha256=(
                record.local_reconstructed_input_bundle_sha256
            ),
            source_relative_root=record.source_relative_root,
            transfer_relative_path=record.transfer_relative_path,
            registered_at=record.registered_at,
        )
        if expected != record:
            raise ValueError("Advanced Final record differs from verified evidence")
        return RegisteredAdvancedFinalSuite(
            record=record,
            output_root=output,
            tasks=verified.tasks,
        )

    def list_tasks(self, import_id: str) -> tuple[ImportedAdvancedFinalTaskRecord, ...]:
        return self.resolve(import_id).tasks

    def _verify_registry_directories(self) -> None:
        _managed_directory(self._records, root=self._root, name="records")


def _verify_final(
    root: Path,
    *,
    expected_output_sha256: str,
    expected_output_index_file_sha256: str,
) -> _VerifiedFinal:
    _reject_nested_reparse_points(root)
    index_path = _regular_file(root / "output_index.json", "Advanced Final index")
    if _sha256_file(index_path) != expected_output_index_file_sha256:
        raise ValueError("Advanced Final index file SHA-256 does not match")
    index = load_advanced_training_output_index(index_path)
    if index.mode != "final" or index.operation_count != 80 or index.run_directory_count != 80:
        raise ValueError("Advanced Final index is not an exact 80-run final suite")
    if index.output_sha256 != expected_output_sha256:
        raise ValueError("Advanced Final output SHA-256 does not match")
    verify_advanced_training_output_index(root, index)

    aggregate = _strict_json(root / "aggregate_metrics.json")
    if (
        aggregate.get("mode") != "final"
        or aggregate.get("dataset_id") != "MATR"
        or aggregate.get("target") != PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value
        or aggregate.get("run_count") != 80
    ):
        raise ValueError("Advanced Final aggregate identity is invalid")
    aggregate_input = _sha256(
        aggregate.get("input_bundle_sha256"),
        "aggregate input bundle SHA-256",
    )
    aggregate_rows = aggregate.get("runs")
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != 80:
        raise ValueError("Advanced Final aggregate run matrix is incomplete")
    aggregate_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for raw in aggregate_rows:
        if not isinstance(raw, dict):
            raise ValueError("Advanced Final aggregate run entry is invalid")
        key = (
            str(raw.get("family", "")),
            str(raw.get("candidate_id", "")),
            _integer(raw.get("cutoff_cycle"), "aggregate cutoff_cycle"),
            _integer(raw.get("seed"), "aggregate seed"),
        )
        if key in aggregate_by_key:
            raise ValueError("Advanced Final aggregate contains duplicate coordinates")
        aggregate_by_key[key] = raw

    expected_keys = {
        (family, candidate, cutoff, seed)
        for family, candidate in _FAMILY_CANDIDATES.items()
        for cutoff in _CUTOFFS
        for seed in _SEEDS
    }
    actual_status_roots = {
        path.parent.relative_to(root).as_posix()
        for path in root.rglob("run_status.json")
    }
    expected_status_roots = {
        _task_relative_root(family, candidate, cutoff, seed)
        for family, candidate, cutoff, seed in expected_keys
    }
    if set(aggregate_by_key) != expected_keys or actual_status_roots != expected_status_roots:
        raise ValueError("Advanced Final candidate matrix is incomplete or unexpected")

    tasks = tuple(
        _task_record(
            root,
            family=family,
            candidate_id=candidate,
            cutoff_cycle=cutoff,
            seed=seed,
            source_commit=index.source_commit,
            config_sha256=index.config_sha256,
            aggregate=aggregate_by_key[(family, candidate, cutoff, seed)],
        )
        for family, candidate, cutoff, seed in sorted(expected_keys)
    )
    if len({task.run_id for task in tasks}) != 80:
        raise ValueError("Advanced Final run IDs must be unique")
    common = {
        (
            task.input_bundle_sha256,
            task.data_version,
            task.split_version,
            task.feature_version,
            task.selection_manifest_sha256,
        )
        for task in tasks
    }
    if len(common) != 1:
        raise ValueError("Advanced Final tasks do not share one frozen provenance context")
    input_hash, data_version, split_version, feature_version, selection_hash = next(
        iter(common)
    )
    if input_hash != aggregate_input:
        raise ValueError("Advanced Final aggregate input bundle differs from checkpoints")
    return _VerifiedFinal(
        index=index,
        tasks=tasks,
        training_input_bundle_sha256=input_hash,
        data_version=data_version,
        split_version=split_version,
        feature_version=feature_version,
        selection_manifest_sha256=selection_hash,
    )


def _task_record(
    root: Path,
    *,
    family: str,
    candidate_id: str,
    cutoff_cycle: int,
    seed: int,
    source_commit: str,
    config_sha256: str,
    aggregate: dict[str, Any],
) -> ImportedAdvancedFinalTaskRecord:
    relative_root = _task_relative_root(family, candidate_id, cutoff_cycle, seed)
    run_root = _regular_directory(root / relative_root, "Advanced Final run")
    result = TrainingRunResult.model_validate(_strict_json(run_root / "run_status.json"))
    if result.status not in {
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.EARLY_STOPPED,
    }:
        raise ValueError("Advanced Final run status is not terminal")
    status: Literal["COMPLETED", "EARLY_STOPPED"] = (
        "COMPLETED"
        if result.status is TrainingRunStatus.COMPLETED
        else "EARLY_STOPPED"
    )
    if result.best_epoch is None:
        raise ValueError("Advanced Final run is missing a best epoch")
    pointer = _strict_json(run_root / "checkpoints" / "best.json")
    checkpoint_name = pointer.get("checkpoint")
    expected_checkpoint = f"epoch-{result.best_epoch:06d}"
    if checkpoint_name != expected_checkpoint:
        raise ValueError("Advanced Final best pointer differs from run status")
    checkpoint_root = _regular_directory(
        run_root / "checkpoints" / expected_checkpoint,
        "Advanced Final best checkpoint",
    )
    manifest_path = _regular_file(
        checkpoint_root / "manifest.json",
        "Advanced Final checkpoint manifest",
    )
    manifest = AdvancedTrainingCheckpointManifest.model_validate(_strict_json(manifest_path))
    unsigned = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if manifest.manifest_sha256 != sha256_canonical(unsigned):
        raise ValueError("Advanced Final checkpoint manifest SHA-256 does not match")
    expected_files = {item.relative_path for item in manifest.files} | {"manifest.json"}
    actual_files = {path.name for path in checkpoint_root.iterdir()}
    if actual_files != expected_files:
        raise ValueError("Advanced Final checkpoint file inventory is not closed")
    for item in manifest.files:
        path = _regular_file(checkpoint_root / item.relative_path, "checkpoint file")
        if path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
            raise ValueError("Advanced Final checkpoint file differs from its manifest")

    context = manifest.context
    expected_run_id = f"matr-{family}-{candidate_id}-c{cutoff_cycle}-s{seed}"
    if (
        context.run_id != expected_run_id
        or context.dataset_id != "MATR"
        or context.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        or context.model_name != family
        or context.cutoff_cycle != cutoff_cycle
        or context.seed != seed
        or context.source_commit != source_commit
        or context.config_sha256 != config_sha256
        or context.run_mode != "final"
        or context.stage != "final"
        or context.selection_manifest_sha256 is None
    ):
        raise ValueError("Advanced Final checkpoint context differs from its coordinates")
    context_hash = sha256_canonical(context.model_dump(mode="json"))
    if result.run_id != context.run_id or result.context_sha256 != context_hash:
        raise ValueError("Advanced Final run status differs from checkpoint context")
    if manifest.progress.best_epoch != result.best_epoch:
        raise ValueError("Advanced Final best epoch differs from checkpoint progress")
    aggregate_root = f"{_FINAL_PREFIX}/{relative_root}"
    if (
        aggregate.get("status") != result.status.value
        or aggregate.get("best_epoch") != result.best_epoch
        or str(aggregate.get("run_directory", "")).replace("\\", "/")
        != aggregate_root
    ):
        raise ValueError("Advanced Final aggregate run context does not match evidence")
    task, output_target = _FAMILY_TASKS[family]
    return ImportedAdvancedFinalTaskRecord(
        task=task,
        output_target=output_target,
        checkpoint_target=context.target,
        family=family,
        candidate_id=candidate_id,
        cutoff_cycle=cutoff_cycle,
        seed=seed,
        run_id=context.run_id,
        status=status,
        best_epoch=result.best_epoch,
        last_epoch=result.last_epoch,
        source_commit=context.source_commit,
        config_sha256=context.config_sha256,
        input_bundle_sha256=context.input_bundle_sha256,
        data_version=context.data_version,
        split_version=context.split_version,
        feature_version=context.feature_version,
        selection_manifest_sha256=context.selection_manifest_sha256,
        candidate_config_sha256=context.candidate_config_sha256,
        model_architecture_sha256=context.model_architecture_sha256,
        normalization_sha256=context.normalization_sha256,
        reference_library_sha256=context.reference_library_sha256,
        checkpoint_context_sha256=context_hash,
        checkpoint_manifest_sha256=manifest.manifest_sha256,
        checkpoint_manifest_file_sha256=_sha256_file(manifest_path),
        checkpoint_model_sha256=_sha256_file(checkpoint_root / "model.safetensors"),
        task_relative_root=relative_root,
        finished_at=result.finished_at,
    )


def _suite_record(
    verified: _VerifiedFinal,
    *,
    expected_output_index_file_sha256: str,
    transfer_sha256: str,
    local_reconstructed_input_bundle_sha256: str,
    source_relative_root: str,
    transfer_relative_path: str,
    registered_at: datetime,
) -> ImportedAdvancedFinalSuiteRecord:
    completed = sum(task.status == "COMPLETED" for task in verified.tasks)
    payload = {
        "schema_version": "imported-advanced-final-suite-v1",
        "import_id": verified.index.output_sha256,
        "task_scope": "MIXED_RUL_SOH",
        "mode": "final",
        "source_commit": verified.index.source_commit,
        "config_sha256": verified.index.config_sha256,
        "output_sha256": verified.index.output_sha256,
        "output_index_file_sha256": expected_output_index_file_sha256,
        "transfer_sha256": transfer_sha256,
        "training_input_bundle_sha256": verified.training_input_bundle_sha256,
        "local_reconstructed_input_bundle_sha256": (
            local_reconstructed_input_bundle_sha256
        ),
        "input_bundle_hashes_match": (
            verified.training_input_bundle_sha256
            == local_reconstructed_input_bundle_sha256
        ),
        "data_version": verified.data_version,
        "split_version": verified.split_version,
        "feature_version": verified.feature_version,
        "selection_manifest_sha256": verified.selection_manifest_sha256,
        "task_count": 80,
        "rul_task_count": 40,
        "soh_task_count": 40,
        "completed_task_count": completed,
        "early_stopped_task_count": 80 - completed,
        "file_count": len(verified.index.files),
        "source_relative_root": _safe_relative(
            source_relative_root,
            "source_relative_root",
        ),
        "transfer_relative_path": _safe_relative(
            transfer_relative_path,
            "transfer_relative_path",
        ),
        "output_created_at": _utc_iso(
            verified.index.created_at,
            "output_created_at",
        ),
        "registered_at": _utc_iso(registered_at, "registered_at"),
    }
    return ImportedAdvancedFinalSuiteRecord.model_validate(
        {**payload, "record_sha256": sha256_canonical(payload)}
    )


def _task_relative_root(
    family: str,
    candidate_id: str,
    cutoff_cycle: int,
    seed: int,
) -> str:
    return Path(
        f"cutoff-{cutoff_cycle}",
        family,
        candidate_id,
        f"seed-{seed}",
    ).as_posix()


def _assert_reimport_context(
    existing: ImportedAdvancedFinalSuiteRecord,
    candidate: ImportedAdvancedFinalSuiteRecord,
) -> None:
    excluded = {"registered_at", "record_sha256"}
    if existing.model_dump(exclude=excluded) != candidate.model_dump(exclude=excluded):
        raise ValueError("Advanced Final registration context conflicts")


def _write_record_atomic(
    path: Path,
    record: ImportedAdvancedFinalSuiteRecord,
    *,
    root: Path,
    before_link: Callable[[], None],
) -> bool:
    temporary = root / f".record-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        before_link()
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _load_record(path: Path) -> ImportedAdvancedFinalSuiteRecord:
    return ImportedAdvancedFinalSuiteRecord.model_validate(_strict_json(path))


def _strict_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "JSON evidence")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON evidence must contain an object")
    return payload


def _inside_root(root: Path, path: Path, label: str) -> Path:
    candidate = _regular_directory(path, label)
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} must remain inside the evidence root")
    return candidate


def _inside_file(root: Path, path: Path, label: str) -> Path:
    candidate = _regular_file(path, label)
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} must remain inside the evidence root")
    return candidate


def _regular_directory(path: Path, label: str) -> Path:
    if _is_reparse_point(path):
        raise ValueError(f"{label} must be a regular directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return resolved


def _regular_file(path: Path, label: str) -> Path:
    if _is_reparse_point(path):
        raise ValueError(f"{label} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _managed_directory(path: Path, *, root: Path, name: str) -> Path:
    if _is_reparse_point(root) or _is_reparse_point(path):
        raise ValueError("Advanced Final registry directories contain a reparse point")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Advanced Final registry directories must exist") from exc
    if (
        resolved_root != root
        or resolved != root / name
        or not resolved_root.is_dir()
        or not resolved.is_dir()
    ):
        raise ValueError("Advanced Final registry directories escaped the registry root")
    return resolved


def _reject_nested_reparse_points(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        if current_path != root and _is_reparse_point(current_path):
            raise ValueError("Advanced Final evidence contains a nested reparse point")
        for name in (*directory_names, *file_names):
            if _is_reparse_point(current_path / name):
                raise ValueError("Advanced Final evidence contains a nested reparse point")


def _is_reparse_point(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _safe_relative(value: str, label: str) -> str:
    normalized = value.replace("\\", "/") if isinstance(value, str) else ""
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or path.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must remain relative")
    return path.as_posix()


def _sha256(value: Any, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if integer != value:
        raise ValueError(f"{label} must be an integer")
    return integer


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value.astimezone(UTC)


def _utc_iso(value: datetime, label: str) -> str:
    return _utc(value, label).isoformat().replace("+00:00", "Z")


__all__ = [
    "AdvancedFinalSuiteImporter",
    "ImportedAdvancedFinalSuiteRecord",
    "ImportedAdvancedFinalTaskRecord",
    "RegisteredAdvancedFinalSuite",
]
