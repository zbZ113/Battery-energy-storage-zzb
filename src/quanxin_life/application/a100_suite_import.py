"""Closed-world import of completed three-batch MATR A100 experiment suites."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.training.suite import MatrThreeBatchRunConfig

_ROOT_EVIDENCE = frozenset(
    {
        "aggregate_metrics.csv",
        "aggregate_metrics.json",
        "experiment_summary.md",
        "metrics_test.csv",
    }
)
_TASK_EVIDENCE = frozenset(
    {
        "config_resolved.json",
        "environment.json",
        "metrics_epoch.csv",
        "metrics_test.csv",
        "metrics_test.json",
        "metrics_validation.csv",
        "model_card.md",
        "training_log.jsonl",
    }
)
_SAFE_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".md",
        ".png",
        ".safetensors",
        ".svg",
        ".ubj",
    }
)
_MLFLOW_SAFE_SUFFIXES = frozenset({"", ".txt", ".yaml", ".yml"})
_FORBIDDEN_SUFFIXES = frozenset(
    {".joblib", ".key", ".pem", ".pickle", ".pkl", ".pt", ".pth"}
)
_SECRET_PATTERN = re.compile(
    rb"(?i)['\"]?(?:api[_-]?key|app[_-]?secret|password|private[_-]?key)"
    rb"['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-]{8,}"
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


class A100SuiteFile(ContractModel):
    """One verified regular file in an imported experiment suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if (
            path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("suite file path must remain inside the output root")
        return path.as_posix()


class ImportedA100SuiteRecord(ContractModel):
    """Immutable registry record for one accepted A100 suite byte tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["imported-a100-suite-v1"] = "imported-a100-suite-v1"
    import_id: Sha256
    mode: Literal["smoke", "final"]
    dataset_id: Literal["MATR"] = "MATR"
    target: Literal["matr_official_cycle_life"] = "matr_official_cycle_life"
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    output_sha256: Sha256
    transfer_sha256: Sha256
    task_count: int = Field(gt=0)
    file_count: int = Field(gt=0)
    formal_performance_claim: bool
    registered_relative_root: str = Field(min_length=1)
    imported_at: datetime

    @field_validator("imported_at")
    @classmethod
    def imported_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("imported_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("registered_relative_root")
    @classmethod
    def registered_root_is_safe(cls, value: str) -> str:
        path = Path(value.replace("\\", "/"))
        if (
            path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("registered root must be repository-relative")
        return path.as_posix()


class ImportedA100TaskRecord(ContractModel):
    """Identity and immutable context for one verified suite task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["imported-a100-task-v1"] = "imported-a100-task-v1"
    run_id: str = Field(min_length=1, max_length=200)
    dataset_id: Literal["MATR"] = "MATR"
    target: Literal["matr_official_cycle_life"] = "matr_official_cycle_life"
    model_name: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(gt=0)
    seed: int = Field(gt=0)
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    context_sha256: Sha256
    task_relative_root: str = Field(min_length=1)
    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def completed_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("task_relative_root")
    @classmethod
    def task_root_is_safe(cls, value: str) -> str:
        path = Path(value.replace("\\", "/"))
        if (
            path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("task root must remain inside the suite output")
        return path.as_posix()


@dataclass(frozen=True, slots=True)
class RegisteredA100Suite:
    """A registry record paired with a freshly reverified managed directory."""

    record: ImportedA100SuiteRecord
    output_root: Path


@dataclass(frozen=True, slots=True)
class _VerifiedSuite:
    config: MatrThreeBatchRunConfig
    source_commit: str
    input_bundle_sha256: str
    aggregate: dict[str, Any]
    files: tuple[A100SuiteFile, ...]
    output_sha256: str
    task_count: int


class A100SuiteRunImporter:
    """Verify, atomically copy and resolve completed MATR experiment suites."""

    def __init__(self, registry_root: Path) -> None:
        self._root = registry_root.resolve()
        self._runs = self._root / "runs"
        self._records = self._root / "records"
        self._runs.mkdir(parents=True, exist_ok=True)
        self._records.mkdir(parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise ValueError("A100 suite registry root must be a regular directory")

    def import_run(
        self,
        output_root: Path,
        *,
        config_path: Path,
        expected_source_commit: str,
        transfer_archive: Path,
        transfer_sha256: str,
        imported_at: datetime,
    ) -> ImportedA100SuiteRecord:
        """Accept one complete suite, or return its identical existing record."""

        _validate_commit(expected_source_commit)
        transfer = _validate_sha256(transfer_sha256, name="transfer_sha256")
        archive = _validated_regular_file(transfer_archive)
        if _sha256_file(archive) != transfer:
            raise ValueError("transfer archive SHA-256 does not match")
        config = _load_run_config(config_path)
        source = _validated_directory(output_root)
        if source.is_relative_to(self._root) or self._root.is_relative_to(source):
            raise ValueError("source output and registry roots must not overlap")
        verified = _verify_suite(
            source,
            config=config,
            expected_source_commit=expected_source_commit,
        )
        relative_root = Path("runs", verified.output_sha256).as_posix()
        record = ImportedA100SuiteRecord(
            import_id=verified.output_sha256,
            mode=config.mode,
            source_commit=verified.source_commit,
            config_sha256=config.suite.config_sha256,
            input_bundle_sha256=verified.input_bundle_sha256,
            data_version=config.suite.data_version,
            split_version=config.suite.split_version,
            feature_version=config.suite.feature_version,
            output_sha256=verified.output_sha256,
            transfer_sha256=transfer,
            task_count=verified.task_count,
            file_count=len(verified.files),
            formal_performance_claim=(
                verified.aggregate.get("formal_performance_claim") is True
            ),
            registered_relative_root=relative_root,
            imported_at=imported_at,
        )
        record_path = self._records / f"{record.import_id}.json"
        destination = self._root / relative_root
        if record_path.exists():
            existing = _load_record(record_path)
            _assert_reimport_context(existing, record)
            self._verify_registered_bytes(existing)
            return existing
        if destination.exists():
            raise ValueError("unregistered A100 suite destination already exists")

        temporary = self._runs / f".tmp-{uuid4().hex[:12]}"
        try:
            shutil.copytree(source, temporary, symlinks=False)
            copied_files = _collect_files(temporary)
            if copied_files != verified.files:
                raise ValueError("A100 suite bytes changed during managed import")
            temporary.replace(destination)
            try:
                _write_record_atomic(record_path, record)
            except BaseException:
                shutil.rmtree(destination)
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return record

    def resolve(self, import_id: str) -> RegisteredA100Suite:
        """Resolve one accepted suite and recheck every managed byte."""

        normalized = _validate_sha256(import_id, name="import_id")
        record_path = self._records / f"{normalized}.json"
        if not record_path.is_file() or record_path.is_symlink():
            raise KeyError(f"unknown A100 suite import: {normalized}")
        record = _load_record(record_path)
        output_root = self._verify_registered_bytes(record)
        return RegisteredA100Suite(record=record, output_root=output_root)

    def list_tasks(self, import_id: str) -> tuple[ImportedA100TaskRecord, ...]:
        """Return the verified task matrix without reading model or metric payloads."""

        registered = self.resolve(import_id)
        records = tuple(
            _task_record(
                registered.output_root,
                manifest_path,
                suite=registered.record,
            )
            for manifest_path in sorted(
                registered.output_root.glob(
                    "cutoff-*/*/seed-*/run_manifest.json"
                )
            )
        )
        keys = {
            (record.cutoff_cycle, record.model_name, record.seed)
            for record in records
        }
        run_ids = {record.run_id for record in records}
        if (
            len(records) != registered.record.task_count
            or len(keys) != len(records)
            or len(run_ids) != len(records)
        ):
            raise ValueError("registered A100 task catalog is incomplete or duplicated")
        return records

    def _verify_registered_bytes(self, record: ImportedA100SuiteRecord) -> Path:
        output_root = _validated_directory(
            self._root / record.registered_relative_root
        )
        files = _collect_files(output_root)
        if _output_sha256(files) != record.output_sha256:
            raise ValueError("registered A100 suite output SHA-256 does not match")
        if len(files) != record.file_count:
            raise ValueError("registered A100 suite file count does not match")
        return output_root


def _verify_suite(
    root: Path,
    *,
    config: MatrThreeBatchRunConfig,
    expected_source_commit: str,
) -> _VerifiedSuite:
    files = _collect_files(root)
    paths = {item.relative_path for item in files}
    missing_root = _ROOT_EVIDENCE - paths
    if missing_root:
        raise ValueError(f"A100 suite root evidence is missing: {sorted(missing_root)}")

    expected_keys = {
        (cutoff, model.name, seed)
        for cutoff in config.suite.cutoffs
        for model in config.suite.models
        for seed in config.suite.seeds
    }
    manifest_paths = tuple(sorted(root.glob("cutoff-*/*/seed-*/run_manifest.json")))
    actual_keys: set[tuple[int, str, int]] = set()
    input_hashes: set[str] = set()
    run_ids: set[str] = set()
    for manifest_path in manifest_paths:
        cutoff, model, seed = _task_key(root, manifest_path)
        key = (cutoff, model, seed)
        if key in actual_keys:
            raise ValueError("A100 suite contains a duplicate task matrix entry")
        actual_keys.add(key)
        run_id, input_hash = _verify_task(
            manifest_path.parent,
            key=key,
            config=config,
            expected_source_commit=expected_source_commit,
        )
        if run_id in run_ids:
            raise ValueError("A100 suite task run_id values must be unique")
        run_ids.add(run_id)
        input_hashes.add(input_hash)
    if actual_keys != expected_keys:
        raise ValueError("A100 suite task matrix is incomplete or unexpected")
    if len(input_hashes) != 1:
        raise ValueError("A100 suite tasks do not share one input bundle")
    task_prefixes = {
        f"cutoff-{cutoff}/{model}/seed-{seed}/"
        for cutoff, model, seed in expected_keys
    }
    for item in files:
        relative = item.relative_path
        if relative in _ROOT_EVIDENCE or relative.startswith("mlruns/"):
            continue
        if not any(relative.startswith(prefix) for prefix in task_prefixes):
            raise ValueError("A100 suite contains unregistered root evidence")

    aggregate = _strict_json(root / "aggregate_metrics.json")
    _verify_aggregate(aggregate, config=config, expected_keys=expected_keys)
    _verify_metrics_csv(root / "metrics_test.csv", expected_keys=expected_keys)
    return _VerifiedSuite(
        config=config,
        source_commit=expected_source_commit,
        input_bundle_sha256=next(iter(input_hashes)),
        aggregate=aggregate,
        files=files,
        output_sha256=_output_sha256(files),
        task_count=len(expected_keys),
    )


def _verify_task(
    task_root: Path,
    *,
    key: tuple[int, str, int],
    config: MatrThreeBatchRunConfig,
    expected_source_commit: str,
) -> tuple[str, str]:
    manifest = _strict_json(task_root / "run_manifest.json")
    if manifest.get("schema_version") != "completed-run-v2":
        raise ValueError("A100 task completed manifest schema is unsupported")
    cutoff, model, seed = key
    expected_context = {
        "dataset_id": "MATR",
        "target": PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value,
        "model_name": model,
        "cutoff_cycle": cutoff,
        "seed": seed,
        "config_sha256": config.suite.config_sha256,
        "source_commit": expected_source_commit,
        "data_version": config.suite.data_version,
        "split_version": config.suite.split_version,
        "feature_version": config.suite.feature_version,
    }
    for name, expected in expected_context.items():
        if manifest.get(name) != expected:
            if name == "source_commit":
                raise ValueError("A100 task source commit does not match")
            raise ValueError(f"A100 task {name} does not match the approved config")
    run_id = manifest.get("run_id")
    input_bundle_sha256 = manifest.get("input_bundle_sha256")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A100 task run_id is invalid")
    input_hash = _validate_sha256(
        input_bundle_sha256,
        name="input_bundle_sha256",
    )

    config_resolved = _strict_json(task_root / "config_resolved.json")
    context = {
        "run_id": run_id,
        **expected_context,
        "input_bundle_sha256": input_hash,
    }
    if config_resolved != context:
        raise ValueError("A100 task resolved config does not match its manifest")
    if manifest.get("context_sha256") != sha256_canonical(context):
        raise ValueError("A100 task context SHA-256 does not match")

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("A100 task file inventory is invalid")
    indexed: dict[str, A100SuiteFile] = {}
    for raw in rows:
        item = A100SuiteFile.model_validate(raw)
        if item.relative_path in indexed:
            raise ValueError("A100 task file inventory contains duplicates")
        indexed[item.relative_path] = item
    actual = {
        path.relative_to(task_root).as_posix()
        for path in task_root.rglob("*")
        if path.is_file()
        and path.name != "run_manifest.json"
        and "checkpoints" not in path.relative_to(task_root).parts
    }
    if set(indexed) != actual:
        raise ValueError("A100 task file inventory does not match task contents")
    if not _TASK_EVIDENCE.issubset(indexed):
        raise ValueError("A100 task required evidence is missing")
    if not any(path.startswith("artifacts/") for path in indexed):
        raise ValueError("A100 task safe model artifact is missing")
    if not any(path.startswith("plots/") for path in indexed):
        raise ValueError("A100 task plot evidence is missing")
    for relative, item in indexed.items():
        path = task_root / relative
        if path.stat().st_size != item.size_bytes:
            raise ValueError("A100 task file size does not match its manifest")
        if _sha256_file(path) != item.sha256:
            raise ValueError("A100 task file SHA-256 does not match its manifest")

    metrics = _strict_json(task_root / "metrics_test.json")
    for name, expected in (
        ("model", model),
        ("cutoff_cycle", cutoff),
        ("seed", seed),
        ("target", PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value),
    ):
        if metrics.get(name) != expected:
            raise ValueError(f"A100 task metrics {name} does not match")
    return run_id, input_hash


def _task_record(
    root: Path,
    manifest_path: Path,
    *,
    suite: ImportedA100SuiteRecord,
) -> ImportedA100TaskRecord:
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != "completed-run-v2":
        raise ValueError("registered A100 task manifest schema is unsupported")
    cutoff, model, seed = _task_key(root, manifest_path)
    context = {
        "run_id": manifest.get("run_id"),
        "dataset_id": manifest.get("dataset_id"),
        "target": manifest.get("target"),
        "model_name": manifest.get("model_name"),
        "cutoff_cycle": manifest.get("cutoff_cycle"),
        "seed": manifest.get("seed"),
        "config_sha256": manifest.get("config_sha256"),
        "input_bundle_sha256": manifest.get("input_bundle_sha256"),
        "source_commit": manifest.get("source_commit"),
        "data_version": manifest.get("data_version"),
        "split_version": manifest.get("split_version"),
        "feature_version": manifest.get("feature_version"),
    }
    expected = {
        "dataset_id": suite.dataset_id,
        "target": suite.target,
        "model_name": model,
        "cutoff_cycle": cutoff,
        "seed": seed,
        "config_sha256": suite.config_sha256,
        "input_bundle_sha256": suite.input_bundle_sha256,
        "source_commit": suite.source_commit,
        "data_version": suite.data_version,
        "split_version": suite.split_version,
        "feature_version": suite.feature_version,
    }
    if any(context.get(name) != value for name, value in expected.items()):
        raise ValueError("registered A100 task context does not match its suite")
    if manifest.get("context_sha256") != sha256_canonical(context):
        raise ValueError("registered A100 task context SHA-256 does not match")
    if _strict_json(manifest_path.parent / "config_resolved.json") != context:
        raise ValueError("registered A100 task resolved config does not match")
    context_sha256 = _validate_sha256(
        manifest.get("context_sha256"),
        name="context_sha256",
    )
    try:
        completed_at = TypeAdapter(datetime).validate_python(
            manifest.get("completed_at")
        )
    except ValidationError as exc:
        raise ValueError("registered A100 task completion time is invalid") from exc
    return ImportedA100TaskRecord(
        run_id=context["run_id"],
        dataset_id=context["dataset_id"],
        target=context["target"],
        model_name=model,
        cutoff_cycle=cutoff,
        seed=seed,
        config_sha256=context["config_sha256"],
        input_bundle_sha256=context["input_bundle_sha256"],
        source_commit=context["source_commit"],
        data_version=context["data_version"],
        split_version=context["split_version"],
        feature_version=context["feature_version"],
        context_sha256=context_sha256,
        task_relative_root=manifest_path.parent.relative_to(root).as_posix(),
        completed_at=completed_at,
    )


def _verify_aggregate(
    aggregate: dict[str, Any],
    *,
    config: MatrThreeBatchRunConfig,
    expected_keys: set[tuple[int, str, int]],
) -> None:
    if aggregate.get("schema_version") != "matr-aggregate-metrics-v1":
        raise ValueError("A100 aggregate metrics schema is unsupported")
    if aggregate.get("mode") != config.mode:
        raise ValueError("A100 aggregate mode does not match the approved config")
    if aggregate.get("run_count") != len(expected_keys):
        raise ValueError("A100 aggregate run count does not match the task matrix")
    if aggregate.get("expected_seed_count") != len(config.suite.seeds):
        raise ValueError("A100 aggregate seed count does not match")
    failures = aggregate.get("failed_metric_rows")
    if not isinstance(failures, list) or failures:
        raise ValueError("A100 suite contains failed or invalid metric rows")
    formal = aggregate.get("formal_performance_claim")
    if config.mode == "final" and formal is not True:
        raise ValueError("complete final A100 suite must be marked formal")
    if config.mode == "smoke" and formal is not False:
        raise ValueError("Smoke A100 suite cannot be marked formal")

    summaries = aggregate.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("A100 aggregate summaries are invalid")
    expected_groups = {
        (cutoff, model.name)
        for cutoff in config.suite.cutoffs
        for model in config.suite.models
    }
    actual_groups: set[tuple[int, str]] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("A100 aggregate summary row is invalid")
        model = summary.get("model")
        cutoff = summary.get("cutoff_cycle")
        if not isinstance(model, str) or not isinstance(cutoff, int):
            raise ValueError("A100 aggregate summary identity is invalid")
        actual_groups.add((cutoff, model))
        if (
            summary.get("target")
            != PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value
            or summary.get("run_count") != len(config.suite.seeds)
            or summary.get("seeds") != list(config.suite.seeds)
            or summary.get("complete_seed_matrix") is not True
        ):
            raise ValueError("A100 aggregate summary context is incomplete")
    if actual_groups != expected_groups or len(summaries) != len(expected_groups):
        raise ValueError("A100 aggregate summary matrix is incomplete")


def _verify_metrics_csv(
    path: Path,
    *,
    expected_keys: set[tuple[int, str, int]],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("A100 suite metrics CSV is invalid") from exc
    keys: set[tuple[int, str, int]] = set()
    for row in rows:
        try:
            key = (
                int(row["cutoff_cycle"]),
                row["model"],
                int(row["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A100 suite metrics CSV identity is invalid") from exc
        if row.get("target") != PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value:
            raise ValueError("A100 suite metrics CSV target is invalid")
        keys.add(key)
    if keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("A100 suite metrics CSV matrix is incomplete")


def _collect_files(root: Path) -> tuple[A100SuiteFile, ...]:
    files: list[A100SuiteFile] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            if directory.is_symlink():
                raise ValueError("symbolic links are forbidden in A100 suite output")
            _validate_relative_path(directory.relative_to(root), is_file=False)
        for name in file_names:
            path = current / name
            relative = path.relative_to(root)
            _validate_relative_path(relative, is_file=True)
            if path.is_symlink() or not path.is_file():
                raise ValueError("A100 suite output must contain regular files only")
            _scan_for_secrets(path)
            files.append(
                A100SuiteFile(
                    relative_path=relative.as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=_sha256_file(path),
                )
            )
    files.sort(key=lambda item: item.relative_path)
    if not files:
        raise ValueError("A100 suite output is empty")
    return tuple(files)


def _validate_relative_path(relative: Path, *, is_file: bool) -> None:
    lowered = tuple(part.lower() for part in relative.parts)
    if any(part.startswith(".env") for part in lowered) or any(
        marker in part
        for part in lowered
        for marker in ("credential", "private-key", "secret", "token")
    ):
        raise ValueError("forbidden secret-bearing path in A100 suite output")
    if not is_file:
        return
    suffix = relative.suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        raise ValueError("forbidden executable model format in A100 suite output")
    if suffix in _SAFE_SUFFIXES:
        return
    if relative.parts and relative.parts[0] == "mlruns" and suffix in _MLFLOW_SAFE_SUFFIXES:
        return
    raise ValueError("forbidden file format in A100 suite output")


def _scan_for_secrets(path: Path) -> None:
    if path.suffix.lower() in {".png", ".safetensors", ".ubj"}:
        return
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("unreviewed large text file in A100 suite output")
    payload = path.read_bytes()
    if b"-----BEGIN PRIVATE KEY-----" in payload or _SECRET_PATTERN.search(payload):
        raise ValueError("forbidden secret material in A100 suite output")


def _task_key(root: Path, manifest_path: Path) -> tuple[int, str, int]:
    relative = manifest_path.relative_to(root)
    parts = relative.parts
    if len(parts) != 4:
        raise ValueError("A100 task manifest path is invalid")
    try:
        cutoff = int(parts[0].removeprefix("cutoff-"))
        seed = int(parts[2].removeprefix("seed-"))
    except ValueError as exc:
        raise ValueError("A100 task path contains an invalid cutoff or seed") from exc
    if parts[0] != f"cutoff-{cutoff}" or parts[2] != f"seed-{seed}":
        raise ValueError("A100 task path is not canonical")
    return cutoff, parts[1], seed


def _load_run_config(path: Path) -> MatrThreeBatchRunConfig:
    return MatrThreeBatchRunConfig.model_validate(_strict_json(path))


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict JSON file: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError("A100 suite JSON evidence must contain an object")
    _reject_nonfinite(payload)
    return payload


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("A100 suite JSON cannot contain non-finite values")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _output_sha256(files: tuple[A100SuiteFile, ...]) -> str:
    return sha256_canonical(
        {
            "schema_version": "a100-suite-byte-tree-v1",
            "files": [item.model_dump(mode="json") for item in files],
        }
    )


def _validated_directory(path: Path) -> Path:
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("A100 suite output directory must exist") from exc
    if path.is_symlink() or not root.is_dir():
        raise ValueError("A100 suite output must be a regular directory")
    return root


def _validated_regular_file(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("transfer archive must exist") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("transfer archive must be a regular file")
    return resolved


def _validate_commit(value: str) -> str:
    if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
        raise ValueError("expected_source_commit must be a lowercase Git commit")
    return value


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_record_atomic(path: Path, record: ImportedA100SuiteRecord) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_record(path: Path) -> ImportedA100SuiteRecord:
    return ImportedA100SuiteRecord.model_validate(_strict_json(path))


def _assert_reimport_context(
    existing: ImportedA100SuiteRecord,
    candidate: ImportedA100SuiteRecord,
) -> None:
    immutable_fields = (
        "import_id",
        "mode",
        "dataset_id",
        "target",
        "source_commit",
        "config_sha256",
        "input_bundle_sha256",
        "data_version",
        "split_version",
        "feature_version",
        "output_sha256",
        "transfer_sha256",
        "task_count",
        "file_count",
        "formal_performance_claim",
        "registered_relative_root",
    )
    if any(
        getattr(existing, field) != getattr(candidate, field)
        for field in immutable_fields
    ):
        raise ValueError("A100 suite re-import context does not match its registry record")


__all__ = [
    "A100SuiteRunImporter",
    "ImportedA100SuiteRecord",
    "ImportedA100TaskRecord",
    "RegisteredA100Suite",
]
