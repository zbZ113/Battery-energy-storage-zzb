"""Closed-world lifecycle runner shared by PBT and MAGNet A100 jobs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast

import torch

from quanxin_life.core import PredictionTarget, TrainingReadableSplit, sha256_canonical
from quanxin_life.data.model_views.schemas import ModelViewArtifact, ModelViewManifest
from quanxin_life.training.adapters.magnet import MAGNetAdapter
from quanxin_life.training.adapters.pbt import PBTAdapter
from quanxin_life.training.checkpoint import CheckpointContext
from quanxin_life.training.config import ModelTrainingConfig
from quanxin_life.training.engine import EpochMetrics, TrainingEngine, TrainingRunStatus

_MODELS = ("pbt", "magnet")
_STAGES = ("smoke", "selection", "freeze-selection", "final", "collect")
_PBT_EXPERIMENTS = (
    "matr_in_domain",
    "hust_zero_shot",
    "hust_5_shot",
    "hust_10_shot",
)
_MAGNET_EXPERIMENTS = ("unseen_condition",)


@dataclass(frozen=True)
class ViewTask:
    model: str
    root: Path
    manifest: ModelViewManifest
    view_sha256: str
    experiment: str
    split_sha256: str


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class _EngineTask:
    def __init__(self, task: Any, evidence_path: Path) -> None:
        self._task = task
        self._evidence_path = evidence_path
        self.model = task.model
        self.optimizer = task.optimizer
        self.scheduler = task.scheduler
        self.scaler = getattr(task, "scaler", None)
        self.train_history: dict[int, EpochMetrics] = {}

    def train_epoch(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        metrics = cast(EpochMetrics, self._task.train_epoch(epoch, device=device))
        self.train_history[epoch] = metrics
        with self._evidence_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(
                    {"epoch": epoch, "loss": metrics.loss, "metrics": metrics.metrics},
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return metrics

    def validate(self, epoch: int, *, device: torch.device) -> EpochMetrics:
        del device
        result = self._task.validate(epoch, split=TrainingReadableSplit.VALIDATION)
        return EpochMetrics(loss=result.loss, metrics=result.metrics)


def run_stage(
    *,
    project_root: Path,
    models: tuple[str, ...],
    stage: str,
    seeds: tuple[int, ...],
    device: str,
    enforce_formal_matrix: bool = True,
    recovery_seed42: bool = False,
) -> dict[str, object]:
    root = project_root.resolve(strict=True)
    _validate_environment(stage=stage, device=device)
    if recovery_seed42 and seeds != (42,):
        raise ValueError("seed42 recovery requires the exact seed 42")
    lifecycle_root = (
        root / "runs" / "a100" / "pbt_magnet_recovery_seed42_v2"
        if recovery_seed42
        else root / "runs" / "a100" / "pbt_magnet"
    )
    if stage == "freeze-selection":
        return _freeze_selection(root, lifecycle_root, models)
    if stage == "collect":
        return _collect(lifecycle_root, models, recovery_seed42=recovery_seed42)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and selected_device.index not in {None, 0}:
        raise ValueError("masked A100 process device must be cuda:0")
    if not enforce_formal_matrix and device != "cpu":
        raise ValueError("partial matrices are permitted only for local CPU verification")
    if (
        stage == "final"
        and enforce_formal_matrix
        and not recovery_seed42
        and seeds != (38, 39, 40, 41, 42)
    ):
        raise ValueError("formal final training requires the exact seeds 38,39,40,41,42")
    tasks = _discover_view_tasks(
        root, models, formal=stage == "final" and enforce_formal_matrix
    )
    results: list[dict[str, object]] = []
    for model in models:
        config_path = root / "configs" / "training" / model / f"{stage}.json"
        payload = _read_json(config_path)
        for view in (item for item in tasks if item.model == model):
            for seed in seeds:
                results.append(
                    _run_one(
                        root=root,
                        lifecycle_root=lifecycle_root,
                        view=view,
                        config_path=config_path,
                        config_payload=payload,
                        stage=stage,
                        seed=seed,
                        device=selected_device,
                        recovery_seed42=recovery_seed42,
                    )
                )
    if not results:
        raise ValueError("no PBT/MAGNet tasks were selected")
    return {
        "status": "COMPLETED",
        "stage": stage,
        "execution_mode": "recovery_seed42" if recovery_seed42 else "formal",
        "run_namespace": lifecycle_root.name,
        "objective_profile": (
            "label_only_train_log_zscore_v1"
            if view.model == "pbt"
            else "upstream_mldg_observed_targets_v1"
        ),
        "runs": results,
    }


def _run_one(
    *,
    root: Path,
    lifecycle_root: Path,
    view: ViewTask,
    config_path: Path,
    config_payload: dict[str, object],
    stage: str,
    seed: int,
    device: torch.device,
    recovery_seed42: bool,
) -> dict[str, object]:
    config_sha256 = _sha256(config_path)
    selection_sha256 = (
        _selection_sha256(lifecycle_root) if stage == "final" else None
    )
    run_id = _run_id(view, stage=stage, seed=seed)
    run_root = lifecycle_root / stage / view.model / run_id
    run_manifest = {
        "schema_version": "pbt-magnet-run-manifest-v1",
        "run_id": run_id,
        "model": view.model,
        "stage": stage,
        "seed": seed,
        "device": str(device),
        "view_root": view.root.relative_to(root).as_posix(),
        "view_sha256": view.view_sha256,
        "split_sha256": view.split_sha256,
        "training_entity_ids_sha256": _training_entity_ids_sha256(view),
        "experiment": view.experiment,
        "cutoff_cycle": view.manifest.cutoff_cycle,
        "execution_mode": "recovery_seed42" if recovery_seed42 else "formal",
        "run_namespace": lifecycle_root.name,
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": config_sha256,
        "selection_manifest_sha256": selection_sha256,
        "source_commit": _source_commit(root),
        "test_policy": "READ_ONCE_AFTER_BEST_AND_SELECTION_FREEZE"
        if stage == "final"
        else "TEST_FORBIDDEN",
    }
    context_sha256 = sha256_canonical(run_manifest)
    success_path = run_root / "SUCCESS.json"
    if success_path.is_file() and not success_path.is_symlink():
        success = _read_json(success_path)
        if success.get("context_sha256") != context_sha256:
            raise ValueError(f"completed run context changed: {run_id}")
        print(f"run_id={run_id} status=SKIPPED_COMPLETED")
        return {"run_id": run_id, "status": "SKIPPED_COMPLETED"}

    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "run_manifest.json", run_manifest)
    stdout_path = run_root / "stdout.log"
    with stdout_path.open("a", encoding="utf-8", newline="") as log:
        tee_out = _Tee(sys.stdout, log)
        tee_error = _Tee(sys.stderr, log)
        with redirect_stdout(tee_out), redirect_stderr(tee_error):
            print(f"run_id={run_id} status=STARTED")
            prepared = _prepare_view(run_root, view)
            resolved = _resolved_config(view, config_payload, seed=seed)
            effective_batch_size = int(resolved.effective_batch_size)
            while True:
                adapter, task = _build_task(
                    view.model, prepared, resolved, device=device, seed=seed
                )
                engine_task = _EngineTask(task, run_root / "task_metrics.jsonl")
                engine = TrainingEngine(
                    task=engine_task,
                    context=_checkpoint_context(
                        view=view,
                        resolved=resolved,
                        run_id=run_id,
                        config_sha256=config_sha256,
                        source_commit=str(run_manifest["source_commit"]),
                    ),
                    config=ModelTrainingConfig(
                        name=view.model,
                        max_epochs=int(resolved.max_epochs),
                        validation_interval=5,
                        early_stopping_patience=10,
                        learning_rate=float(resolved.learning_rate),
                        minimum_learning_rate=min(1e-6, float(resolved.learning_rate)),
                    ),
                    run_directory=run_root,
                    device=device,
                )
                try:
                    result = engine.run()
                    break
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if not _is_out_of_memory(exc):
                        raise
                    micro_batch_size, accumulation = _next_oom_batch(
                        effective_batch_size=effective_batch_size,
                        micro_batch_size=int(resolved.micro_batch_size),
                    )
                    resolved.micro_batch_size = micro_batch_size
                    resolved.gradient_accumulation_steps = accumulation
                    event = {
                        "effective_batch_size": effective_batch_size,
                        "gradient_accumulation_steps": accumulation,
                        "micro_batch_size": micro_batch_size,
                    }
                    with (run_root / "oom_backoff.jsonl").open(
                        "a", encoding="utf-8", newline=""
                    ) as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(
                        "OOM_BACKOFF "
                        f"micro_batch_size={micro_batch_size} "
                        f"gradient_accumulation_steps={accumulation} "
                        f"effective_batch_size={effective_batch_size}"
                    )
                    task.optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if result.status not in {
                TrainingRunStatus.COMPLETED,
                TrainingRunStatus.EARLY_STOPPED,
                TrainingRunStatus.SKIPPED_COMPLETED,
            }:
                raise RuntimeError(f"training did not complete: {result.status}")
            _publish_engine_evidence(run_root, engine_task.train_history)
            _publish_checkpoints(run_root)
            test_read = False
            if stage == "final":
                test_read = True
                seen_metrics = (
                    {
                        "seen_condition_query_loss": _training_metric_at_epoch(
                            run_root, "train_query_loss", result.best_epoch
                        )
                    }
                    if view.model == "magnet"
                    else {}
                )
                _publish_test_evidence(
                    run_root, view.model, task, training_metrics=seen_metrics
                )
                _export_safe_model(run_root, view.model, adapter, task)
            success = {
                "schema_version": "pbt-magnet-success-v1",
                "run_id": run_id,
                "context_sha256": context_sha256,
                "status": "SUCCESS",
                "test_read": test_read,
                "best_epoch": result.best_epoch,
                "best_metric": result.best_metric,
                "finished_at": _utc_now(),
            }
            _write_json(success_path, success)
            print(f"run_id={run_id} status=SUCCESS")
    return {"run_id": run_id, "status": "SUCCESS"}


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _next_oom_batch(
    *, effective_batch_size: int, micro_batch_size: int
) -> tuple[int, int]:
    for candidate in range(micro_batch_size - 1, 1, -1):
        if effective_batch_size % candidate == 0:
            return candidate, effective_batch_size // candidate
    raise RuntimeError(
        "OOM persists at the minimum safe micro-batch; data, epochs, seeds and model "
        "size will not be reduced automatically"
    )


def _build_task(
    model: str,
    prepared: tuple[Path, ModelViewManifest],
    resolved: SimpleNamespace,
    *,
    device: torch.device,
    seed: int,
) -> tuple[Any, Any]:
    view_root, manifest = prepared
    adapter: Any = (
        PBTAdapter(view_root=view_root)
        if model == "pbt"
        else MAGNetAdapter(view_root=view_root)
    )
    adapter.build_model(resolved)
    adapter.load_view(manifest)
    task = adapter.attach_training(resolved, device=device, seed=seed)
    task.micro_batch_size = int(resolved.micro_batch_size)
    task.gradient_accumulation_steps = int(resolved.gradient_accumulation_steps)
    return adapter, task


def _checkpoint_context(
    *,
    view: ViewTask,
    resolved: SimpleNamespace,
    run_id: str,
    config_sha256: str,
    source_commit: str,
) -> CheckpointContext:
    if view.model == "pbt":
        target = PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        cutoff = int(resolved.cutoff_cycle)
        upstream_commit = PBTAdapter.upstream_commit
        adapter_version = PBTAdapter.adapter_version
    else:
        target = PredictionTarget.MAGNET_QD_ED_TRAJECTORY
        cutoff = None
        upstream_commit = MAGNetAdapter.upstream_commit
        adapter_version = MAGNetAdapter.adapter_version
    return CheckpointContext(
        run_id=run_id,
        dataset_id="MATR_HUST" if view.model == "pbt" else "MAGNET_MULTICONDITION",
        target=target,
        model_name=view.model,
        cutoff_cycle=cutoff,
        seed=int(resolved.seed),
        config_sha256=config_sha256,
        input_bundle_sha256=view.manifest.canonical_sha256,
        data_version="pbt-magnet-canonical-v1",
        split_version=f"{view.experiment}-v1",
        feature_version=view.manifest.view_version,
        source_commit=source_commit,
        adapter_version=adapter_version,
        model_view_sha256=view.view_sha256,
        effective_batch_size=int(resolved.effective_batch_size),
        selection_metric_name=(
            "validation_mae" if view.model == "pbt" else "validation_observed_mae"
        ),
        upstream_commit=upstream_commit,
    )


def _resolved_config(
    view: ViewTask, payload: dict[str, object], *, seed: int
) -> SimpleNamespace:
    model = payload.get("model")
    optimization = payload.get("optimization")
    if not isinstance(model, dict) or not isinstance(optimization, dict):
        raise ValueError("training config requires model and optimization objects")
    merged: dict[str, object] = {**payload, **model, **optimization}
    merged.update(
        model_family=view.model,
        seed=seed,
        cutoff_cycle=view.manifest.cutoff_cycle,
        prediction_horizon=model.get("prediction_horizon"),
    )
    if view.model == "pbt":
        if view.manifest.cutoff_cycle is None:
            raise ValueError("PBT model view requires a cutoff cycle")
        merged["early_cycle_threshold"] = view.manifest.cutoff_cycle
    if view.model == "pbt" and view.manifest.cutoff_cycle is not None:
        merged["early_cycle_threshold"] = view.manifest.cutoff_cycle
    merged["e_layers"] = model.get("encoder_layers", model.get("e_layers", 2))
    merged["d_layers"] = model.get("decoder_layers", model.get("d_layers", 2))
    merged["pred_len"] = model.get("prediction_horizon", model.get("pred_len", 500))
    for required in (
        "learning_rate",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "max_epochs",
    ):
        if required not in merged:
            raise ValueError(f"training config is missing {required}")
    effective_batch_size = _positive_int(merged["effective_batch_size"], "effective_batch_size")
    micro_batch_size = _positive_int(merged["micro_batch_size"], "micro_batch_size")
    accumulation = _positive_int(
        merged["gradient_accumulation_steps"], "gradient_accumulation_steps"
    )
    max_epochs = _positive_int(merged["max_epochs"], "max_epochs")
    if max_epochs < 5:
        raise ValueError("PBT/MAGNet stages require at least five epochs for validation")
    if effective_batch_size != micro_batch_size * accumulation:
        raise ValueError("effective batch size differs from micro-batch * accumulation")
    return SimpleNamespace(**merged)


def _discover_view_tasks(
    root: Path, models: tuple[str, ...], *, formal: bool
) -> tuple[ViewTask, ...]:
    view_root = root / "data" / "model_views"
    if not view_root.is_dir():
        raise ValueError("data/model_views is missing")
    tasks: list[ViewTask] = []
    for path in sorted(view_root.rglob("manifest.json")):
        if path.is_symlink() or not path.is_file():
            continue
        manifest = ModelViewManifest.model_validate(_read_json(path))
        model = (
            "pbt"
            if manifest.view_id == "pbt_multidomain_early_life"
            else "magnet"
            if manifest.view_id == "magnet_multicondition_qd_ed"
            else None
        )
        if model is None or model not in models:
            continue
        directory = path.parent.resolve(strict=True)
        view_sha = _verify_view(directory, manifest)
        experiments = _experiments(directory, manifest, model=model)
        for experiment, split_sha in experiments:
            tasks.append(
                ViewTask(
                    model=model,
                    root=directory,
                    manifest=manifest,
                    view_sha256=view_sha,
                    experiment=experiment,
                    split_sha256=split_sha,
                )
            )
    if formal:
        pbt = [item for item in tasks if item.model == "pbt"]
        magnet = [item for item in tasks if item.model == "magnet"]
        if "pbt" in models:
            matrix = {(item.manifest.cutoff_cycle, item.experiment) for item in pbt}
            expected = {
                (cutoff, experiment)
                for cutoff in (20, 50, 100, 150)
                for experiment in _PBT_EXPERIMENTS
            }
            if matrix != expected or len(pbt) != 16:
                raise ValueError(
                    "formal PBT views do not cover the exact 4 cutoff x 4 experiment matrix"
                )
        if "magnet" in models and (
            len(magnet) != 1
            or {item.experiment for item in magnet} != set(_MAGNET_EXPERIMENTS)
        ):
            raise ValueError(
                "formal MAGNet requires one condition-isolated unseen meta-test task; "
                "seen-condition evidence comes from support/query training episodes"
            )
    if not tasks:
        raise ValueError("no committed PBT/MAGNet model views were found")
    return tuple(tasks)


def _experiments(
    root: Path, manifest: ModelViewManifest, *, model: str
) -> tuple[tuple[str, str], ...]:
    split_files = sorted(root.glob("split_*.json"))
    if split_files:
        return tuple(
            (path.stem.removeprefix("split_"), _sha256(path)) for path in split_files
        )
    if model == "pbt":
        return (("default", manifest.split_sha256),)
    metadata = _metadata_payload(root, manifest)
    splits = metadata.get("splits")
    seen = metadata.get("seen")
    if not isinstance(splits, list) or not isinstance(seen, list) or len(splits) != len(seen):
        raise ValueError("MAGNet metadata cannot determine condition experiment")
    test_seen = [bool(flag) for split, flag in zip(splits, seen, strict=True) if split == "test"]
    if not test_seen:
        raise ValueError("MAGNet condition view has no test condition")
    if all(test_seen):
        experiment = "seen_condition"
    elif not any(test_seen):
        experiment = "unseen_condition"
    else:
        experiment = "mixed_condition"
    return ((experiment, manifest.split_sha256),)


def _prepare_view(
    run_root: Path, view: ViewTask
) -> tuple[Path, ModelViewManifest]:
    if view.model != "pbt" or view.experiment == "default":
        return view.root, view.manifest
    split_path = view.root / f"split_{view.experiment}.json"
    if not split_path.is_file() or split_path.is_symlink():
        raise ValueError("PBT experiment split manifest is missing")
    split_payload = _read_json(split_path)
    assignments = split_payload.get(
        "membership", split_payload.get("assignments", split_payload)
    )
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("PBT split manifest requires cell-to-split assignments")
    metadata = _metadata_payload(view.root, view.manifest)
    entity_ids = metadata.get("entity_ids")
    if not isinstance(entity_ids, list) or set(map(str, entity_ids)) != set(map(str, assignments)):
        raise ValueError("PBT split assignments do not exactly cover view cells")
    selected_splits = [str(assignments[str(entity_id)]) for entity_id in entity_ids]
    allowed_splits = {"train", "validation", "calibration", "test", "excluded"}
    if any(value not in allowed_splits for value in selected_splits):
        raise ValueError("PBT split manifest contains an unknown split")
    selected_metadata = dict(metadata)
    selected_metadata["splits"] = selected_splits
    domains = selected_metadata.get("domains")
    if not isinstance(domains, list) or len(domains) != len(selected_splits):
        raise ValueError("PBT metadata domains do not align with split assignments")
    selected_metadata["seen"] = [
        str(domain) == "MATR" or split in {"train", "validation", "calibration"}
        for domain, split in zip(domains, selected_splits, strict=True)
    ]
    selected_metadata.pop("splits_by_experiment", None)
    prepared = run_root / "input_view"
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir()
    tensor_artifacts = [
        artifact
        for artifact in view.manifest.artifacts
        if artifact.relative_path.endswith(".safetensors")
    ]
    for artifact in tensor_artifacts:
        shutil.copy2(view.root / artifact.relative_path, prepared / artifact.relative_path)
    metadata_path = prepared / "metadata.json"
    _write_json(metadata_path, selected_metadata)
    artifacts = tuple(
        sorted(
            [
                *tensor_artifacts,
                ModelViewArtifact(
                    relative_path=metadata_path.name,
                    size_bytes=metadata_path.stat().st_size,
                    sha256=_sha256(metadata_path),
                ),
            ],
            key=lambda item: item.relative_path,
        )
    )
    training_entity_ids = sorted(
        str(entity_id)
        for entity_id, split in zip(entity_ids, selected_splits, strict=True)
        if split == "train"
    )
    if not training_entity_ids:
        raise ValueError("PBT prepared experiment has no training cells")
    prepared_manifest = view.manifest.model_copy(
        update={
            "split_sha256": view.split_sha256,
            "training_entity_ids_sha256": sha256_canonical(training_entity_ids),
            "artifacts": artifacts,
        }
    )
    return prepared, prepared_manifest


def _training_entity_ids_sha256(view: ViewTask) -> str:
    if view.model != "pbt" or view.experiment == "default":
        return view.manifest.training_entity_ids_sha256
    split_payload = _read_json(view.root / f"split_{view.experiment}.json")
    assignments = split_payload.get(
        "membership", split_payload.get("assignments", split_payload)
    )
    metadata = _metadata_payload(view.root, view.manifest)
    entity_ids = metadata.get("entity_ids")
    if not isinstance(assignments, dict) or not isinstance(entity_ids, list):
        raise ValueError("PBT experiment cannot determine training cells")
    training_ids = sorted(
        str(entity_id)
        for entity_id in entity_ids
        if str(assignments.get(str(entity_id))) == "train"
    )
    if not training_ids:
        raise ValueError("PBT experiment has no training cells")
    return sha256_canonical(training_ids)


def _verify_view(root: Path, manifest: ModelViewManifest) -> str:
    committed = root / "COMMITTED"
    if not committed.is_file() or committed.is_symlink():
        raise ValueError(f"model view is not committed: {root}")
    canonical_digest = sha256_canonical(manifest.model_dump(mode="json"))
    manifest_digest = _sha256(root / "manifest.json")
    marker = committed.read_text(encoding="ascii").strip()
    if marker not in {canonical_digest, manifest_digest}:
        raise ValueError(f"model view COMMITTED digest mismatch: {root}")
    for artifact in manifest.artifacts:
        path = root / artifact.relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"model view artifact is missing: {path}")
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"model view artifact digest mismatch: {path}")
    return marker


def _metadata_payload(root: Path, manifest: ModelViewManifest) -> dict[str, object]:
    for artifact in manifest.artifacts:
        if not artifact.relative_path.endswith(".json"):
            continue
        payload = _read_json(root / artifact.relative_path)
        if "entity_ids" in payload and "splits" in payload:
            return payload
    raise ValueError("model view metadata artifact was not found")


def _publish_engine_evidence(
    run_root: Path, train_history: dict[int, EpochMetrics]
) -> None:
    persisted_history: dict[int, EpochMetrics] = {}
    task_metrics_path = run_root / "task_metrics.jsonl"
    if task_metrics_path.is_file() and not task_metrics_path.is_symlink():
        for line in task_metrics_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            epoch = payload.get("epoch")
            loss = payload.get("loss")
            metrics = payload.get("metrics")
            if (
                not isinstance(epoch, int)
                or not isinstance(loss, (int, float))
                or not isinstance(metrics, dict)
            ):
                raise ValueError("persisted task metrics contain an invalid record")
            persisted_history[epoch] = EpochMetrics(
                loss=float(loss),
                metrics={str(name): float(value) for name, value in metrics.items()},
            )
    persisted_history.update(train_history)
    train_history = persisted_history
    source = run_root / "training_log.jsonl"
    if not source.is_file():
        raise ValueError("training engine did not emit event evidence")
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for event in events:
        epoch = event.get("epoch")
        if isinstance(epoch, int) and epoch in train_history:
            event.update(
                {
                    f"train_{name}": value
                    for name, value in train_history[epoch].metrics.items()
                }
            )
    (run_root / "events.jsonl").write_text(
        "".join(json.dumps(event, allow_nan=False, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    epoch_path = run_root / "metrics_epoch.csv"
    if not epoch_path.is_file():
        raise ValueError("training engine did not emit epoch metrics")
    with epoch_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    augmented_rows: list[dict[str, object]] = []
    for row in rows:
        epoch = int(row["epoch"])
        augmented_rows.append(
            {
                **row,
                **{
                    f"train_{name}": value
                    for name, value in train_history[epoch].metrics.items()
                },
            }
        )
    fieldnames = list(dict.fromkeys(name for row in augmented_rows for name in row))
    with epoch_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmented_rows)
    resource_fields = (
        "epoch",
        "elapsed_seconds",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
    )
    with (run_root / "resource.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields)
        writer.writeheader()
        for row in augmented_rows:
            writer.writerow({name: row[name] for name in resource_fields})


def _publish_checkpoints(run_root: Path) -> None:
    checkpoint_root = run_root / "checkpoints"
    for name in ("best", "last"):
        pointer = _read_json(checkpoint_root / f"{name}.json")
        checkpoint = pointer.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ValueError(f"{name} checkpoint pointer is invalid")
        source = checkpoint_root / checkpoint
        destination = run_root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _publish_test_evidence(
    run_root: Path,
    model: str,
    task: Any,
    *,
    training_metrics: dict[str, float] | None = None,
) -> None:
    access_marker = run_root / "TEST_READ.json"
    metrics_path = run_root / "metrics_test.json"
    predictions_path = run_root / "predictions.parquet"
    if access_marker.is_file() and not access_marker.is_symlink():
        access = _read_json(access_marker)
        if access.get("status") != "COMPLETED":
            raise RuntimeError("test split was already opened but evidence did not complete")
        if (
            not metrics_path.is_file()
            or not predictions_path.is_file()
            or access.get("metrics_sha256") != _sha256(metrics_path)
            or access.get("predictions_sha256") != _sha256(predictions_path)
        ):
            raise RuntimeError("completed test evidence differs from its one-read marker")
        return
    _write_json(
        access_marker,
        {
            "schema_version": "pbt-magnet-test-read-v1",
            "status": "STARTED",
            "opened_at": _utc_now(),
        },
    )
    if model == "magnet":
        result, raw_records = task.evaluate_with_records(
            split=TrainingReadableSplit.TEST
        )
    else:
        result = task.validate(0, split=TrainingReadableSplit.TEST)
        raw_records = task.predict_records(split=TrainingReadableSplit.TEST)
    metrics: dict[str, object] = {
        "loss": result.loss,
        **result.metrics,
        **(training_metrics or {}),
    }
    records = tuple(dict(record) for record in raw_records)
    if not records:
        raise ValueError("test prediction table is empty")
    if model == "pbt":
        metrics["domain_wise"] = task.domain_metrics(records)
        if any(
            task.dataset[index]["split"] == TrainingReadableSplit.CALIBRATION.value
            for index in range(len(task.dataset))
        ):
            calibration = task.predict_records(split=TrainingReadableSplit.CALIBRATION)
            if all(
                record.get("target_semantics") == "matr_official_cycle_life"
                for record in (*records, *calibration)
            ):
                _add_conformal_evidence(records, calibration, metrics)
            else:
                metrics["conformal"] = {
                    "status": "UNAVAILABLE_DOMAIN_TARGET_MISMATCH",
                    "reason": (
                        "MATR official-EOL calibration cannot provide a coverage "
                        "guarantee for HUST observed-cycle-count targets"
                    ),
                }
    else:
        observed_targets = sorted(
            {str(record["target_name"]) for record in records}
        )
        metrics["target_observation"] = {
            "evaluated_targets": observed_targets,
            "explicitly_unobserved_targets": sorted(
                {"Qd", "Ed"} - set(observed_targets)
            ),
        }
    _write_json(metrics_path, metrics)
    _write_parquet(predictions_path, records)
    _write_json(
        access_marker,
        {
            "schema_version": "pbt-magnet-test-read-v1",
            "status": "COMPLETED",
            "metrics_sha256": _sha256(metrics_path),
            "predictions_sha256": _sha256(predictions_path),
        },
    )


def _training_metric_at_epoch(
    run_root: Path, metric_name: str, epoch: int | None
) -> float:
    if epoch is None:
        raise ValueError("best epoch is required for seen-condition training evidence")
    path = run_root / "metrics_epoch.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["epoch"]) == epoch and row.get(metric_name)]
    if len(matches) != 1:
        raise ValueError(f"training metric {metric_name} is unavailable at best epoch {epoch}")
    value = float(matches[0][metric_name])
    if not math.isfinite(value):
        raise ValueError(f"training metric {metric_name} is not finite")
    return value


def _add_conformal_evidence(
    records: tuple[dict[str, object], ...],
    calibration: tuple[dict[str, object], ...],
    metrics: dict[str, object],
    *,
    level: float = 0.9,
) -> None:
    residuals = sorted(
        _finite_float(record.get("abs_error"), "calibration abs_error")
        for record in calibration
    )
    if not residuals:
        raise ValueError("PBT conformal calibration requires observed residuals")
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * level))
    quantile = residuals[rank - 1]
    covered = 0
    for record in records:
        prediction = _finite_float(record.get("y_pred"), "test y_pred")
        target = _finite_float(record.get("y_true"), "test y_true")
        lower = prediction - quantile
        upper = prediction + quantile
        is_covered = lower <= target <= upper
        covered += int(is_covered)
        record.update(
            conformal_level=level,
            conformal_lower=lower,
            conformal_upper=upper,
            conformal_covered=is_covered,
        )
    metrics["conformal"] = {
        "level": level,
        "calibration_count": len(residuals),
        "absolute_error_quantile": quantile,
        "test_coverage": covered / len(records),
        "mean_interval_width": 2.0 * quantile,
    }


def _export_safe_model(
    run_root: Path, model: str, adapter: Any, task: Any
) -> None:
    export_root = run_root / "export"
    model_path = cast(Path, adapter.save(export_root))
    manifest = {
        "schema_version": "pbt-magnet-safe-model-v1",
        "model": model,
        "adapter_version": str(adapter.adapter_version),
        "upstream_commit": str(adapter.upstream_commit),
        "weights": model_path.name,
        "weights_sha256": _sha256(model_path),
    }
    if model == "pbt":
        manifest["target_transform"] = task.target_transform_metadata()
    _write_json(export_root / "model_manifest.json", manifest)


def _write_parquet(path: Path, records: tuple[dict[str, object], ...]) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for auditable predictions.parquet") from exc
    table = pa.Table.from_pylist(list(records))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _freeze_selection(
    root: Path, lifecycle_root: Path, models: tuple[str, ...]
) -> dict[str, object]:
    selection_root = lifecycle_root / "selection"
    entries: list[dict[str, object]] = []
    for model in models:
        successes = sorted((selection_root / model).rglob("SUCCESS.json"))
        if not successes:
            raise ValueError(f"no successful selection runs for {model}")
        config = root / "configs" / "training" / model / "selection.json"
        entries.extend(
            {
                "model": model,
                "success": path.relative_to(root).as_posix(),
                "success_sha256": _sha256(path),
                "config": config.relative_to(root).as_posix(),
                "config_sha256": _sha256(config),
            }
            for path in successes
        )
    frozen = lifecycle_root / "selection_frozen"
    manifest = frozen / "selection_manifest.json"
    marker = frozen / "COMMITTED"
    if manifest.is_file() and marker.is_file() and not marker.is_symlink():
        existing = _read_json(manifest)
        digest = _sha256(manifest)
        if marker.read_text(encoding="ascii").strip() != digest:
            raise ValueError("existing frozen selection digest mismatch")
        if existing.get("entries") != entries:
            raise ValueError("selection inputs changed after they were frozen")
        return {"status": "SKIPPED_FROZEN", "selection_manifest_sha256": digest}
    payload = {
        "schema_version": "pbt-magnet-frozen-selection-v1",
        "created_at": _utc_now(),
        "entries": entries,
    }
    frozen.mkdir(parents=True, exist_ok=True)
    _write_json(manifest, payload)
    digest = _sha256(manifest)
    marker.write_text(digest + "\n", encoding="ascii")
    return {"status": "FROZEN", "selection_manifest_sha256": digest}


def _selection_sha256(lifecycle_root: Path) -> str:
    frozen = lifecycle_root / "selection_frozen"
    manifest = frozen / "selection_manifest.json"
    marker = frozen / "COMMITTED"
    if not manifest.is_file() or not marker.is_file() or marker.is_symlink():
        raise ValueError("final training requires frozen selection")
    digest = _sha256(manifest)
    if marker.read_text(encoding="ascii").strip() != digest:
        raise ValueError("frozen selection digest mismatch")
    return digest


def _collect(
    lifecycle_root: Path,
    models: tuple[str, ...],
    *,
    recovery_seed42: bool,
) -> dict[str, object]:
    final_root = lifecycle_root / "final"
    runs: list[dict[str, object]] = []
    prediction_paths: list[Path] = []
    for model in models:
        for success_path in sorted((final_root / model).rglob("SUCCESS.json")):
            payload = _read_json(success_path)
            if payload.get("status") != "SUCCESS" or payload.get("test_read") is not True:
                raise ValueError(f"invalid final success marker: {success_path}")
            metrics = _read_json(success_path.parent / "metrics_test.json")
            run_manifest = _read_json(success_path.parent / "run_manifest.json")
            if recovery_seed42 and (
                run_manifest.get("seed") != 42
                or run_manifest.get("execution_mode") != "recovery_seed42"
            ):
                raise ValueError("recovery collection encountered a non-seed42 run")
            runs.append(
                {
                    "model": model,
                    "run_id": payload["run_id"],
                    "seed": run_manifest["seed"],
                    "experiment": run_manifest["experiment"],
                    "cutoff_cycle": run_manifest.get("cutoff_cycle"),
                    "metrics": metrics,
                    "success_sha256": _sha256(success_path),
                }
            )
            prediction_paths.append(success_path.parent / "predictions.parquet")
    if not runs:
        raise ValueError("no completed final runs to collect")
    aggregate_root = lifecycle_root / "summary"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    _write_json(aggregate_root / "run_index.json", {"runs": runs})
    _aggregate_predictions(prediction_paths, aggregate_root / "predictions.parquet")
    summary = _aggregate_scalar_metrics(runs, detailed_groups=recovery_seed42)
    _write_json(aggregate_root / "seed_summary.json", summary)
    _publish_plot_sources(
        aggregate_root=aggregate_root,
        final_root=final_root,
        runs=runs,
        prediction_paths=prediction_paths,
        seed_summary=summary,
    )
    success_payload = {
        "schema_version": "pbt-magnet-collection-success-v1",
        "status": "SUCCESS",
        "run_count": len(runs),
        "execution_mode": "recovery_seed42" if recovery_seed42 else "formal",
        "run_namespace": lifecycle_root.name,
        "finished_at": _utc_now(),
    }
    _write_json(aggregate_root / "SUCCESS.json", success_payload)
    return success_payload


def _aggregate_predictions(paths: list[Path], output: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to collect prediction tables") from exc
    tables = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"prediction table is missing: {path}")
        table = pq.read_table(path)
        tables.append(table.append_column("source_run", pa.array([path.parent.name] * len(table))))
    pq.write_table(pa.concat_tables(tables, promote_options="default"), output, compression="zstd")


def _aggregate_scalar_metrics(
    runs: list[dict[str, object]], *, detailed_groups: bool = False
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for run in runs:
        metrics = run["metrics"]
        if not isinstance(metrics, dict):
            continue
        cutoff = (
            "trajectory"
            if run.get("cutoff_cycle") is None
            else f"c{run['cutoff_cycle']}"
        )
        experiment = str(run.get("experiment", "default"))
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                group_cutoff = cutoff if detailed_groups else "all_cutoffs"
                group_experiment = experiment if detailed_groups else "all_experiments"
                grouped.setdefault(
                    (str(run["model"]), group_cutoff, group_experiment, name), []
                ).append(float(value))
    result: dict[str, object] = {}
    for (model, cutoff, experiment, metric), values in sorted(grouped.items()):
        mean = math.fsum(values) / len(values)
        key = (
            f"{model}.{cutoff}.{experiment}.{metric}"
            if detailed_groups
            else f"{model}.{metric}"
        )
        if len(values) == 1 and detailed_groups:
            result[key] = {
                "count": 1,
                "mean": mean,
                "std": None,
                "ci95_low": None,
                "ci95_high": None,
                "uncertainty_status": "UNAVAILABLE_SINGLE_SEED",
            }
            continue
        variance = math.fsum((value - mean) ** 2 for value in values) / max(
            len(values) - 1, 1
        )
        standard_deviation = math.sqrt(variance)
        half_width = 1.96 * standard_deviation / math.sqrt(len(values))
        result[key] = {
            "count": len(values),
            "mean": mean,
            "std": standard_deviation,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        }
    return result


def _publish_plot_sources(
    *,
    aggregate_root: Path,
    final_root: Path,
    runs: list[dict[str, object]],
    prediction_paths: list[Path],
    seed_summary: dict[str, object],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc  # type: ignore[import-untyped]
        import pyarrow.csv as pacsv  # type: ignore[import-untyped]
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to publish plot-source tables") from exc

    predictions = pq.read_table(aggregate_root / "predictions.parquet")
    predicted_columns = [
        name
        for name in (
            "source_run",
            "cell_id",
            "entity_id",
            "domain",
            "condition_key",
            "target_name",
            "horizon",
            "y_true",
            "y_pred",
            "conformal_lower",
            "conformal_upper",
        )
        if name in predictions.column_names
    ]
    pq.write_table(
        predictions.select(predicted_columns),
        aggregate_root / "predicted_vs_true.parquet",
        compression="zstd",
    )
    residual_columns = [
        name
        for name in (
            "source_run",
            "cell_id",
            "entity_id",
            "domain",
            "condition_key",
            "target_name",
            "horizon",
            "error",
            "abs_error",
        )
        if name in predictions.column_names
    ]
    pq.write_table(
        predictions.select(residual_columns),
        aggregate_root / "residuals.parquet",
        compression="zstd",
    )
    if "ood_status" in predictions.column_names:
        ood = predictions.filter(pc.not_equal(predictions["ood_status"], "IN_DOMAIN"))
    else:
        ood = predictions.slice(0, 0)
    pq.write_table(ood, aggregate_root / "ood_results.parquet", compression="zstd")

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in predictions.to_pylist():
        group = str(row.get("domain") or row.get("condition_key") or "UNSPECIFIED")
        target = str(row.get("target_name") or "cycle_life")
        error = row.get("error")
        if isinstance(error, (int, float)) and math.isfinite(float(error)):
            grouped.setdefault((str(row["source_run"]), group, target), []).append(
                float(error)
            )
    comparison = [
        {
            "source_run": source_run,
            "domain_or_condition": group,
            "target_name": target,
            "count": len(errors),
            "mae": math.fsum(abs(value) for value in errors) / len(errors),
            "rmse": math.sqrt(math.fsum(value * value for value in errors) / len(errors)),
        }
        for (source_run, group, target), errors in sorted(grouped.items())
    ]
    pq.write_table(
        pa.Table.from_pylist(comparison),
        aggregate_root / "domain_condition_comparison.parquet",
        compression="zstd",
    )

    histories = []
    for run in runs:
        run_id = str(run["run_id"])
        candidates = tuple(final_root.rglob(f"{run_id}/metrics_epoch.csv"))
        if len(candidates) != 1:
            raise ValueError(f"expected one metrics history for {run_id}")
        table = pacsv.read_csv(candidates[0])
        histories.append(
            table.append_column("source_run", pa.array([run_id] * len(table)))
        )
    pq.write_table(
        pa.concat_tables(histories, promote_options="default"),
        aggregate_root / "training_validation_loss.parquet",
        compression="zstd",
    )

    with (aggregate_root / "seed_stability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_metric",
                "count",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
                "uncertainty_status",
            ),
        )
        writer.writeheader()
        for name, values in sorted(seed_summary.items()):
            if not isinstance(values, dict):
                continue
            writer.writerow({"model_metric": name, **values})


def _validate_environment(*, stage: str, device: str) -> None:
    if stage not in _STAGES:
        raise ValueError(f"unknown stage: {stage}")
    forbidden = {"transformers", "huggingface_hub"} & set(sys.modules)
    if forbidden:
        raise RuntimeError(f"forbidden online model modules are loaded: {sorted(forbidden)}")
    if device.startswith("cuda") and device != "cuda:0":
        raise ValueError("A100 training must use process-local cuda:0")
    if device == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("formal A100 training requires CUDA_VISIBLE_DEVICES=1")
        offline = {
            "PIP_NO_INDEX": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        missing = [name for name, value in offline.items() if os.environ.get(name) != value]
        if missing:
            raise ValueError(f"formal A100 offline environment is incomplete: {missing}")


def _source_commit(root: Path) -> str:
    override = os.environ.get("QUANXIN_SOURCE_COMMIT")
    if override is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", override):
            raise ValueError("QUANXIN_SOURCE_COMMIT must be a 40-character lowercase digest")
        return override
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        package_manifest = _read_json(root / "package_manifest.json")
        if package_manifest.get("schema_version") != "quanxin-pbt-magnet-a100-v1":
            raise ValueError("offline package source revision evidence is invalid") from None
        commit = str(package_manifest.get("source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source git commit is invalid")
    return commit


def _run_id(view: ViewTask, *, stage: str, seed: int) -> str:
    cutoff = (
        "trajectory"
        if view.manifest.cutoff_cycle is None
        else f"c{view.manifest.cutoff_cycle}"
    )
    raw = f"{view.model}-{stage}-{cutoff}-{view.experiment}-seed{seed}"
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON file is missing or unsafe: {path}")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


__all__ = ["run_stage"]
