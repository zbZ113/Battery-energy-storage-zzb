"""Fail-closed orchestration for advanced MATR model training."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
)
from quanxin_life.models.cyclepatch import CyclePatchConfig
from quanxin_life.models.hybridpatch_v2 import HybridPatchV2Config
from quanxin_life.training.advanced_config import (
    AdvancedCandidate,
    AdvancedMatrThreeBatchRunConfig,
    AdvancedRunKey,
    CurrentHybridCandidate,
    CyclePatchBatLiNetCandidate,
    build_advanced_run_matrix,
)
from quanxin_life.training.advanced_data import (
    AdvancedFinalMatrData,
    AdvancedSelectionMatrData,
    load_advanced_matr_final_data,
    load_advanced_matr_selection_data,
)
from quanxin_life.training.advanced_selection import (
    AdvancedModelSelectionManifest,
    AdvancedValidationEvidence,
    BaselineValidationEvidence,
    build_advanced_model_selection_manifest,
    select_stage1_candidates,
    select_stage2_finalists,
)
from quanxin_life.training.advanced_tasks import (
    CyclePatchBatLiNetTrainingTask,
    CyclePatchDirectTrainingTask,
    HybridPatchV2TrainingTask,
)
from quanxin_life.training.checkpoint import AdvancedCheckpointContext, model_architecture_sha256
from quanxin_life.training.config import ModelTrainingConfig
from quanxin_life.training.engine import TrainingEngine, TrainingRunStatus

AdvancedMode = Literal["smoke", "select", "final"]
SelectionStage = Literal["selection_stage1", "selection_stage2", "selection_recheck"]
ADVANCED_FEATURE_VERSION = "cyclepatch-multichannel-v1"


def plan_advanced_run(
    project_root: Path,
    mode: AdvancedMode,
    *,
    selection_stage: SelectionStage = "selection_stage1",
    candidate_ids: tuple[str, ...] = (),
) -> tuple[AdvancedRunKey, ...]:
    """Load the governed mode configuration and produce its immutable run matrix."""

    root = project_root.resolve(strict=True)
    config = _load_mode_config(root, mode)
    return build_advanced_run_matrix(
        config,
        selection_stage=selection_stage,
        candidate_ids=candidate_ids,
        repository_root=root,
    )


def execute_advanced_smoke(project_root: Path, device: torch.device) -> dict[str, object]:
    config = _load_mode_config(project_root.resolve(strict=True), "smoke")
    return execute_advanced_matr_three_batch_suite(
        project_root=project_root, config=config, device=device
    )


def execute_advanced_select(project_root: Path, device: torch.device) -> dict[str, object]:
    config = _load_mode_config(project_root.resolve(strict=True), "select")
    return execute_advanced_matr_three_batch_suite(
        project_root=project_root, config=config, device=device
    )


def execute_advanced_final(project_root: Path, device: torch.device) -> dict[str, object]:
    config = _load_mode_config(project_root.resolve(strict=True), "final")
    return execute_advanced_matr_three_batch_suite(
        project_root=project_root, config=config, device=device
    )


@dataclass(frozen=True)
class _RegisteredInputs:
    manifest: MatrThreeBatchManifest
    split: SplitManifest
    input_bundle_sha256: str
    data_version: str
    split_version: str
    source_commit: str


def execute_advanced_matr_three_batch_suite(
    *,
    project_root: Path,
    config: AdvancedMatrThreeBatchRunConfig,
    device: torch.device,
    seed: int | None = None,
) -> dict[str, object]:
    """Run the governed advanced matrix in deterministic, sequential order.

    Selection intentionally calls the selection-only data loader, while final
    calls the four-partition loader. The engine owns checkpointing and resume.
    """

    root = project_root.resolve(strict=True)
    registered = _load_registered_inputs(root, config)
    run_root = _inside(root, config.paths.run_root, must_exist=False)
    run_root.mkdir(parents=True, exist_ok=True)
    if config.mode == "smoke":
        keys = build_advanced_run_matrix(config, repository_root=root)
        data_by_cutoff: dict[int, AdvancedSelectionMatrData | AdvancedFinalMatrData] = {}
        for key in keys:
            if key.cutoff_cycle not in data_by_cutoff:
                data_by_cutoff[key.cutoff_cycle] = load_advanced_matr_selection_data(
                    project_root=root,
                    manifest=registered.manifest,
                    combined_split=registered.split,
                    cutoff_cycle=key.cutoff_cycle,
                    feature_version=ADVANCED_FEATURE_VERSION,
                )
    elif config.mode == "select":
        if seed is not None:
            raise ValueError("Select owns its fixed 38/39/40 seed protocol")
        return _execute_selection(
            root=root,
            config=config,
            registered=registered,
            run_root=run_root,
            device=device,
        )
    else:
        keys = build_advanced_run_matrix(config, repository_root=root)
        data_by_cutoff = {
            cutoff: load_advanced_matr_final_data(
                project_root=root,
                manifest=registered.manifest,
                combined_split=registered.split,
                cutoff_cycle=cutoff,
                feature_version=ADVANCED_FEATURE_VERSION,
            )
            for cutoff in config.cutoffs
        }

    if seed is not None:
        if seed not in config.seeds:
            raise ValueError("requested seed is not configured for this mode")
        keys = tuple(key for key in keys if key.seed == seed)
    results: list[dict[str, object]] = []
    for key in keys:
        results.append(
            _run_key(
                root=root,
                run_root=run_root,
                config=config,
                registered=registered,
                key=key,
                data=data_by_cutoff[key.cutoff_cycle],
                device=device,
                epoch_limit=None,
            )
        )
    aggregate: dict[str, object] = {
        "mode": config.mode,
        "dataset_id": config.dataset_id,
        "target": config.target.value,
        "runs": results,
        "run_count": len(results),
        "input_bundle_sha256": registered.input_bundle_sha256,
    }
    return _write_merged_aggregate(run_root, aggregate)


def _run_key(
    *,
    root: Path,
    run_root: Path,
    config: AdvancedMatrThreeBatchRunConfig,
    registered: _RegisteredInputs,
    key: AdvancedRunKey,
    data: Any,
    device: torch.device,
    epoch_limit: int | None,
) -> dict[str, object]:
    candidate = _candidate_for_key(config, key)
    task = _build_training_task(run_key=key, candidate=candidate, data=data, split=registered.split)
    selection_hash = config.expected_selection_sha256 if config.mode == "final" else None
    context = AdvancedCheckpointContext(
        run_id=f"matr-{key.family}-{key.candidate_id}-c{key.cutoff_cycle}-s{key.seed}",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name=key.family,
        cutoff_cycle=key.cutoff_cycle,
        seed=key.seed,
        config_sha256=config.config_sha256,
        input_bundle_sha256=registered.input_bundle_sha256,
        data_version=registered.data_version,
        split_version=registered.split_version,
        feature_version=ADVANCED_FEATURE_VERSION,
        source_commit=registered.source_commit,
        run_mode=config.mode,
        stage=(
            "smoke"
            if config.mode == "smoke"
            else "final"
            if config.mode == "final"
            else "selection_recheck"
            if key.stage == "selection_recheck"
            else "selection_candidate"
        ),
        candidate_config_sha256=candidate.config_sha256,
        model_architecture_sha256=model_architecture_sha256(task.model, candidate.config_sha256),
        normalization_sha256=_normalization_hash(data, key.family),
        selection_manifest_sha256=selection_hash,
        reference_library_sha256=_reference_hash(task),
    )
    run_directory = (
        run_root / f"cutoff-{key.cutoff_cycle}" / key.family / key.candidate_id / f"seed-{key.seed}"
    )
    model_epochs: int
    if config.mode == "select":
        model_epochs = int(config.selection_policy.stage2_epochs)
    else:
        model_epochs = (
            min(key.max_epochs, 300)
            if key.family in {"cyclepatch_direct", "cyclepatch_batlinet"}
            else key.max_epochs
        )
    result = TrainingEngine(
        task=task,
        context=context,
        config=ModelTrainingConfig(
            name=key.family,
            max_epochs=model_epochs,
            validation_interval=5,
            early_stopping_patience=10,
            learning_rate=float(candidate.learning_rate),
        ),
        run_directory=run_directory,
        device=device,
    ).run(epoch_limit=epoch_limit)
    if config.mode == "final" and result.status in {
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.EARLY_STOPPED,
        TrainingRunStatus.SKIPPED_COMPLETED,
    }:
        if not isinstance(data, AdvancedFinalMatrData):
            raise ValueError("final evaluation requires held-out MATR data")
        _write_final_test_metrics(
            run_directory=run_directory,
            family=key.family,
            task=task,
            data=data,
            device=device,
        )
    return {
        "family": key.family,
        "candidate_id": key.candidate_id,
        "cutoff_cycle": key.cutoff_cycle,
        "seed": key.seed,
        "status": result.status.value,
        "best_epoch": result.best_epoch,
        "best_metric": result.best_metric,
        "run_directory": run_directory.relative_to(root).as_posix(),
    }


def _validation_metric(run_directory: Path) -> tuple[float, float, float]:
    path = run_directory / "metrics_validation.csv"
    if not path.is_file():
        raise ValueError("validation evidence is missing")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("validation evidence is empty")
    def row_mae(row: dict[str, str]) -> float:
        return float(row.get("mae", row.get("validation_loss", "nan")))

    row = min(rows, key=row_mae)
    mae = row_mae(row)
    rmse = float(row.get("rmse", mae))
    violation = float(row.get("monotonic_violation_rate", 0.0))
    if not all(math.isfinite(value) for value in (mae, rmse, violation)):
        raise ValueError("validation evidence contains a non-finite metric")
    return mae, rmse, violation


def _execute_selection(
    *,
    root: Path,
    config: AdvancedMatrThreeBatchRunConfig,
    registered: _RegisteredInputs,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Execute stage1, stage2 and 12-run recheck without touching held-out data."""

    run_root.mkdir(parents=True, exist_ok=True)
    stage1_keys = build_advanced_run_matrix(
        config, selection_stage="selection_stage1", repository_root=root
    )
    data100 = load_advanced_matr_selection_data(
        project_root=root,
        manifest=registered.manifest,
        combined_split=registered.split,
        cutoff_cycle=100,
        feature_version=ADVANCED_FEATURE_VERSION,
    )
    stage1_rows: list[AdvancedValidationEvidence] = []
    for key in stage1_keys:
        row = _run_key(
            root=root,
            run_root=run_root / "stage1",
            config=config,
            registered=registered,
            key=key,
            data=data100,
            device=device,
            epoch_limit=30,
        )
        directory = root / str(row["run_directory"])
        mae, rmse, violation = _validation_metric(directory)
        stage1_rows.append(
            AdvancedValidationEvidence(
                family=key.family,
                candidate_id=key.candidate_id,
                candidate_config_sha256=key.candidate_config_sha256,
                stage="selection_stage1",
                cutoff_cycle=100,
                seed=38,
                status="completed",
                validation_mae=mae,
                validation_rmse=rmse,
                monotonic_violation_rate=violation,
            )
        )
    stage1_evidence = tuple(stage1_rows)
    survivors = select_stage1_candidates(stage1_evidence)
    survivor_ids = tuple(candidate_id for ids in survivors.values() for candidate_id in ids)
    stage2_keys = build_advanced_run_matrix(
        config,
        selection_stage="selection_stage2",
        candidate_ids=survivor_ids,
        repository_root=root,
    )
    stage2_rows: list[AdvancedValidationEvidence] = []
    for key in stage2_keys:
        row = _run_key(
            root=root,
            run_root=run_root / "stage1",
            config=config,
            registered=registered,
            key=key,
            data=data100,
            device=device,
            epoch_limit=90,
        )
        directory = root / str(row["run_directory"])
        mae, rmse, violation = _validation_metric(directory)
        stage2_rows.append(
            AdvancedValidationEvidence(
                family=key.family,
                candidate_id=key.candidate_id,
                candidate_config_sha256=key.candidate_config_sha256,
                stage="selection_stage2",
                cutoff_cycle=100,
                seed=38,
                status="completed",
                validation_mae=mae,
                validation_rmse=rmse,
                monotonic_violation_rate=violation,
            )
        )
    stage2_evidence = tuple(stage2_rows)
    finalist_groups = select_stage2_finalists(stage2_evidence)
    finalist_id_values = tuple(
        candidate_id for ids in finalist_groups.values() for candidate_id in ids
    )
    recheck_rows: list[AdvancedValidationEvidence] = []
    data_cache: dict[int, AdvancedSelectionMatrData] = {100: data100}
    for key in build_advanced_run_matrix(
        config,
        selection_stage="selection_recheck",
        candidate_ids=finalist_id_values,
        repository_root=root,
    ):
        if key.cutoff_cycle not in data_cache:
            data_cache[key.cutoff_cycle] = load_advanced_matr_selection_data(
                project_root=root,
                manifest=registered.manifest,
                combined_split=registered.split,
                cutoff_cycle=key.cutoff_cycle,
                feature_version=ADVANCED_FEATURE_VERSION,
            )
        row = _run_key(
            root=root,
            run_root=run_root / "recheck",
            config=config,
            registered=registered,
            key=key,
            data=data_cache[key.cutoff_cycle],
            device=device,
            epoch_limit=90,
        )
        mae, rmse, violation = _validation_metric(root / str(row["run_directory"]))
        recheck_rows.append(
            AdvancedValidationEvidence(
                family=key.family,
                candidate_id=key.candidate_id,
                candidate_config_sha256=key.candidate_config_sha256,
                stage="selection_recheck",
                cutoff_cycle=cast(Literal[20, 50, 100, 150], key.cutoff_cycle),
                seed=cast(Literal[38, 39, 40], key.seed),
                status="completed",
                validation_mae=mae,
                validation_rmse=rmse,
                monotonic_violation_rate=violation,
            )
        )
    trace: dict[str, object] = {
        "schema_version": "advanced-selection-trace-v1",
        "stage1_evidence": [item.model_dump(mode="json") for item in stage1_evidence],
        "stage1_survivors": survivors,
        "stage2_evidence": [item.model_dump(mode="json") for item in stage2_evidence],
        "stage2_finalists": finalist_groups,
        "recheck_evidence": [item.model_dump(mode="json") for item in recheck_rows],
    }
    finalist_candidates = tuple(
        candidate
        for candidate in config.candidates
        if candidate.candidate_id in finalist_id_values
    )
    baseline_evidence = _build_baseline_evidence(
        config=config,
        data_cache=data_cache,
        recheck_rows=tuple(recheck_rows),
    )
    manifest = build_advanced_model_selection_manifest(
        stage1_evidence=stage1_evidence,
        stage2_evidence=stage2_evidence,
        finalists=finalist_candidates,
        recheck_evidence=tuple(recheck_rows),
        baselines=baseline_evidence,
        input_bundle_sha256=registered.input_bundle_sha256,
        data_sha256=sha256_canonical(
            [batch.raw_sha256 for batch in registered.manifest.batches]
        ),
        split_sha256=registered.manifest.combined_split_sha256,
        feature_sha256=sha256_canonical(
            {
                "feature_version": ADVANCED_FEATURE_VERSION,
                "cutoffs": list(config.cutoffs),
            }
        ),
        normalization_sha256=sha256_canonical(
            {
                str(cutoff): {
                    "scalar": data.scalar_normalizer.statistics_sha256,
                    "hybrid": data.hybrid_normalizer.statistics_sha256,
                }
                for cutoff, data in sorted(data_cache.items())
            }
        ),
        source_commit=registered.source_commit,
        validation_cell_sha256=sha256_canonical(list(registered.split.validation)),
    )
    _write_json(run_root / "selection_trace.json", trace)
    _write_json(run_root / "selection_manifest.json", manifest.model_dump(mode="json"))
    _write_resolved_final_config(root, manifest, run_root)
    return {
        "mode": "select",
        **trace,
        "selection_manifest_sha256": _sha256_file(run_root / "selection_manifest.json"),
        "final_config_path": (
            run_root / "final_config_resolved.json"
        ).relative_to(root).as_posix(),
        "run_count": len(stage1_rows) + len(stage2_rows) + len(recheck_rows),
    }


def _build_baseline_evidence(
    *,
    config: AdvancedMatrThreeBatchRunConfig,
    data_cache: dict[int, AdvancedSelectionMatrData],
    recheck_rows: tuple[AdvancedValidationEvidence, ...],
) -> tuple[BaselineValidationEvidence, ...]:
    """Build the 12-axis XGBoost baseline and reuse current-Hybrid validation rows."""

    hybrid_rows = tuple(
        row for row in recheck_rows if row.family == "current_hybrid" and row.status == "completed"
    )
    if len(hybrid_rows) != 12:
        raise ValueError("current Hybrid baseline requires a complete 12-run grid")
    xgb_rows: list[BaselineValidationEvidence] = []
    for cutoff in config.selection_policy.recheck_cutoffs:
        data = data_cache[cutoff]
        train_x = _baseline_features(data.scalar_train)
        validation_x = _baseline_features(data.scalar_validation)
        train_y = data.scalar_train.raw_labels.detach().cpu().numpy()
        validation_y = data.scalar_validation.raw_labels.detach().cpu().numpy()
        for seed in config.selection_policy.recheck_seeds:
            prediction = _fit_xgboost_baseline(
                train_x,
                train_y,
                validation_x,
                validation_y,
                seed=seed,
            )
            error = prediction - validation_y
            xgb_rows.append(
                BaselineValidationEvidence(
                    baseline="xgboost",
                    cutoff_cycle=cutoff,
                    seed=seed,
                    validation_mae=float(abs(error).mean()),
                    validation_rmse=float((error**2).mean() ** 0.5),
                )
            )
    return tuple(
        sorted(
            (*xgb_rows, *(BaselineValidationEvidence(
                baseline="current_hybrid",
                cutoff_cycle=row.cutoff_cycle,
                seed=row.seed,
                validation_mae=cast(float, row.validation_mae),
                validation_rmse=cast(float, row.validation_rmse),
                monotonic_violation_rate=row.monotonic_violation_rate,
            ) for row in hybrid_rows)),
            key=lambda row: (row.baseline, row.cutoff_cycle, row.seed),
        )
    )


def _baseline_features(batch: Any) -> Any:
    values = batch.early_batch.values.detach().cpu()
    mask = batch.early_batch.sample_mask.detach().cpu().unsqueeze(-1)
    safe = torch.where(mask, values, torch.zeros_like(values))
    counts = mask.to(values.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
    pooled = safe.sum(dim=(1, 2, 3)) / counts
    condition = batch.early_batch.condition_values.detach().cpu()
    condition_mask = batch.early_batch.condition_mask.detach().cpu()
    condition = torch.where(condition_mask, condition, torch.zeros_like(condition))
    return torch.cat((pooled, condition), dim=1).numpy()


def _fit_xgboost_baseline(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    *,
    seed: int,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - guarded by the A100 lock file
        raise RuntimeError("xgboost is required for advanced selection baselines") from exc
    train_matrix = xgb.DMatrix(train_x, label=train_y)
    validation_matrix = xgb.DMatrix(validation_x, label=validation_y)
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": seed,
            "nthread": 1,
            "tree_method": "hist",
        },
        train_matrix,
        num_boost_round=2000,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    return booster.predict(validation_matrix, iteration_range=(0, best_iteration + 1))


def _write_resolved_final_config(
    root: Path,
    manifest: AdvancedModelSelectionManifest,
    run_root: Path,
) -> None:
    template_path = root / "configs" / "training" / "advanced" / "final.json"
    template = AdvancedMatrThreeBatchRunConfig.model_validate_json(template_path.read_bytes())
    manifest_path = run_root / "selection_manifest.json"
    resolved = template.model_copy(
        update={
            "candidates": manifest.selection.selected_candidates,
            "expected_selection_sha256": _sha256_file(manifest_path),
            "selection_manifest_path": manifest_path.relative_to(root).as_posix(),
        }
    )
    _write_json(run_root / "final_config_resolved.json", resolved.model_dump(mode="json"))


def _candidate_for_key(
    config: AdvancedMatrThreeBatchRunConfig, key: AdvancedRunKey
) -> AdvancedCandidate:
    for candidate in config.candidates:
        if candidate.candidate_id == key.candidate_id:
            return candidate
    raise ValueError(f"candidate {key.candidate_id!r} is not configured")


def _build_training_task(
    *, run_key: AdvancedRunKey, candidate: AdvancedCandidate, data: Any, split: SplitManifest
) -> Any:
    if isinstance(data, AdvancedFinalMatrData) and run_key.stage != "final":
        raise ValueError("selection task cannot consume final data")
    train_scalar = data.scalar_train
    validation_scalar = data.scalar_validation
    if run_key.family == "cyclepatch_direct":
        return CyclePatchDirectTrainingTask(
            train_batch=train_scalar,
            validation_batch=validation_scalar,
            split_manifest=split,
            config=_cyclepatch_config(candidate),
            learning_rate=float(candidate.learning_rate),
            weight_decay=float(getattr(candidate, "weight_decay", 0.0)),
            seed=run_key.seed,
        )
    if run_key.family == "cyclepatch_batlinet":
        batlinet_candidate = cast(CyclePatchBatLiNetCandidate, candidate)
        labels = dict(zip(train_scalar.cell_ids, train_scalar.raw_labels.tolist(), strict=True))
        supervised_split = SplitManifest(
            dataset_id=split.dataset_id,
            seed=split.seed,
            train=train_scalar.cell_ids,
            validation=validation_scalar.cell_ids,
            calibration=(),
            test=(),
        )
        scaler = CycleLifeTargetScaler.fit(
            labels,
            training_cell_ids=train_scalar.cell_ids,
            split_manifest=supervised_split,
            cutoff_cycle=run_key.cutoff_cycle,
            dataset_id="MATR",
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        )
        library = CycleLifeReferenceLibrary.build(
            labels,
            training_cell_ids=train_scalar.cell_ids,
            split_manifest=supervised_split,
            scaler=scaler,
            reference_count=min(batlinet_candidate.reference_count, len(train_scalar.cell_ids)),
            seed=run_key.seed,
        )
        return CyclePatchBatLiNetTrainingTask(
            train_batch=train_scalar,
            validation_batch=validation_scalar,
            split_manifest=split,
            target_scaler=scaler,
            reference_library=library,
            config=BatLiNetConfig(
                encoder=_cyclepatch_config(candidate),
                lambda_pair=batlinet_candidate.lambda_pair,
                lambda_rank=batlinet_candidate.lambda_rank,
                fusion_alpha=batlinet_candidate.fusion_alpha,
                reference_count=library.reference_count,
            ),
            learning_rate=float(candidate.learning_rate),
            weight_decay=batlinet_candidate.weight_decay,
            seed=run_key.seed,
        )
    if run_key.family == "current_hybrid":
        return _build_current_hybrid_task(
            data, cast(CurrentHybridCandidate, candidate), run_key.seed
        )
    return HybridPatchV2TrainingTask(
        train_batch=data.hybrid_train,
        validation_batch=data.hybrid_validation,
        split_manifest=split,
        config=_hybrid_config(candidate),
        learning_rate=float(candidate.learning_rate),
        weight_decay=float(getattr(candidate, "weight_decay", 0.0)),
        seed=run_key.seed,
    )


def _cyclepatch_config(candidate: AdvancedCandidate) -> CyclePatchConfig:
    return CyclePatchConfig(
        d_model=int(getattr(candidate, "d_model", 128)),
        layers=int(getattr(candidate, "layers", 2)),
        heads=int(getattr(candidate, "heads", 4)),
        dropout=float(getattr(candidate, "dropout", 0.1)),
    )


def _hybrid_config(candidate: AdvancedCandidate) -> HybridPatchV2Config:
    return HybridPatchV2Config(
        cyclepatch=_cyclepatch_config(candidate),
        query_token_count=int(getattr(candidate, "query_token_count", 0)),
        query_layers=int(getattr(candidate, "query_layers", 1)),
        decoder_hidden_dim=int(
            getattr(candidate, "decoder_hidden_dim", getattr(candidate, "hidden_dim", 64))
        ),
        huber_delta=float(getattr(candidate, "huber_delta", 1.0)),
        lambda_history=float(getattr(candidate, "lambda_history", 0.0)),
        lambda_smooth=float(getattr(candidate, "lambda_smooth", 0.0)),
        lambda_order=float(getattr(candidate, "lambda_order", 0.0)),
        lambda_residual=float(getattr(candidate, "lambda_residual", 0.01)),
    )


def _build_current_hybrid_task(data: Any, candidate: CurrentHybridCandidate, seed: int) -> Any:
    """Adapt real masked SOH batches to the legacy current-Hybrid baseline."""

    from quanxin_life.training.tasks import HybridTrajectoryTrainingTask

    prediction_cycles = _current_hybrid_prediction_cycles(
        data.hybrid_train, data.hybrid_validation
    )
    train = _current_hybrid_batch(
        data.hybrid_train, prediction_cycles=prediction_cycles
    )
    validation = _current_hybrid_batch(
        data.hybrid_validation, prediction_cycles=prediction_cycles
    )
    return HybridTrajectoryTrainingTask(
        train_batch=train,
        validation_batch=validation,
        hidden_dim=candidate.hidden_dim,
        learning_rate=float(candidate.learning_rate),
        seed=seed,
    )


def _current_hybrid_prediction_cycles(*batches: Any) -> tuple[int, ...]:
    if not batches:
        raise ValueError("current Hybrid baseline requires trajectory batches")
    source_cycles = batches[0].inputs.prediction_cycles.detach().cpu()
    common = torch.ones_like(source_cycles, dtype=torch.bool)
    for batch in batches:
        cycles = batch.inputs.prediction_cycles.detach().cpu()
        if not torch.equal(source_cycles, cycles):
            raise ValueError("current Hybrid source prediction axes must match")
        common &= batch.targets.target_mask.detach().cpu().all(dim=0)
    prediction_cycles = tuple(int(value) for value in source_cycles[common].tolist())
    if len(prediction_cycles) < 3:
        raise ValueError("current Hybrid baseline needs three common real target cycles")
    if prediction_cycles[-1] != 500:
        raise ValueError("current Hybrid baseline requires a real cycle-500 target")
    return prediction_cycles


def _current_hybrid_batch(
    batch: Any, *, prediction_cycles: tuple[int, ...]
) -> Any:
    from quanxin_life.training.tasks import HybridTrajectoryBatch

    if len(prediction_cycles) < 3:
        raise ValueError("current Hybrid baseline needs three common real target cycles")
    if prediction_cycles[-1] != 500:
        raise ValueError("current Hybrid baseline requires a real cycle-500 target")
    source_cycles = tuple(int(value) for value in batch.inputs.prediction_cycles.tolist())
    source_indices = {cycle: index for index, cycle in enumerate(source_cycles)}
    if any(cycle not in source_indices for cycle in prediction_cycles):
        raise ValueError("current Hybrid prediction axis is unavailable in this partition")
    indices = torch.tensor(
        [source_indices[cycle] for cycle in prediction_cycles],
        dtype=torch.int64,
        device=batch.targets.target_soh.device,
    )
    selected_mask = batch.targets.target_mask.index_select(1, indices)
    if not bool(selected_mask.all().item()):
        raise ValueError("current Hybrid prediction axis contains missing real targets")
    values = batch.inputs.early_batch.values
    mask = batch.inputs.early_batch.sample_mask.unsqueeze(-1)
    clean = torch.where(mask, values, torch.zeros_like(values))
    count = mask.to(values.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
    features = clean.sum(dim=(1, 2, 3)) / count
    targets = batch.targets.target_soh.index_select(1, indices)
    return HybridTrajectoryBatch(
        dataset_id="MATR",
        cell_ids=batch.cell_ids,
        features=features,
        initial_soh=batch.inputs.initial_soh,
        target_soh=targets,
        prediction_cycles=prediction_cycles,
        cutoff_cycle=int(batch.inputs.early_batch.cycle_mask.shape[1] - 1),
    )


def _write_final_test_metrics(
    *,
    run_directory: Path,
    family: str,
    task: Any,
    data: AdvancedFinalMatrData,
    device: torch.device,
) -> None:
    batch = (
        data.scalar_test
        if family in {"cyclepatch_direct", "cyclepatch_batlinet"}
        else _current_hybrid_batch(
            data.hybrid_test,
            prediction_cycles=task.train_batch.prediction_cycles,
        )
        if family == "current_hybrid"
        else data.hybrid_test
    )
    metrics = task.evaluate(batch, device=device)
    _write_json(
        run_directory / "metrics_test.json",
        {
            "schema_version": "advanced-test-metrics-v1",
            "family": family,
            "partition": "test",
            "cell_count": len(batch.cell_ids),
            "loss": metrics.loss,
            "metrics": metrics.metrics,
        },
    )


def _load_registered_inputs(
    root: Path, config: AdvancedMatrThreeBatchRunConfig
) -> _RegisteredInputs:
    manifest_path = _inside(root, config.paths.three_batch_manifest)
    split_path = _inside(root, config.paths.split_manifest)
    manifest = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
    split = SplitManifest.model_validate_json(split_path.read_bytes())
    if manifest.combined_split_manifest != config.paths.split_manifest:
        raise ValueError("advanced MATR manifest and config split path differ")
    input_bundle_sha = sha256_canonical(
        {
            "manifest_sha256": _sha256_file(manifest_path),
            "split_sha256": _sha256_file(split_path),
            "config_sha256": config.config_sha256,
        }
    )
    return _RegisteredInputs(
        manifest=manifest,
        split=split,
        input_bundle_sha256=input_bundle_sha,
        data_version=manifest.data_version,
        split_version=manifest.split_version,
        source_commit=_source_commit(root),
    )


def _normalization_hash(data: Any, family: str) -> str:
    normalizer = (
        data.scalar_normalizer
        if family in {"cyclepatch_direct", "cyclepatch_batlinet"}
        else data.hybrid_normalizer
    )
    value = normalizer.statistics_sha256
    if not isinstance(value, str):
        raise ValueError("normalizer statistics hash is invalid")
    return value


def _reference_hash(task: Any) -> str | None:
    library = getattr(task, "reference_library", None)
    if library is None:
        return None
    value = getattr(library, "library_sha256", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("BatLiNet reference library is missing its registered hash")
    return value


def _source_commit(root: Path) -> str:
    import subprocess

    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision_path = root / "source_revision.json"
        if revision_path.is_symlink() or not revision_path.is_file():
            raise ValueError("cannot determine source commit") from None
        payload = json.loads(revision_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "source-revision-v1"
            or payload.get("git_dirty") is not False
        ):
            raise ValueError("source revision evidence is invalid") from None
        value = str(payload.get("git_commit", ""))
    if len(value) not in {40, 64}:
        raise ValueError("source commit is invalid")
    return value


def _inside(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = (root / Path(*relative.replace("\\", "/").split("/"))).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise ValueError("advanced training path escapes project root")
    return path


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:

    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_merged_aggregate(
    run_root: Path, aggregate: Mapping[str, object]
) -> dict[str, object]:
    destination = run_root / "aggregate_metrics.json"
    existing_runs: list[dict[str, object]] = []
    if destination.is_file() and not destination.is_symlink():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        for field in ("mode", "dataset_id", "target", "input_bundle_sha256"):
            if existing.get(field) != aggregate.get(field):
                raise ValueError("existing advanced aggregate belongs to another run context")
        existing_runs = list(existing.get("runs", []))
    combined: dict[tuple[object, ...], dict[str, object]] = {}
    for row in (*existing_runs, *cast(list[dict[str, object]], aggregate["runs"])):
        identity = (
            row.get("family"),
            row.get("candidate_id"),
            row.get("cutoff_cycle"),
            row.get("seed"),
        )
        combined[identity] = row
    runs = sorted(
        combined.values(),
        key=lambda row: (
            int(cast(int, row["cutoff_cycle"])),
            str(row["family"]),
            str(row["candidate_id"]),
            int(cast(int, row["seed"])),
        ),
    )
    merged = {
        **dict(aggregate),
        "runs": runs,
        "run_count": len(runs),
    }
    _write_json(destination, merged)
    return merged


def _load_mode_config(root: Path, mode: AdvancedMode) -> AdvancedMatrThreeBatchRunConfig:
    resolved_final = (
        root
        / "runs"
        / "a100"
        / "matr-three-batch"
        / "advanced"
        / "selection"
        / "final_config_resolved.json"
    )
    if mode == "final" and resolved_final.is_file() and not resolved_final.is_symlink():
        path = resolved_final
    else:
        path = (
            root
            / "configs"
            / "training"
            / "advanced"
            / ("selection.json" if mode == "select" else f"{mode}.json")
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"advanced {mode} configuration is unavailable")
    config = AdvancedMatrThreeBatchRunConfig.model_validate_json(path.read_bytes())
    if config.mode != mode:
        raise ValueError("advanced configuration mode does not match requested mode")
    return config
