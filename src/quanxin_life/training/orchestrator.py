"""End-to-end MATR model-matrix execution on the approved A100 binding."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from quanxin_life.core import (
    CycleLifePrediction,
    PredictionTarget,
    sha256_canonical,
)
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.matr_pipeline import MatrSupervisionArtifact
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.metrics import evaluate_cycle_life_predictions
from quanxin_life.training.checkpoint import CheckpointContext
from quanxin_life.training.classic import (
    fit_dummy_cycle_life,
    fit_variance_cycle_life,
    train_xgboost_cycle_life,
)
from quanxin_life.training.config import ModelTrainingConfig
from quanxin_life.training.engine import TrainingEngine
from quanxin_life.training.matr_data import (
    MatrCurveCohorts,
    MatrHybridCohorts,
    load_matr_cycle_life_curve_cohorts,
    load_matr_hybrid_trajectory_cohorts,
)
from quanxin_life.training.plots import (
    write_cycle_life_evaluation_plot,
    write_hybrid_trajectory_plot,
)
from quanxin_life.training.reports import write_matr_experiment_reports
from quanxin_life.training.suite import MatrRunConfig, TrainingRunKey, build_run_matrix
from quanxin_life.training.tasks import (
    CPMLPTrainingTask,
    CycleLifeCurveBatch,
    HybridTrajectoryTrainingTask,
)
from quanxin_life.uncertainty import (
    calibrate_cycle_life_conformal,
    evaluate_cycle_life_interval_coverage,
    make_cycle_life_interval,
)


def execute_matr_suite(
    *,
    project_root: Path,
    config: MatrRunConfig,
    device: torch.device,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    paths = config.paths
    raw_path = _inside(root, paths.raw_mat)
    raw_manifest = RawFileManifest.model_validate_json(
        _inside(root, paths.raw_manifest).read_bytes()
    )
    verify_raw_file(raw_path, raw_manifest)
    supervision = MatrSupervisionArtifact.model_validate_json(
        _inside(root, paths.supervision_report).read_bytes()
    )
    split = SplitManifest.model_validate_json(_inside(root, paths.split_manifest).read_bytes())
    run_root = _inside(root, paths.run_root, must_exist=False)
    run_root.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit(root)
    input_bundle_sha256 = sha256_canonical(
        {
            "raw_sha256": raw_manifest.sha256,
            "supervision_sha256": supervision.parquet_sha256,
            "split": split.model_dump(mode="json"),
            "suite_config_sha256": config.suite.config_sha256,
        }
    )
    all_metrics: list[dict[str, Any]] = []
    for cutoff in config.suite.cutoffs:
        curve_cohorts = load_matr_cycle_life_curve_cohorts(
            processed_root=_inside(root, paths.processed_root),
            supervision_root=_inside(root, paths.supervision_root),
            supervision=supervision,
            split_manifest=split,
            cutoff_cycle=cutoff,
            voltage_min_v=2.0,
            voltage_max_v=3.6,
            voltage_grid_step_v=0.01,
        )
        hybrid_cohorts = load_matr_hybrid_trajectory_cohorts(
            processed_root=_inside(root, paths.processed_root),
            supervision_root=_inside(root, paths.supervision_root),
            supervision=supervision,
            split_manifest=split,
            cutoff_cycle=cutoff,
        )
        keys = tuple(key for key in build_run_matrix(config.suite) if key.cutoff_cycle == cutoff)
        for key in keys:
            model_config = next(
                model for model in config.suite.models if model.name == key.model_name
            )
            run_directory = (
                run_root / f"cutoff-{cutoff}" / key.model_name / f"seed-{key.seed}"
            )
            run_directory.mkdir(parents=True, exist_ok=True)
            context = CheckpointContext(
                run_id=f"matr-{key.model_name}-c{cutoff}-s{key.seed}",
                dataset_id="MATR",
                target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
                model_name=key.model_name,
                cutoff_cycle=cutoff,
                seed=key.seed,
                config_sha256=config.suite.config_sha256,
                input_bundle_sha256=input_bundle_sha256,
                data_version=config.suite.data_version,
                split_version=config.suite.split_version,
                feature_version=config.suite.feature_version,
                source_commit=source_commit,
            )
            metrics = _run_one(
                key=key,
                context=context,
                model_config=model_config,
                curve_cohorts=curve_cohorts,
                hybrid_cohorts=hybrid_cohorts,
                run_directory=run_directory,
                device=device,
                xgboost_early_stopping_rounds=config.xgboost_early_stopping_rounds,
                split_manifest=split,
            )
            all_metrics.append(metrics)
            _log_mlflow(
                run_root=run_root,
                context=context,
                metrics=metrics,
                required=config.mode == "final",
            )
    aggregate = _aggregate(
        all_metrics,
        mode=config.mode,
        expected_seeds=config.suite.seeds,
    )
    _write_json_atomic(run_root / "aggregate_metrics.json", aggregate)
    _write_metrics_csv(run_root / "metrics_test.csv", all_metrics)
    write_matr_experiment_reports(run_root, aggregate)
    return aggregate


def _run_one(
    *,
    key: TrainingRunKey,
    context: CheckpointContext,
    model_config: ModelTrainingConfig,
    curve_cohorts: MatrCurveCohorts,
    hybrid_cohorts: MatrHybridCohorts,
    run_directory: Path,
    device: torch.device,
    xgboost_early_stopping_rounds: int,
    split_manifest: SplitManifest,
) -> dict[str, Any]:
    cached = _load_completed_run(run_directory, context)
    if cached is not None:
        return cached
    _write_json_atomic(run_directory / "config_resolved.json", context.model_dump(mode="json"))
    _write_json_atomic(run_directory / "environment.json", _environment(device))
    artifacts = run_directory / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if key.model_name == "dummy":
        training_started = time.perf_counter()
        dummy_model = fit_dummy_cycle_life(curve_cohorts.train)
        training_time_seconds = time.perf_counter() - training_started
        validation_predicted = dummy_model.predict(curve_cohorts.validation)
        _write_static_baseline_logs(
            run_directory,
            context=context,
            training_time_seconds=training_time_seconds,
            validation_metrics=_validation_cycle_metrics(
                curve_cohorts.validation,
                validation_predicted,
                context,
            ),
        )
        predicted = dummy_model.predict(curve_cohorts.test)
        calibration_predicted = dummy_model.predict(curve_cohorts.calibration)
        _write_json_atomic(
            artifacts / "dummy.json",
            {"predicted_cycle": dummy_model.predicted_cycle},
        )
        metrics = _cycle_metrics(
            curve_cohorts.test,
            predicted,
            context,
            calibration_batch=curve_cohorts.calibration,
            calibration_predicted=calibration_predicted,
            split_manifest=split_manifest,
        )
        metrics["training_time_seconds"] = training_time_seconds
        _write_cycle_plot(run_directory, curve_cohorts.test, predicted, metrics)
    elif key.model_name == "variance":
        training_started = time.perf_counter()
        variance_model = fit_variance_cycle_life(curve_cohorts.train)
        training_time_seconds = time.perf_counter() - training_started
        validation_predicted = variance_model.predict(curve_cohorts.validation)
        _write_static_baseline_logs(
            run_directory,
            context=context,
            training_time_seconds=training_time_seconds,
            validation_metrics=_validation_cycle_metrics(
                curve_cohorts.validation,
                validation_predicted,
                context,
            ),
        )
        predicted = variance_model.predict(curve_cohorts.test)
        calibration_predicted = variance_model.predict(curve_cohorts.calibration)
        _write_json_atomic(
            artifacts / "variance.json",
            {
                "intercept": variance_model.intercept,
                "coefficient": variance_model.coefficient,
            },
        )
        metrics = _cycle_metrics(
            curve_cohorts.test,
            predicted,
            context,
            calibration_batch=curve_cohorts.calibration,
            calibration_predicted=calibration_predicted,
            split_manifest=split_manifest,
        )
        metrics["training_time_seconds"] = training_time_seconds
        _write_cycle_plot(run_directory, curve_cohorts.test, predicted, metrics)
    elif key.model_name == "xgboost":
        xgboost_model = train_xgboost_cycle_life(
            train_batch=curve_cohorts.train,
            validation_batch=curve_cohorts.validation,
            max_rounds=model_config.max_epochs,
            early_stopping_rounds=xgboost_early_stopping_rounds,
            seed=key.seed,
            device="cuda" if device.type == "cuda" else "cpu",
            checkpoint_directory=run_directory / "checkpoints",
            log_directory=run_directory,
        )
        xgboost_model.save_model(artifacts / "model.ubj")
        predicted = xgboost_model.predict(curve_cohorts.test)
        calibration_predicted = xgboost_model.predict(curve_cohorts.calibration)
        metrics = {
            **_cycle_metrics(
                curve_cohorts.test,
                predicted,
                context,
                calibration_batch=curve_cohorts.calibration,
                calibration_predicted=calibration_predicted,
                split_manifest=split_manifest,
            ),
            "best_iteration": xgboost_model.best_iteration,
            "training_time_seconds": xgboost_model.training_time_seconds,
        }
        _write_cycle_plot(run_directory, curve_cohorts.test, predicted, metrics)
        _write_json_atomic(
            run_directory / "training_log.json",
            {"evaluation_history": xgboost_model.evaluation_history},
        )
    elif key.model_name == "cpmlp":
        cpmlp_task = CPMLPTrainingTask(
            train_batch=curve_cohorts.train,
            validation_batch=curve_cohorts.validation,
            curve_hidden_dim=32,
            aggregation_hidden_dim=32,
            learning_rate=model_config.learning_rate,
            seed=key.seed,
        )
        result = TrainingEngine(
            task=cpmlp_task,
            context=context,
            config=model_config,
            run_directory=run_directory,
            device=device,
            keep_recent_checkpoints=3,
        ).run()
        cpmlp_task.model.eval()
        with torch.no_grad():
            predicted_tensor = cpmlp_task.model(
                curve_cohorts.test.curve_values.to(device),
                curve_cohorts.test.observed_mask.to(device),
            )
            calibration_tensor = cpmlp_task.model(
                curve_cohorts.calibration.curve_values.to(device),
                curve_cohorts.calibration.observed_mask.to(device),
            )
        predicted = predicted_tensor.detach().cpu().numpy().astype(np.float64)
        calibration_predicted = (
            calibration_tensor.detach().cpu().numpy().astype(np.float64)
        )
        metrics = {
            **_cycle_metrics(
                curve_cohorts.test,
                predicted,
                context,
                calibration_batch=curve_cohorts.calibration,
                calibration_predicted=calibration_predicted,
                split_manifest=split_manifest,
            ),
            "best_epoch": result.best_epoch,
            "last_epoch": result.last_epoch,
            "training_status": result.status.value,
            "training_time_seconds": result.training_time_seconds,
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
        }
        _write_cycle_plot(run_directory, curve_cohorts.test, predicted, metrics)
        _save_safe_tensor_artifact(cpmlp_task.model, artifacts)
    elif key.model_name == "hybrid":
        hybrid_task = HybridTrajectoryTrainingTask(
            train_batch=hybrid_cohorts.train,
            validation_batch=hybrid_cohorts.validation,
            hidden_dim=32,
            learning_rate=model_config.learning_rate,
            seed=key.seed,
        )
        result = TrainingEngine(
            task=hybrid_task,
            context=context,
            config=model_config,
            run_directory=run_directory,
            device=device,
            keep_recent_checkpoints=3,
        ).run()
        test_metrics = hybrid_task.evaluate(hybrid_cohorts.test, device=device)
        test_prediction = (
            hybrid_task.predict(hybrid_cohorts.test, device=device)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        metrics = {
            "dataset_id": "MATR",
            "target": "real_soh_trajectory_through_cycle_500",
            "model": "hybrid",
            "cutoff_cycle": key.cutoff_cycle,
            "seed": key.seed,
            **test_metrics.metrics,
            "best_epoch": result.best_epoch,
            "last_epoch": result.last_epoch,
            "training_status": result.status.value,
            "training_time_seconds": result.training_time_seconds,
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
            "conformal_status": "NOT_APPLICABLE_TRAJECTORY_TARGET",
        }
        write_hybrid_trajectory_plot(
            run_directory / "plots" / "soh_trajectory.png",
            batch=hybrid_cohorts.test,
            predicted=test_prediction,
        )
        _save_safe_tensor_artifact(hybrid_task.model, artifacts)
    else:  # pragma: no cover - configuration model rejects this branch
        raise ValueError(f"unsupported MATR model: {key.model_name}")

    _write_json_atomic(run_directory / "metrics_test.json", metrics)
    _write_metrics_csv(run_directory / "metrics_test.csv", [metrics])
    (run_directory / "model_card.md").write_text(
        _model_card(context=context, metrics=metrics), encoding="utf-8"
    )
    _write_completed_run_manifest(run_directory, context)
    return metrics


def _cycle_metrics(
    batch: CycleLifeCurveBatch,
    predicted: np.ndarray,
    context: CheckpointContext,
    *,
    calibration_batch: CycleLifeCurveBatch,
    calibration_predicted: np.ndarray,
    split_manifest: SplitManifest,
) -> dict[str, Any]:
    predictions = _cycle_predictions(batch, predicted, context)
    calibration_predictions = _cycle_predictions(
        calibration_batch,
        calibration_predicted,
        context,
    )
    calibration = calibrate_cycle_life_conformal(
        calibration_predictions,
        split_manifest=split_manifest,
        alpha=0.1,
    )
    intervals = tuple(
        make_cycle_life_interval(prediction, calibration) for prediction in predictions
    )
    coverage = evaluate_cycle_life_interval_coverage(
        intervals,
        split_manifest=split_manifest,
    )
    metrics = evaluate_cycle_life_predictions(predictions)
    return {
        "dataset_id": "MATR",
        "target": metrics.target.value,
        "model": context.model_name,
        "cutoff_cycle": context.cutoff_cycle,
        "seed": context.seed,
        "evaluated_cell_count": metrics.evaluated_cell_count,
        "mae": metrics.mae_cycle,
        "rmse": metrics.rmse_cycle,
        "mape": metrics.mape_percent,
        "r2": metrics.r2,
        "warnings": metrics.warnings,
        "conformal_alpha": calibration.alpha,
        "conformal_calibration_cell_count": calibration.calibration_cell_count,
        "conformal_residual_quantile_cycle": calibration.residual_quantile_cycle,
        "picp": coverage.picp,
        "mpiw_cycle": coverage.mpiw_cycle,
        "conformal_warnings": coverage.warnings,
    }


def _cycle_predictions(
    batch: CycleLifeCurveBatch,
    predicted: np.ndarray,
    context: CheckpointContext,
) -> list[CycleLifePrediction]:
    return [
        CycleLifePrediction(
            dataset_id="MATR",
            cell_id=cell_id,
            cutoff_cycle=context.cutoff_cycle,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            predicted_cycle=float(estimate),
            observed_cycle=int(observed),
            right_censored=False,
            feature_version=context.feature_version,
            split_version=context.split_version,
            model_version=context.model_name,
            data_version=context.data_version,
        )
        for cell_id, estimate, observed in zip(
            batch.cell_ids,
            predicted,
            batch.observed_cycles.tolist(),
            strict=True,
        )
    ]


def _validation_cycle_metrics(
    batch: CycleLifeCurveBatch,
    predicted: np.ndarray,
    context: CheckpointContext,
) -> dict[str, float | None]:
    metrics = evaluate_cycle_life_predictions(_cycle_predictions(batch, predicted, context))
    return {
        "mae": metrics.mae_cycle,
        "rmse": metrics.rmse_cycle,
        "mape": metrics.mape_percent,
        "r2": metrics.r2,
    }


def _write_static_baseline_logs(
    run_directory: Path,
    *,
    context: CheckpointContext,
    training_time_seconds: float,
    validation_metrics: dict[str, float | None],
) -> None:
    event = {
        "event": "fit_and_validation_complete",
        "dataset": context.dataset_id,
        "target": context.target.value,
        "model": context.model_name,
        "cutoff_cycle": context.cutoff_cycle,
        "seed": context.seed,
        "epoch": 0,
        "training_time_seconds": training_time_seconds,
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mape": validation_metrics["mape"],
        "validation_r2": validation_metrics["r2"],
        "gpu_memory_allocated_bytes": 0,
        "gpu_memory_reserved_bytes": 0,
    }
    _write_text_atomic(
        run_directory / "training_log.jsonl",
        json.dumps(event, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _write_metrics_csv(
        run_directory / "metrics_epoch.csv",
        [
            {
                "dataset": context.dataset_id,
                "target": context.target.value,
                "model": context.model_name,
                "cutoff_cycle": context.cutoff_cycle,
                "seed": context.seed,
                "epoch": 0,
                "training_time_seconds": training_time_seconds,
            }
        ],
    )
    _write_metrics_csv(
        run_directory / "metrics_validation.csv",
        [
            {
                "dataset": context.dataset_id,
                "target": context.target.value,
                "model": context.model_name,
                "cutoff_cycle": context.cutoff_cycle,
                "seed": context.seed,
                "epoch": 0,
                **validation_metrics,
            }
        ],
    )
    print(
        f"dataset={context.dataset_id} model={context.model_name} "
        f"cutoff={context.cutoff_cycle} seed={context.seed} epoch=0 "
        f"validation_mae={validation_metrics['mae']} "
        f"training_time_seconds={training_time_seconds:.6f}"
    )


def _write_cycle_plot(
    run_directory: Path,
    batch: CycleLifeCurveBatch,
    predicted: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    radius = metrics.get("conformal_residual_quantile_cycle")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool):
        raise ValueError("cycle-life plot requires a conformal residual radius")
    write_cycle_life_evaluation_plot(
        run_directory / "plots" / "predicted_vs_observed.png",
        batch=batch,
        predicted=predicted,
        interval_radius_cycle=float(radius),
    )


def _save_safe_tensor_artifact(model: torch.nn.Module, artifact_directory: Path) -> None:
    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    if any(
        tensor.is_floating_point() and not torch.isfinite(tensor).all()
        for tensor in weights.values()
    ):
        raise ValueError("formal model weights must be finite")
    path = artifact_directory / "model.safetensors"
    save_file(weights, str(path))
    _write_json_atomic(
        artifact_directory / "manifest.json",
        {
            "schema_version": "training-artifact-v1",
            "relative_path": "model.safetensors",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        },
    )


def _aggregate(
    metrics: list[dict[str, Any]],
    *,
    mode: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    metric_names = (
        "mae",
        "rmse",
        "mape",
        "r2",
        "picp",
        "mpiw_cycle",
        "monotonic_violation_rate",
        "best_epoch",
        "last_epoch",
        "best_iteration",
        "training_time_seconds",
        "peak_gpu_memory_bytes",
    )
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for row in metrics:
        value = row.get("mae")
        seed = row.get("seed")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            and isinstance(seed, int)
            and not isinstance(seed, bool)
        ):
            key = (str(row["model"]), int(row["cutoff_cycle"]), str(row["target"]))
            groups.setdefault(key, []).append(row)
        else:
            failures.append(row)
    summaries: list[dict[str, Any]] = []
    expected_seed_set = set(expected_seeds)
    for key, rows in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in rows)
        if len(seeds) != len(set(seeds)):
            failures.extend(rows)
            continue
        aggregated: dict[str, dict[str, float | int]] = {}
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in rows
                if isinstance(row.get(metric_name), (int, float))
                and not isinstance(row.get(metric_name), bool)
                and np.isfinite(row[metric_name])
            ]
            if values:
                aggregated[metric_name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        mae = aggregated["mae"]
        summaries.append(
            {
                "model": key[0],
                "cutoff_cycle": key[1],
                "target": key[2],
                "run_count": len(rows),
                "seeds": seeds,
                "complete_seed_matrix": set(seeds) == expected_seed_set,
                "metrics": aggregated,
                "mae_mean": mae["mean"],
                "mae_std": mae["std"],
            }
        )
    complete = bool(summaries) and all(
        bool(summary["complete_seed_matrix"]) for summary in summaries
    )
    return {
        "schema_version": "matr-aggregate-metrics-v1",
        "mode": mode,
        "formal_performance_claim": mode == "final" and complete and not failures,
        "expected_seed_count": len(expected_seeds),
        "run_count": len(metrics),
        "summaries": summaries,
        "failed_metric_rows": failures,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def _log_mlflow(
    *,
    run_root: Path,
    context: CheckpointContext,
    metrics: dict[str, Any],
    required: bool,
) -> None:
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        if required:
            raise RuntimeError("formal training requires MLflow") from None
        return
    tracking = run_root / "mlruns"
    tracking.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment("quanxin-matr-a100")
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
    }
    with mlflow.start_run(run_name=context.run_id):
        mlflow.log_params(
            {
                "dataset_id": context.dataset_id,
                "target": context.target.value,
                "model": context.model_name,
                "cutoff_cycle": context.cutoff_cycle,
                "seed": context.seed,
                "data_version": context.data_version,
                "split_version": context.split_version,
                "feature_version": context.feature_version,
            }
        )
        mlflow.log_metrics(numeric)


def _source_commit(root: Path) -> str:
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
        if not revision_path.is_file():
            raise ValueError("training package requires a source revision") from None
        value = str(json.loads(revision_path.read_text(encoding="utf-8"))["git_commit"])
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("source revision must be a lowercase Git commit hash")
    return value


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _model_card(*, context: CheckpointContext, metrics: dict[str, Any]) -> str:
    return (
        "# 泉芯智寿模型卡\n\n"
        f"- 数据集: {context.dataset_id}\n"
        f"- 目标: {context.target.value}\n"
        f"- 模型: {context.model_name}\n"
        f"- 截断循环: {context.cutoff_cycle}\n"
        f"- 随机种子: {context.seed}\n"
        f"- 数据版本: {context.data_version}\n"
        f"- 测试指标: `{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}`\n\n"
        "本结果来自公开 MATR 实验室电芯, 不代表海辰或济南企业工业电芯验证。\n"
    )


def _load_completed_run(
    run_directory: Path, context: CheckpointContext
) -> dict[str, Any] | None:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.exists():
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("completed run manifest must be a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "completed-run-v2":
        raise ValueError("completed run manifest schema is invalid")
    expected_provenance = _completed_run_provenance(context)
    if any(payload.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("completed run explicit context does not match the requested task")
    expected_context = sha256_canonical(context.model_dump(mode="json"))
    if payload.get("context_sha256") != expected_context:
        raise ValueError("completed run context does not match the requested task")
    file_rows = payload.get("files")
    if not isinstance(file_rows, list):
        raise ValueError("completed run file inventory is invalid")
    expected_paths: set[str] = set()
    for row in file_rows:
        if not isinstance(row, dict):
            raise ValueError("completed run file row is invalid")
        relative = row.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("completed run file path is invalid")
        path = _inside(run_directory, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError("completed run evidence file is missing")
        if path.stat().st_size != row.get("size_bytes") or _sha256_file(path) != row.get(
            "sha256"
        ):
            raise ValueError("completed run evidence hash or size mismatch")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name != "run_manifest.json"
        and "checkpoints" not in path.relative_to(run_directory).parts
    }
    if actual_paths != expected_paths:
        raise ValueError("completed run contains unregistered evidence files")
    metrics = json.loads((run_directory / "metrics_test.json").read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("completed run test metrics are invalid")
    return metrics


def _write_completed_run_manifest(
    run_directory: Path, context: CheckpointContext
) -> None:
    files = []
    for path in sorted(run_directory.rglob("*")):
        if (
            not path.is_file()
            or path.name == "run_manifest.json"
            or "checkpoints" in path.relative_to(run_directory).parts
        ):
            continue
        files.append(
            {
                "relative_path": path.relative_to(run_directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json_atomic(
        run_directory / "run_manifest.json",
        {
            "schema_version": "completed-run-v2",
            **_completed_run_provenance(context),
            "context_sha256": sha256_canonical(context.model_dump(mode="json")),
            "files": files,
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def _completed_run_provenance(context: CheckpointContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "dataset_id": context.dataset_id,
        "target": context.target.value,
        "model_name": context.model_name,
        "cutoff_cycle": context.cutoff_cycle,
        "seed": context.seed,
        "config_sha256": context.config_sha256,
        "input_bundle_sha256": context.input_bundle_sha256,
        "source_commit": context.source_commit,
        "data_version": context.data_version,
        "split_version": context.split_version,
        "feature_version": context.feature_version,
    }


def _inside(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = (root / Path(relative)).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise ValueError("training path escapes the project root")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
