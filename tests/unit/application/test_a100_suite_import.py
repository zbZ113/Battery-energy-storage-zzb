from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.application.a100_suite_import import A100SuiteRunImporter
from quanxin_life.core import sha256_canonical
from quanxin_life.training.suite import MatrThreeBatchRunConfig

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
SOURCE_COMMIT = "d" * 40
TRANSFER_BYTES = b"reviewed A100 result transfer archive"
TRANSFER_SHA256 = hashlib.sha256(TRANSFER_BYTES).hexdigest()
MODELS = ("dummy", "variance", "xgboost", "cpmlp", "hybrid")


def _write_config(path: Path, *, mode: str = "smoke") -> MatrThreeBatchRunConfig:
    seeds = [20260712] if mode == "smoke" else [
        20260712,
        20260713,
        20260714,
        20260715,
        20260716,
    ]
    max_epochs = {
        "dummy": 1,
        "variance": 1,
        "xgboost": 10 if mode == "smoke" else 2000,
        "cpmlp": 2 if mode == "smoke" else 300,
        "hybrid": 2 if mode == "smoke" else 500,
    }
    validation_interval = {
        "dummy": 1,
        "variance": 1,
        "xgboost": 1,
        "cpmlp": 1 if mode == "smoke" else 5,
        "hybrid": 1 if mode == "smoke" else 5,
    }
    patience = {
        "dummy": 1,
        "variance": 1,
        "xgboost": 2 if mode == "smoke" else 100,
        "cpmlp": 2 if mode == "smoke" else 10,
        "hybrid": 2 if mode == "smoke" else 10,
    }
    payload = {
        "schema_version": "matr-three-batch-run-config-v1",
        "mode": mode,
        "suite": {
            "schema_version": "training-suite-v1",
            "dataset_id": "MATR",
            "target": "matr_official_cycle_life",
            "cutoffs": [20, 50, 100, 150],
            "seeds": seeds,
            "models": [
                {
                    "name": model,
                    "max_epochs": max_epochs[model],
                    "validation_interval": validation_interval[model],
                    "early_stopping_patience": patience[model],
                }
                for model in MODELS
            ],
            "physical_gpu_index": 1,
            "precision": "fp32",
            "batch_size": 64,
            "data_version": "matr-three-batch-test-v1",
            "split_version": "matr-three-batch-split-test-v1",
            "feature_version": "matr-three-batch-features-test-v1",
        },
        "paths": {
            "three_batch_manifest": "reports/data_quality/three-batch.json",
            "split_manifest": "configs/data_splits/three-batch.json",
            "run_root": f"runs/a100/matr-three-batch/{mode}",
        },
        "xgboost_early_stopping_rounds": patience["xgboost"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return MatrThreeBatchRunConfig.model_validate(payload)


def _write_transfer(path: Path) -> Path:
    path.write_bytes(TRANSFER_BYTES)
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_task(
    root: Path,
    *,
    config: MatrThreeBatchRunConfig,
    cutoff: int,
    model: str,
    seed: int,
) -> dict[str, object]:
    task = root / f"cutoff-{cutoff}" / model / f"seed-{seed}"
    (task / "artifacts").mkdir(parents=True, exist_ok=True)
    (task / "plots").mkdir(exist_ok=True)
    context = {
        "run_id": str(uuid4()),
        "dataset_id": "MATR",
        "target": "matr_official_cycle_life",
        "model_name": model,
        "cutoff_cycle": cutoff,
        "seed": seed,
        "config_sha256": config.suite.config_sha256,
        "input_bundle_sha256": "a" * 64,
        "source_commit": SOURCE_COMMIT,
        "data_version": config.suite.data_version,
        "split_version": config.suite.split_version,
        "feature_version": config.suite.feature_version,
    }
    files = {
        "artifacts/model.safetensors": b"safe tensor bytes",
        "config_resolved.json": (json.dumps(context) + "\n").encode(),
        "environment.json": b'{"device":"cuda:0"}\n',
        "metrics_epoch.csv": b"epoch,train_loss\n1,1.0\n",
        "metrics_test.csv": b"model,mae\ndummy,1.0\n",
        "metrics_test.json": (
            json.dumps(
                {
                    "model": model,
                    "cutoff_cycle": cutoff,
                    "seed": seed,
                    "target": "matr_official_cycle_life",
                    "mae": 1.0,
                }
            )
            + "\n"
        ).encode(),
        "metrics_validation.csv": b"epoch,mae\n1,1.0\n",
        "model_card.md": b"# Model card\n",
        "plots/predicted.png": b"\x89PNG\r\n\x1a\n",
        "training_log.jsonl": b'{"epoch":1}\n',
    }
    for relative, payload in files.items():
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest_files = [
        {
            "relative_path": relative,
            "size_bytes": (task / relative).stat().st_size,
            "sha256": _hash(task / relative),
        }
        for relative in sorted(files)
    ]
    manifest = {
        "schema_version": "completed-run-v2",
        **context,
        "context_sha256": sha256_canonical(context),
        "files": manifest_files,
        "completed_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    (task / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return json.loads((task / "metrics_test.json").read_text(encoding="utf-8"))


def _write_suite(root: Path, config: MatrThreeBatchRunConfig) -> None:
    metrics = [
        _write_task(root, config=config, cutoff=cutoff, model=model, seed=seed)
        for cutoff in config.suite.cutoffs
        for model in MODELS
        for seed in config.suite.seeds
    ]
    aggregate = {
        "schema_version": "matr-aggregate-metrics-v1",
        "mode": config.mode,
        "formal_performance_claim": config.mode == "final",
        "expected_seed_count": len(config.suite.seeds),
        "run_count": len(metrics),
        "summaries": [
            {
                "model": model,
                "cutoff_cycle": cutoff,
                "target": "matr_official_cycle_life",
                "run_count": len(config.suite.seeds),
                "seeds": list(config.suite.seeds),
                "complete_seed_matrix": True,
                "metrics": {
                    "mae": {
                        "count": len(config.suite.seeds),
                        "mean": 1.0,
                        "std": 0.0,
                    }
                },
                "mae_mean": 1.0,
                "mae_std": 0.0,
            }
            for cutoff in config.suite.cutoffs
            for model in MODELS
        ],
        "failed_metric_rows": [],
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    (root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate),
        encoding="utf-8",
    )
    (root / "aggregate_metrics.csv").write_text(
        "model,cutoff_cycle,target\n",
        encoding="utf-8",
    )
    (root / "experiment_summary.md").write_text(
        "# MATR experiment summary\n",
        encoding="utf-8",
    )
    with (root / "metrics_test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def test_importer_verifies_complete_matrix_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "downloaded"
    source.mkdir()
    config_path = tmp_path / "smoke.json"
    config = _write_config(config_path)
    _write_suite(source, config)
    registry = tmp_path / "registry"
    importer = A100SuiteRunImporter(registry)
    transfer = _write_transfer(tmp_path / "results.tgz")

    first = importer.import_run(
        source,
        config_path=config_path,
        expected_source_commit=SOURCE_COMMIT,
        transfer_archive=transfer,
        transfer_sha256=TRANSFER_SHA256,
        imported_at=NOW,
    )
    second = importer.import_run(
        source,
        config_path=config_path,
        expected_source_commit=SOURCE_COMMIT,
        transfer_archive=transfer,
        transfer_sha256=TRANSFER_SHA256,
        imported_at=NOW.replace(hour=13),
    )

    assert first == second
    assert first.task_count == 20
    assert first.mode == "smoke"
    assert first.formal_performance_claim is False
    assert first.source_commit == SOURCE_COMMIT
    assert len(first.output_sha256) == 64
    resolved = importer.resolve(first.import_id)
    assert resolved.record == first
    assert resolved.output_root.is_dir()
    (resolved.output_root / "aggregate_metrics.csv").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output SHA-256"):
        importer.resolve(first.import_id)


def test_importer_rejects_missing_task_tampering_and_unsafe_formats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "downloaded"
    source.mkdir()
    config_path = tmp_path / "smoke.json"
    config = _write_config(config_path)
    _write_suite(source, config)
    importer = A100SuiteRunImporter(tmp_path / "registry")
    transfer = _write_transfer(tmp_path / "results.tgz")

    missing = source / "cutoff-150" / "hybrid" / "seed-20260712"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()
    with pytest.raises(ValueError, match="matrix"):
        importer.import_run(
                source,
                config_path=config_path,
                expected_source_commit=SOURCE_COMMIT,
                transfer_archive=transfer,
                transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )

    _write_task(
        source,
        config=config,
        cutoff=150,
        model="hybrid",
        seed=20260712,
    )
    metric_path = source / "cutoff-20" / "dummy" / "seed-20260712" / "metrics_test.json"
    metric_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"SHA-256|size"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit=SOURCE_COMMIT,
            transfer_archive=transfer,
            transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )

    _write_suite(source, config)
    (source / "cutoff-20" / "dummy" / "seed-20260712" / "artifacts" / "model.pt").write_bytes(
        b"unsafe pickle-compatible artifact"
    )
    with pytest.raises(ValueError, match="forbidden"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit=SOURCE_COMMIT,
            transfer_archive=transfer,
            transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )

    (source / "cutoff-20" / "dummy" / "seed-20260712" / "artifacts" / "model.pt").unlink()
    (source / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered root"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit=SOURCE_COMMIT,
            transfer_archive=transfer,
            transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )


def test_importer_rejects_source_context_mismatch_and_nonformal_final(
    tmp_path: Path,
) -> None:
    source = tmp_path / "downloaded"
    source.mkdir()
    config_path = tmp_path / "final.json"
    config = _write_config(config_path, mode="final")
    _write_suite(source, config)
    importer = A100SuiteRunImporter(tmp_path / "registry")
    transfer = _write_transfer(tmp_path / "results.tgz")

    with pytest.raises(ValueError, match="transfer archive SHA-256"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit=SOURCE_COMMIT,
            transfer_archive=transfer,
            transfer_sha256="0" * 64,
            imported_at=NOW,
        )

    with pytest.raises(ValueError, match="source commit"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit="f" * 40,
            transfer_archive=transfer,
            transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )

    aggregate_path = source / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["formal_performance_claim"] = False
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ValueError, match="formal"):
        importer.import_run(
            source,
            config_path=config_path,
            expected_source_commit=SOURCE_COMMIT,
            transfer_archive=transfer,
            transfer_sha256=TRANSFER_SHA256,
            imported_at=NOW,
        )
