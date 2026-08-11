from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from quanxin_life.core import TrainingTaskType, sha256_canonical
from quanxin_life.data.model_views.schemas import ModelViewArtifact, ModelViewManifest
from quanxin_life.training import pbt_magnet_runner

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "run_pbt_magnet_training.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(project: Path) -> None:
    config_root = project / "configs" / "training" / "pbt"
    config_root.mkdir(parents=True)
    config = {
        "schema_version": "pbt-offline-training-v1",
        "mode": "select",
        "enabled": True,
        "adapter_version": "pbt-offline-v1",
        "upstream_commit": "a2df9d36db3f57ab2c5686638952ad7715ea0646",
        "license_status": "VERIFIED_LICENSE_PRESENT",
        "dataset_ids": ["MATR", "HUST"],
        "model_view_version": "tiny-v1",
        "cutoff_cycle": 3,
        "model": {
            "curve_length": 4,
            "early_cycle_threshold": 1,
            "d_model": 8,
            "n_heads": 2,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "d_ff": 7,
            "dropout": 0.0,
            "num_experts": 4,
            "num_general_experts": 1,
            "condition_embedding_dim": 6,
            "gate_d_ff": 5,
            "top_k": 2,
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 0.000025,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 4,
            "max_epochs": 5,
        },
    }
    for stage in ("selection", "final"):
        stage_config = {**config, "mode": "select" if stage == "selection" else "final"}
        (config_root / f"{stage}.json").write_text(
            json.dumps(stage_config, sort_keys=True) + "\n", encoding="utf-8"
        )

    view = project / "data" / "model_views" / "pbt" / "tiny"
    view.mkdir(parents=True)
    tensors = view / "tensors.safetensors"
    generator = torch.Generator().manual_seed(38)
    save_file(
        {
            "curves": torch.randn(8, 3, 3, 4, generator=generator),
            "condition_embeddings": torch.randn(8, 6, generator=generator),
            "targets": torch.linspace(10, 17, 8),
            "valid_cycle_mask": torch.ones(8, 3, dtype=torch.bool),
            "expert_mask": torch.ones(8, 4, dtype=torch.bool),
        },
        str(tensors),
    )
    metadata = view / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "entity_ids": [f"MATR-{index}" for index in range(8)],
                "splits": [
                    "train",
                    "train",
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "test",
                    "test",
                ],
                "domains": ["MATR"] * 8,
                "seen": [True] * 8,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    split_path = view / "split_hust_zero_shot.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "pbt-cell-split-v1",
                "experiment": "hust_zero_shot",
                "membership": {
                    "MATR-0": "train",
                    "MATR-1": "train",
                    "MATR-2": "excluded",
                    "MATR-3": "calibration",
                    "MATR-4": "validation",
                    "MATR-5": "validation",
                    "MATR-6": "test",
                    "MATR-7": "test",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = tuple(
        ModelViewArtifact(
            relative_path=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in sorted((metadata, split_path, tensors), key=lambda item: item.name)
    )
    manifest = ModelViewManifest(
        schema_version="model-view-manifest-v2",
        view_id="pbt_multidomain_early_life",
        view_version="tiny-v1",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="observed_eol_cycle_life",
        mask_semantics=("valid_cycle_mask", "expert_mask"),
        cutoff_cycle=3,
        canonical_sha256="0" * 64,
        split_sha256="1" * 64,
        builder_version="test",
        builder_code_sha256="2" * 64,
        config_sha256="3" * 64,
        normalization_sha256="4" * 64,
        training_entity_ids_sha256="5" * 64,
        row_count=8,
        entity_key="cell_id",
        source_dataset_ids=("MATR", "HUST"),
        artifacts=artifacts,
    )
    (view / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (view / "COMMITTED").write_text(
        sha256_canonical(manifest.model_dump(mode="json")) + "\n", encoding="ascii"
    )


def _write_magnet_fixture(project: Path) -> None:
    config_root = project / "configs" / "training" / "magnet"
    config_root.mkdir(parents=True)
    config = {
        "schema_version": "magnet-training-v2",
        "mode": "select",
        "enabled": True,
        "adapter_version": "magnet-upstream-informer-mldg-v2",
        "upstream_commit": "aafb90c551d20748251a35fd51a34eae2539aaca",
        "license_status": "VERIFIED_LICENSE_PRESENT",
        "model": {
            "seq_len": 4,
            "label_len": 2,
            "prediction_horizon": 3,
            "d_model": 4,
            "n_heads": 1,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "d_ff": 4,
            "factor": 1,
            "factor2": 1,
            "dropout": 0.0,
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 0.000001,
            "meta_learning_rate": 0.0075,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 4,
            "max_epochs": 5,
        },
    }
    for stage in ("selection", "final"):
        stage_config = {**config, "mode": "select" if stage == "selection" else "final"}
        (config_root / f"{stage}.json").write_text(
            json.dumps(stage_config, sort_keys=True) + "\n", encoding="utf-8"
        )

    view = project / "data" / "model_views" / "magnet" / "tiny"
    view.mkdir(parents=True)
    rows = 8
    tensor_path = view / "tensors.safetensors"
    generator = torch.Generator().manual_seed(71)
    history = torch.rand((rows, 4, 2), generator=generator)
    targets = torch.rand((rows, 3, 2), generator=generator)
    save_file(
        {
            "history": history,
            "history_cycle": torch.arange(4).reshape(1, 4, 1).repeat(rows, 1, 1).float(),
            "decoder_known": history[:, -2:, :].clone(),
            "targets": targets,
            "target_cycle": torch.arange(2, 7).reshape(1, 5, 1).repeat(rows, 1, 1).float(),
            "observation_mask": torch.ones((rows, 3, 2), dtype=torch.bool),
            "cycle_distance": torch.linspace(5, 12, rows).reshape(rows, 1),
            "condition_vectors": torch.tensor(
                [[25.0, 1.0, 1.0, 1.0, 0.5]] * 2
                + [[35.0, 1.0, 1.0, 0.8, 0.5]] * 2
                + [[45.0, 0.5, 1.0, 1.0, 0.8]] * 2
                + [[15.0, 1.0, 0.5, 0.6, 0.2]] * 2
            ),
        },
        str(tensor_path),
    )
    metadata_path = view / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "entity_ids": [f"cell-{index}" for index in range(rows)],
                "condition_keys": ["train-a"] * 2
                + ["train-b"] * 2
                + ["validation-c"] * 2
                + ["test-d"] * 2,
                "splits": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
                "seen": [True] * 6 + [False] * 2,
                "condition_feature_names": [
                    "temperature_c",
                    "charge_rate_c",
                    "discharge_rate_c",
                    "dod_fraction",
                    "soc_fraction",
                ],
                "protocols": ["CCCV"] * rows,
                "monotonic_applicable": [True] * rows,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = tuple(
        ModelViewArtifact(
            relative_path=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in sorted((metadata_path, tensor_path), key=lambda item: item.name)
    )
    manifest = ModelViewManifest(
        schema_version="model-view-manifest-v2",
        view_id="magnet_multicondition_qd_ed",
        view_version="tiny-v1",
        task_type=TrainingTaskType.CONDITION_DEGRADATION,
        target_semantics="observed_qd_ed_trajectory",
        mask_semantics=("observation_mask",),
        canonical_sha256="6" * 64,
        split_sha256="7" * 64,
        builder_version="test",
        builder_code_sha256="8" * 64,
        config_sha256="9" * 64,
        normalization_sha256="a" * 64,
        training_entity_ids_sha256="b" * 64,
        row_count=rows,
        entity_key="cell_id",
        source_dataset_ids=("NAUMANN_CYCLE",),
        artifacts=artifacts,
    )
    (view / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (view / "COMMITTED").write_text(
        sha256_canonical(manifest.model_dump(mode="json")) + "\n", encoding="ascii"
    )


def _run(
    project: Path, stage: str = "selection", model: str = "pbt"
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["QUANXIN_SOURCE_COMMIT"] = "a" * 40
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--models",
            model,
            "--stage",
            stage,
            "--seeds",
            "38",
            "--device",
            "cpu",
            "--project-root",
            str(project),
            *(["--allow-partial-matrix"] if stage == "final" else []),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_selection_runs_real_pbt_and_repeated_command_skips(tmp_path: Path) -> None:
    project = tmp_path / "package"
    _write_fixture(project)

    first = _run(project)

    assert first.returncode == 0, first.stderr
    success_files = tuple((project / "runs").rglob("SUCCESS.json"))
    assert len(success_files) == 1
    run = success_files[0].parent
    required = {
        "run_manifest.json",
        "stdout.log",
        "events.jsonl",
        "resource.csv",
        "metrics_epoch.csv",
        "metrics_validation.csv",
        "SUCCESS.json",
    }
    assert required <= {path.name for path in run.iterdir()}
    assert (run / "best" / "manifest.json").is_file()
    assert (run / "last" / "manifest.json").is_file()
    assert not (run / "metrics_test.json").exists()
    success = json.loads(success_files[0].read_text(encoding="utf-8"))
    assert success["test_read"] is False
    before = success_files[0].stat().st_mtime_ns

    second = _run(project)

    assert second.returncode == 0, second.stderr
    assert "SKIPPED_COMPLETED" in second.stdout
    assert success_files[0].stat().st_mtime_ns == before


def test_prepared_pbt_manifest_hashes_the_experiment_train_cells(tmp_path: Path) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    view = pbt_magnet_runner._discover_view_tasks(
        project, ("pbt",), formal=False
    )[0]
    run_root = project / "prepared-run"
    run_root.mkdir()

    _prepared_root, prepared_manifest = pbt_magnet_runner._prepare_view(
        run_root, view
    )

    assert prepared_manifest.training_entity_ids_sha256 == sha256_canonical(
        ["MATR-0", "MATR-1"]
    )


def test_failure_returns_nonzero_without_success(tmp_path: Path) -> None:
    project = tmp_path / "empty-package"
    project.mkdir()

    result = _run(project)

    assert result.returncode != 0
    assert not tuple(project.rglob("SUCCESS.json"))


def test_oom_reduces_micro_batch_and_preserves_effective_batch(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    monkeypatch.setenv("QUANXIN_SOURCE_COMMIT", "a" * 40)
    original = pbt_magnet_runner._EngineTask.train_epoch
    attempts = 0

    def fail_once(self, epoch, *, device):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original(self, epoch, device=device)

    monkeypatch.setattr(pbt_magnet_runner._EngineTask, "train_epoch", fail_once)

    result = pbt_magnet_runner.run_stage(
        project_root=project,
        models=("pbt",),
        stage="selection",
        seeds=(38,),
        device="cpu",
        enforce_formal_matrix=False,
    )

    assert result["status"] == "COMPLETED"
    run = next((project / "runs").rglob("SUCCESS.json")).parent
    backoff = [
        json.loads(line)
        for line in (run / "oom_backoff.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert backoff == [
        {
            "effective_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "micro_batch_size": 2,
        }
    ]


def test_interrupted_training_resumes_from_last_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    monkeypatch.setenv("QUANXIN_SOURCE_COMMIT", "a" * 40)
    original = pbt_magnet_runner._EngineTask.train_epoch

    def interrupt_on_second_epoch(self, epoch, *, device):
        if epoch == 2:
            raise RuntimeError("simulated power loss")
        return original(self, epoch, device=device)

    monkeypatch.setattr(
        pbt_magnet_runner._EngineTask, "train_epoch", interrupt_on_second_epoch
    )
    try:
        pbt_magnet_runner.run_stage(
            project_root=project,
            models=("pbt",),
            stage="selection",
            seeds=(38,),
            device="cpu",
            enforce_formal_matrix=False,
        )
    except RuntimeError as exc:
        assert "simulated power loss" in str(exc)
    else:
        raise AssertionError("simulated interruption did not stop training")

    run = next((project / "runs").rglob("run_manifest.json")).parent
    assert (run / "checkpoints" / "last.json").is_file()
    assert not (run / "SUCCESS.json").exists()

    monkeypatch.setattr(pbt_magnet_runner._EngineTask, "train_epoch", original)
    resumed = pbt_magnet_runner.run_stage(
        project_root=project,
        models=("pbt",),
        stage="selection",
        seeds=(38,),
        device="cpu",
        enforce_formal_matrix=False,
    )

    assert resumed["status"] == "COMPLETED"
    assert (run / "SUCCESS.json").is_file()
    events = [
        json.loads(line)
        for line in (run / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["epoch"] for event in events] == [1, 2, 3, 4, 5]


def test_tiny_training_completes_with_network_sockets_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    monkeypatch.setenv("QUANXIN_SOURCE_COMMIT", "a" * 40)

    def reject_network(*args, **kwargs):
        raise AssertionError("network access is forbidden in the closed training runtime")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)

    result = pbt_magnet_runner.run_stage(
        project_root=project,
        models=("pbt",),
        stage="selection",
        seeds=(38,),
        device="cpu",
        enforce_formal_matrix=False,
    )

    assert result["status"] == "COMPLETED"


def test_source_commit_falls_back_to_offline_package_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    revision = "c" * 40
    (tmp_path / "package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "quanxin-pbt-magnet-a100-v1",
                "source_commit": revision,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(pbt_magnet_runner.subprocess, "run", unavailable)

    assert pbt_magnet_runner._source_commit(tmp_path) == revision


def test_frozen_selection_final_test_and_collection_are_closed(tmp_path: Path) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    assert _run(project, "selection").returncode == 0

    frozen = _run(project, "freeze-selection")
    first_marker = (
        project
        / "runs"
        / "a100"
        / "pbt_magnet"
        / "selection_frozen"
        / "COMMITTED"
    )
    assert frozen.returncode == 0, frozen.stderr
    digest_before = first_marker.read_text(encoding="ascii")
    assert _run(project, "freeze-selection").returncode == 0
    assert first_marker.read_text(encoding="ascii") == digest_before

    final = _run(project, "final")

    assert final.returncode == 0, final.stderr
    successes = tuple(
        (project / "runs" / "a100" / "pbt_magnet" / "final").rglob("SUCCESS.json")
    )
    assert len(successes) == 1
    run = successes[0].parent
    assert (run / "metrics_test.json").is_file()
    assert (run / "predictions.parquet").is_file()
    assert json.loads(successes[0].read_text(encoding="utf-8"))["test_read"] is True
    metrics = json.loads((run / "metrics_test.json").read_text(encoding="utf-8"))
    assert metrics["conformal"]["calibration_count"] == 1

    collected = _run(project, "collect")

    assert collected.returncode == 0, collected.stderr
    summary = project / "runs" / "a100" / "pbt_magnet" / "summary"
    assert (summary / "predictions.parquet").is_file()
    assert (summary / "seed_summary.json").is_file()
    assert (summary / "predicted_vs_true.parquet").is_file()
    assert (summary / "residuals.parquet").is_file()
    assert (summary / "training_validation_loss.parquet").is_file()
    assert (summary / "domain_condition_comparison.parquet").is_file()
    assert (summary / "seed_stability.csv").is_file()
    assert (summary / "ood_results.parquet").is_file()
    assert (summary / "SUCCESS.json").is_file()


def test_seed42_recovery_is_isolated_and_single_seed_uncertainty_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "package"
    _write_fixture(project)
    monkeypatch.setenv("QUANXIN_SOURCE_COMMIT", "a" * 40)

    common = {
        "project_root": project,
        "models": ("pbt",),
        "seeds": (42,),
        "device": "cpu",
        "enforce_formal_matrix": False,
        "recovery_seed42": True,
    }
    assert pbt_magnet_runner.run_stage(stage="selection", **common)["status"] == "COMPLETED"
    assert pbt_magnet_runner.run_stage(stage="freeze-selection", **common)["status"] in {
        "FROZEN",
        "SKIPPED_FROZEN",
    }
    assert pbt_magnet_runner.run_stage(stage="final", **common)["status"] == "COMPLETED"
    collected = pbt_magnet_runner.run_stage(stage="collect", **common)

    assert collected["status"] == "SUCCESS"
    recovery = project / "runs" / "a100" / "pbt_magnet_recovery_seed42_v2"
    assert (recovery / "final").is_dir()
    assert not (project / "runs" / "a100" / "pbt_magnet" / "final").exists()
    summary = json.loads((recovery / "summary" / "seed_summary.json").read_text())
    assert summary
    for metric in summary.values():
        assert metric["count"] == 1
        assert metric["std"] is None
        assert metric["ci95_low"] is None
        assert metric["ci95_high"] is None
        assert metric["uncertainty_status"] == "UNAVAILABLE_SINGLE_SEED"


def test_selection_runs_real_magnet_support_query_lifecycle(tmp_path: Path) -> None:
    project = tmp_path / "package"
    _write_magnet_fixture(project)

    result = _run(project, "selection", "magnet")

    assert result.returncode == 0, result.stderr
    success = next((project / "runs").rglob("SUCCESS.json"))
    run = success.parent
    events = [
        json.loads(line)
        for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["train_support_loss"] >= 0
    assert events[-1]["train_query_loss"] >= 0
    assert events[-1]["train_meta_outer_loss"] >= 0
    validation = (run / "metrics_validation.csv").read_text(encoding="utf-8")
    assert "validation_observed_mae" in validation


def test_magnet_final_records_seen_query_and_unseen_test_metrics(tmp_path: Path) -> None:
    project = tmp_path / "package"
    _write_magnet_fixture(project)
    assert _run(project, "selection", "magnet").returncode == 0
    assert _run(project, "freeze-selection", "magnet").returncode == 0

    final = _run(project, "final", "magnet")

    assert final.returncode == 0, final.stderr
    metrics_path = next(
        (project / "runs" / "a100" / "pbt_magnet" / "final").rglob(
            "metrics_test.json"
        )
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["seen_condition_query_loss"] >= 0
    assert metrics["unseen_condition_observed_mae"] >= 0
