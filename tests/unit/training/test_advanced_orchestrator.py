from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quanxin_life.training import advanced_orchestrator
from quanxin_life.training.advanced_orchestrator import (
    execute_advanced_matr_three_batch_suite,
    plan_advanced_run,
)
from quanxin_life.training.engine import TrainingRunStatus

ROOT = Path(__file__).resolve().parents[3]


def test_plan_smoke_is_four_sequential_model_runs() -> None:
    plan = plan_advanced_run(ROOT, "smoke")

    assert len(plan) == 4
    assert {item.family for item in plan} == {
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    }
    assert {(item.cutoff_cycle, item.seed, item.max_epochs) for item in plan} == {(50, 38, 10)}


def test_plan_select_stage_one_uses_only_the_initial_validation_axis() -> None:
    plan = plan_advanced_run(ROOT, "select", selection_stage="selection_stage1")

    assert len(plan) == 13
    assert {(item.cutoff_cycle, item.seed, item.max_epochs) for item in plan} == {(100, 38, 30)}


def test_plan_select_later_stage_requires_explicit_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_ids"):
        plan_advanced_run(ROOT, "select", selection_stage="selection_stage2")


def test_plan_final_fails_closed_until_selection_is_bound() -> None:
    with pytest.raises(ValueError, match="selection"):
        plan_advanced_run(ROOT, "final")


def test_smoke_loads_selection_data_once_and_runs_models_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = advanced_orchestrator._load_mode_config(ROOT, "smoke").model_copy(
        update={
            "paths": advanced_orchestrator._load_mode_config(ROOT, "smoke").paths.model_copy(
                update={"run_root": "runs/test-advanced-smoke"}
            )
        }
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "_load_registered_inputs",
        lambda _root, _config: SimpleNamespace(
            manifest=object(),
            split=object(),
            input_bundle_sha256="a" * 64,
            data_version="matr-test-v1",
            split_version="split-test-v1",
            source_commit="b" * 40,
        ),
    )
    loaded: list[int] = []
    synthetic_data = SimpleNamespace(
        scalar_normalizer=SimpleNamespace(statistics_sha256="c" * 64),
        hybrid_normalizer=SimpleNamespace(statistics_sha256="d" * 64),
    )

    def load_data(**kwargs: object) -> object:
        loaded.append(int(kwargs["cutoff_cycle"]))
        return synthetic_data

    monkeypatch.setattr(advanced_orchestrator, "load_advanced_matr_selection_data", load_data)
    tasks: list[str] = []

    def build_task(**kwargs: object) -> object:
        family = str(kwargs["run_key"].family)  # type: ignore[union-attr]
        tasks.append(family)
        task = SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            optimizer=torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
            scheduler=None,
        )
        if family == "cyclepatch_batlinet":
            task.reference_library = SimpleNamespace(library_sha256="e" * 64)
        return task

    monkeypatch.setattr(advanced_orchestrator, "_build_training_task", build_task)
    runs: list[str] = []

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            self.context = kwargs["context"]

        def run(self, *, epoch_limit: int | None = None) -> object:
            runs.append(self.context.model_name)
            assert epoch_limit is None
            return SimpleNamespace(
                status=TrainingRunStatus.COMPLETED,
                best_epoch=10,
                best_metric=1.0,
                last_epoch=10,
                model_dump=lambda mode: {"status": "COMPLETED"},
            )

    monkeypatch.setattr(advanced_orchestrator, "TrainingEngine", FakeEngine)

    result = execute_advanced_matr_three_batch_suite(
        project_root=tmp_path,
        config=config,
        device=torch.device("cpu"),
    )

    assert loaded == [50]
    assert runs == tasks == [item.family for item in plan_advanced_run(ROOT, "smoke")]
    assert result["mode"] == "smoke"
    assert len(result["runs"]) == 4


def test_executor_filters_a_requested_final_seed_before_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = advanced_orchestrator._load_mode_config(ROOT, "smoke").model_copy(
        update={
            "paths": advanced_orchestrator._load_mode_config(ROOT, "smoke").paths.model_copy(
                update={"run_root": "runs/test-seed-filter"}
            )
        }
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "_load_registered_inputs",
        lambda _root, _config: SimpleNamespace(
            manifest=object(),
            split=object(),
            input_bundle_sha256="a" * 64,
            data_version="matr-test-v1",
            split_version="split-test-v1",
            source_commit="b" * 40,
        ),
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "load_advanced_matr_selection_data",
        lambda **_kwargs: object(),
    )
    seen: list[int] = []

    def run_key(**kwargs: object) -> dict[str, object]:
        key = kwargs["key"]
        seen.append(key.seed)  # type: ignore[union-attr]
        return {
            "family": key.family,  # type: ignore[union-attr]
            "candidate_id": key.candidate_id,  # type: ignore[union-attr]
            "cutoff_cycle": key.cutoff_cycle,  # type: ignore[union-attr]
            "seed": key.seed,  # type: ignore[union-attr]
        }

    monkeypatch.setattr(advanced_orchestrator, "_run_key", run_key)

    result = execute_advanced_matr_three_batch_suite(
        project_root=tmp_path,
        config=config,
        device=torch.device("cpu"),
        seed=38,
    )

    assert seen == [38, 38, 38, 38]
    assert result["run_count"] == 4


def test_aggregate_metrics_merge_seed_subprocess_results(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    first = {
        "mode": "final",
        "dataset_id": "MATR",
        "target": "matr_official_cycle_life",
        "runs": [
            {
                "family": "cyclepatch_direct",
                "candidate_id": "candidate",
                "cutoff_cycle": 20,
                "seed": 38,
            }
        ],
        "run_count": 1,
        "input_bundle_sha256": "a" * 64,
    }
    second = {
        **first,
        "runs": [
            {
                "family": "cyclepatch_direct",
                "candidate_id": "candidate",
                "cutoff_cycle": 20,
                "seed": 39,
            }
        ],
    }

    advanced_orchestrator._write_merged_aggregate(run_root, first)
    advanced_orchestrator._write_merged_aggregate(run_root, second)

    payload = json.loads((run_root / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert [row["seed"] for row in payload["runs"]] == [38, 39]


def test_source_commit_falls_back_to_packaged_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    revision = "c" * 40
    (tmp_path / "source_revision.json").write_text(
        json.dumps(
            {
                "schema_version": "source-revision-v1",
                "git_commit": revision,
                "git_dirty": False,
                "created_at": "2026-07-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", unavailable)

    assert advanced_orchestrator._source_commit(tmp_path) == revision


def _masked_current_hybrid_batch(
    cell_ids: tuple[str, ...], *, missing_cycle: int | None = None
) -> SimpleNamespace:
    prediction_cycles = torch.tensor([30, 40, 50, 60, 500], dtype=torch.int64)
    cell_count = len(cell_ids)
    target_soh = torch.tensor(
        [[0.98, 0.96, 0.94, 0.92, 0.80] for _ in cell_ids], dtype=torch.float32
    )
    target_mask = torch.ones_like(target_soh, dtype=torch.bool)
    if missing_cycle is not None:
        column = int(torch.nonzero(prediction_cycles == missing_cycle).item())
        target_mask[0, column] = False
        target_soh[0, column] = float("nan")
    return SimpleNamespace(
        cell_ids=cell_ids,
        inputs=SimpleNamespace(
            early_batch=SimpleNamespace(
                values=torch.ones((cell_count, 21, 1, 1, 1), dtype=torch.float32),
                sample_mask=torch.ones((cell_count, 21, 1, 1), dtype=torch.bool),
                cycle_mask=torch.ones((cell_count, 21), dtype=torch.bool),
            ),
            initial_soh=torch.full((cell_count,), 0.99, dtype=torch.float32),
            prediction_cycles=prediction_cycles,
        ),
        targets=SimpleNamespace(target_soh=target_soh, target_mask=target_mask),
    )


def test_current_hybrid_uses_one_shared_real_prediction_axis() -> None:
    data = SimpleNamespace(
        hybrid_train=_masked_current_hybrid_batch(
            ("train-a", "train-b"), missing_cycle=40
        ),
        hybrid_validation=_masked_current_hybrid_batch(
            ("validation-a",), missing_cycle=50
        ),
    )

    task = advanced_orchestrator._build_current_hybrid_task(
        data,
        SimpleNamespace(hidden_dim=4, learning_rate=1e-3),
        seed=38,
    )

    assert task.train_batch.prediction_cycles == (30, 60, 500)
    assert task.validation_batch.prediction_cycles == (30, 60, 500)


def test_current_hybrid_final_evaluation_reuses_training_axis(tmp_path: Path) -> None:
    data = SimpleNamespace(
        hybrid_train=_masked_current_hybrid_batch(
            ("train-a", "train-b"), missing_cycle=40
        ),
        hybrid_validation=_masked_current_hybrid_batch(
            ("validation-a",), missing_cycle=50
        ),
    )
    task = advanced_orchestrator._build_current_hybrid_task(
        data,
        SimpleNamespace(hidden_dim=4, learning_rate=1e-3),
        seed=38,
    )
    final_data = SimpleNamespace(
        hybrid_test=_masked_current_hybrid_batch(("test-a",))
    )

    advanced_orchestrator._write_final_test_metrics(
        run_directory=tmp_path,
        family="current_hybrid",
        task=task,
        data=final_data,
        device=torch.device("cpu"),
    )

    payload = json.loads((tmp_path / "metrics_test.json").read_text(encoding="utf-8"))
    assert payload["cell_count"] == 1
    assert payload["partition"] == "test"


def test_select_executes_stage1_stage2_and_full_recheck_without_final_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = advanced_orchestrator._load_mode_config(ROOT, "select").model_copy(
        update={
            "paths": advanced_orchestrator._load_mode_config(ROOT, "select").paths.model_copy(
                update={"run_root": "runs/test-selection"}
            )
        }
    )
    registered = SimpleNamespace(
        manifest=SimpleNamespace(
            batches=(SimpleNamespace(raw_sha256="d" * 64),),
            combined_split_sha256="e" * 64,
        ),
        split=SimpleNamespace(validation=("MATR_validation",)),
        input_bundle_sha256="a" * 64,
        source_commit="b" * 40,
    )
    loaded: list[int] = []
    synthetic_selection_data = SimpleNamespace(
        scalar_normalizer=SimpleNamespace(statistics_sha256="f" * 64),
        hybrid_normalizer=SimpleNamespace(statistics_sha256="0" * 64),
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "load_advanced_matr_selection_data",
        lambda **kwargs: loaded.append(int(kwargs["cutoff_cycle"]))
        or synthetic_selection_data,
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "load_advanced_matr_final_data",
        lambda **_kwargs: pytest.fail("Select must not load held-out partitions"),
    )
    runs: list[str] = []

    def run_key(**kwargs: object) -> dict[str, object]:
        key = kwargs["key"]
        runs.append(key.stage)  # type: ignore[union-attr]
        path = tmp_path / "synthetic" / str(len(runs))
        path.mkdir(parents=True)
        return {"run_directory": path.relative_to(tmp_path).as_posix()}

    monkeypatch.setattr(advanced_orchestrator, "_run_key", run_key)
    monkeypatch.setattr(
        advanced_orchestrator,
        "_validation_metric",
        lambda _path: (float(len(runs)), float(len(runs)), 0.0),
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "_build_baseline_evidence",
        lambda **_kwargs: (),
    )
    synthetic_manifest = SimpleNamespace(
        model_dump=lambda mode: {},
        selection=SimpleNamespace(selected_candidates=()),
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "build_advanced_model_selection_manifest",
        lambda **_kwargs: synthetic_manifest,
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "_write_resolved_final_config",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        advanced_orchestrator,
        "_sha256_file",
        lambda _path: "c" * 64,
    )

    result = advanced_orchestrator._execute_selection(
        root=tmp_path,
        config=config,
        registered=registered,
        run_root=tmp_path / "runs" / "test-selection",
        device=torch.device("cpu"),
    )

    assert runs.count("selection_stage1") == 13
    assert runs.count("selection_stage2") == 7
    assert runs.count("selection_recheck") == 84
    assert set(loaded) == {20, 50, 100, 150}
    assert result["run_count"] == 104
