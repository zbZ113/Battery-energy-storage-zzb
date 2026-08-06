from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quanxin_life.training import orchestrator
from quanxin_life.training.adapters.native import NativeMatrViewRoots
from quanxin_life.training.suite import MatrRunConfig, MatrThreeBatchRunConfig

ROOT = Path(__file__).resolve().parents[3]


def test_single_batch_entry_is_explicitly_blocked_without_a_matching_view(
    tmp_path: Path,
) -> None:
    config = MatrRunConfig.model_validate_json(
        (ROOT / "configs/training/matr_smoke.json").read_bytes()
    )

    with pytest.raises(ValueError, match="BLOCKED_DATA_VIEW"):
        orchestrator.execute_matr_suite(
            project_root=tmp_path,
            config=config,
            device=torch.device("cpu"),
        )


def test_three_batch_entry_delegates_to_frozen_views_without_raw_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_relative = Path("reports/data_quality/matr_three_batch_manifest_v1.json")
    split_relative = Path("configs/data_splits/matr_three_batch_cell_split_v1.json")
    for relative in (manifest_relative, split_relative):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    config = MatrThreeBatchRunConfig.model_validate_json(
        (ROOT / "configs/training/matr_three_batch_smoke.json").read_bytes()
    ).model_copy(
        update={
            "paths": MatrThreeBatchRunConfig.model_validate_json(
                (ROOT / "configs/training/matr_three_batch_smoke.json").read_bytes()
            ).paths.model_copy(update={"run_root": "runs/test-native-entry"})
        }
    )
    resolved: list[int] = []

    def resolve(_root: Path, *, cutoff_cycle: int) -> NativeMatrViewRoots:
        resolved.append(cutoff_cycle)
        return NativeMatrViewRoots(
            scalar_root=tmp_path / "scalar",
            trajectory_root=tmp_path / "trajectory",
            model_view_sha256="c" * 64,
        )

    monkeypatch.setattr(orchestrator, "resolve_native_matr_view_roots", resolve)
    monkeypatch.setattr(
        orchestrator,
        "load_native_matr_legacy_cohorts",
        lambda **_kwargs: ("curves", "hybrid"),
    )
    monkeypatch.setattr(orchestrator, "_source_commit", lambda _root: "d" * 40)

    def execute_matrix(**kwargs: object) -> dict[str, object]:
        assert kwargs["load_cohorts"](50) == ("curves", "hybrid")  # type: ignore[operator]
        return {"status": "delegated"}

    monkeypatch.setattr(orchestrator, "_execute_matr_matrix", execute_matrix)

    result = orchestrator.execute_matr_three_batch_suite(
        project_root=tmp_path,
        config=config,
        device=torch.device("cpu"),
    )

    assert result == {"status": "delegated"}
    assert resolved == [20, 50, 100, 150]
