from __future__ import annotations

import json
from pathlib import Path

import pytest

from quanxin_life.core import TrainingBlockedReason, TrainingTaskType
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.blast import (
    BLAST_LICENSE_STATUS,
    BLAST_UPSTREAM_COMMIT,
    BLASTAdapter,
    validate_blast_artifacts,
    validate_blast_cpu_requirements,
)


def test_blast_is_condition_degradation_and_dependency_gated() -> None:
    adapter = BLASTAdapter()
    assert BLAST_UPSTREAM_COMMIT == "b093495b47dc40dd96dba865d91f553619501e94"
    assert BLAST_LICENSE_STATUS == "VERIFIED_LICENSE_PRESENT"
    assert adapter.task_type is TrainingTaskType.CONDITION_DEGRADATION
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.readiness(numpy_major=2) is TrainingBlockedReason.BLOCKED_DEPENDENCY
    assert adapter.readiness(numpy_major=1) is None


def test_blast_cpu_contract_is_explicitly_numpy_less_than_two() -> None:
    assert validate_blast_cpu_requirements("1.26.4") is None
    with pytest.raises(ValueError, match="numpy<2"):
        validate_blast_cpu_requirements("2.4.0")


def test_blast_artifacts_are_json_or_parquet_with_sha(tmp_path: Path) -> None:
    import hashlib

    params = tmp_path / "parameters.json"
    params.write_text(json.dumps({"model": "lfp_gr_250AhPrismatic"}), encoding="utf-8")
    residuals = tmp_path / "residuals.parquet"
    residuals.write_bytes(b"PAR1")
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=r"Parquet|parse"):
        validate_blast_artifacts(
            {"parameters": (params, digest(params)), "residuals": (residuals, digest(residuals))}
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"condition_id": ["c1"], "residual": [0.0]}), residuals)
    assert set(
        validate_blast_artifacts(
            {"parameters": (params, digest(params)), "residuals": (residuals, digest(residuals))}
        )
    ) == {"parameters", "residuals"}


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_blast_configs_are_cpu_only_and_blocked_in_main_env(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "blast" / f"{stage}.json").read_text(encoding="utf-8")
    )
    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW", "BLOCKED_DEPENDENCY"]
    assert payload["task_type"] == TrainingTaskType.CONDITION_DEGRADATION.value
    assert payload["upstream_commit"] == BLAST_UPSTREAM_COMMIT
    assert payload["runtime_environment"] == "quanxin-blast-py311"
    assert payload["optimization"]["effective_batch_size"] == 64


def test_blast_training_matrix_binds_scenario_identity() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "blast_lite")
    assert entry["task_type"] == TrainingTaskType.CONDITION_DEGRADATION.value
    assert entry["optimizer"] == "none"
    assert entry["selection_metric_name"] == "scenario_residual_mae"
    assert entry["source_commit"] == BLAST_UPSTREAM_COMMIT


def test_blast_requirements_keep_python_out_of_pip_and_record_blocked_lock() -> None:
    root = Path(__file__).parents[4]
    input_text = (root / "requirements" / "blast-cpu-py311.in").read_text(encoding="utf-8")
    lock_text = (root / "requirements" / "blast-cpu-py311.lock").read_text(encoding="utf-8")
    assert "numpy<2.0.0" in input_text
    assert "python==" not in input_text and "python==" not in lock_text
    assert "STATUS: BLOCKED_DEPENDENCY" in lock_text
    for dependency in (
        "blast-lite",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "h5pyd",
        "nrel-rex",
        "geopy",
    ):
        assert dependency in input_text and dependency in lock_text
