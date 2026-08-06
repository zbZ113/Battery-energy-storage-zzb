from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quanxin_life.core import TrainingBlockedReason, TrainingMode, TrainingTaskType
from quanxin_life.training.config import TrainingTaskMatrix
from quanxin_life.training.matrix import (
    TrainingMatrixEntry,
    load_training_matrix,
    plan_training_matrix,
)
from quanxin_life.training.suite import build_registered_run_matrix

SHA = "a" * 64
VALID_ENTRY = {
    "task_id": "p0a-cpmlp-matr-c050-s38-select",
    "mode": "select",
    "task_type": "cycle_life",
    "target_semantics": "matr_official_cycle_life",
    "model_family": "cpmlp",
    "model_version": "cpmlp-v1",
    "candidate_id": "cpmlp-reference",
    "dataset_id": "MATR",
    "dataset_version": "matr-three-batch-v1",
    "model_view_version": "early-life-view-v1",
    "split_version": "matr-three-batch-cell-split-v1",
    "cutoff_cycle": 50,
    "prediction_horizon": None,
    "seed": 38,
    "fold": 0,
    "optimizer": "adamw",
    "loss_names": ["huber"],
    "micro_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 64,
    "precision": "fp32",
    "max_epochs": 300,
    "validation_interval": 5,
    "early_stopping_patience": 10,
    "selection_metric_name": "validation_mae",
    "selection_metric_direction": "minimize",
    "required_artifacts": ["best.safetensors", "last.safetensors"],
    "readable_splits": ["train", "validation"],
    "enabled": True,
    "blocked_reason": None,
    "data_sha256": SHA,
    "model_view_sha256": SHA,
    "split_sha256": SHA,
    "config_sha256": SHA,
    "source_commit": "0123456789abcdef0123456789abcdef01234567",
}


def test_task_identity_binds_all_reproducibility_fields() -> None:
    task = TrainingMatrixEntry.model_validate(VALID_ENTRY)

    assert task.mode is TrainingMode.SELECT
    assert task.task_type is TrainingTaskType.CYCLE_LIFE
    assert task.task_id
    assert task.data_sha256
    assert task.model_view_sha256
    assert task.split_sha256
    assert task.config_sha256


def test_naumann_cannot_be_assigned_to_cycle_life_supervision() -> None:
    payload = {**VALID_ENTRY, "dataset_id": "NAUMANN_CYCLE"}

    with pytest.raises(ValidationError, match="not compatible"):
        TrainingMatrixEntry.model_validate(payload)


def test_select_task_cannot_reference_test_split() -> None:
    payload = {
        **VALID_ENTRY,
        "readable_splits": ["train", "validation", "test"],
    }

    with pytest.raises(ValidationError, match="test"):
        TrainingMatrixEntry.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_interval", 1, "validation_interval"),
        ("early_stopping_patience", 5, "early_stopping_patience"),
        ("effective_batch_size", 63, "effective_batch_size"),
    ],
)
def test_training_schedule_is_frozen(field: str, value: int, message: str) -> None:
    payload = {**VALID_ENTRY, field: value}

    with pytest.raises(ValidationError, match=message):
        TrainingMatrixEntry.model_validate(payload)


@pytest.mark.parametrize(
    ("enabled", "blocked_reason"),
    [
        (True, "BLOCKED_DATA_VIEW"),
        (False, None),
        (False, "UNREVIEWED"),
    ],
)
def test_enabled_and_blocked_reason_are_consistent(
    enabled: bool, blocked_reason: str | None
) -> None:
    payload = {**VALID_ENTRY, "enabled": enabled, "blocked_reason": blocked_reason}

    with pytest.raises(ValidationError, match="blocked_reason"):
        TrainingMatrixEntry.model_validate(payload)


def test_versioned_matrix_registers_all_planned_model_families() -> None:
    matrix = load_training_matrix(Path("configs/training/task_matrix_v1.json"))

    assert {entry.model_family for entry in matrix.entries} == {
        "cpmlp",
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
        "pbt",
        "diting_cptransformer",
        "batterymformer",
        "magnet",
        "battgp",
        "blast_lite",
        "smart_feature",
    }
    assert all(entry.enabled or entry.blocked_reason is not None for entry in matrix.entries)
    assert {entry.task_id for entry in plan_training_matrix(matrix)} == {
        entry.task_id for entry in matrix.entries
    }


def test_blocked_entries_do_not_fabricate_unavailable_hashes() -> None:
    matrix = load_training_matrix()

    assert isinstance(matrix, TrainingTaskMatrix)
    assert all(entry.enabled is False for entry in matrix.entries)
    assert all(
        entry.blocked_reason is TrainingBlockedReason.BLOCKED_DATA_VIEW
        for entry in matrix.entries
    )
    assert all(
        entry.data_sha256 is None and entry.model_view_sha256 is None
        for entry in matrix.entries
    )
    assert all(
        entry.split_sha256 != "0" * 64 and entry.config_sha256 != "0" * 64
        for entry in matrix.entries
    )


def test_legacy_suite_exposes_the_registered_matrix_without_opening_data() -> None:
    assert build_registered_run_matrix(mode=TrainingMode.PLAN) == plan_training_matrix(
        load_training_matrix(), mode=TrainingMode.PLAN
    )
