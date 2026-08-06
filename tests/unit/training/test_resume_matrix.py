from __future__ import annotations

import json
from pathlib import Path

import pytest

from quanxin_life.core import TrainingMode, sha256_canonical
from quanxin_life.training.matrix import load_training_matrix
from quanxin_life.training.orchestrator import (
    MatrixResumeAction,
    build_matrix_plan_summary,
    resolve_matrix_resume_action,
)


def test_blocked_tasks_are_not_counted_as_completed() -> None:
    summary = build_matrix_plan_summary(
        load_training_matrix(),
        mode=TrainingMode.PLAN,
    )

    assert summary["task_count"] == 12
    assert summary["ready_count"] == 0
    assert summary["blocked_count"] == 12
    assert summary["completed_count"] == 0
    assert {row["status"] for row in summary["tasks"]} == {"BLOCKED"}


def test_completed_context_is_skipped_and_paused_context_resumes(tmp_path: Path) -> None:
    context = {"run_id": "run-a", "seed": 38, "model_view_sha256": "a" * 64}
    context_sha = sha256_canonical(context)
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "COMPLETED", "context_sha256": context_sha}),
        encoding="utf-8",
    )

    assert (
        resolve_matrix_resume_action(tmp_path, expected_context_sha256=context_sha)
        is MatrixResumeAction.SKIP_COMPLETED
    )

    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "PAUSED_STAGE", "context_sha256": context_sha}),
        encoding="utf-8",
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / "epoch-000001"
    checkpoint.mkdir(parents=True)
    (checkpoints / "last.json").write_text(
        json.dumps({"checkpoint": checkpoint.name}), encoding="utf-8"
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps({"context": context}), encoding="utf-8"
    )

    assert (
        resolve_matrix_resume_action(tmp_path, expected_context_sha256=context_sha)
        is MatrixResumeAction.RESUME
    )


def test_resume_rejects_changed_context_and_corrupt_checkpoint(tmp_path: Path) -> None:
    context = {"run_id": "run-a", "seed": 38, "model_view_sha256": "a" * 64}
    context_sha = sha256_canonical(context)
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "PAUSED_STAGE", "context_sha256": context_sha}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context"):
        resolve_matrix_resume_action(
            tmp_path,
            expected_context_sha256=sha256_canonical({**context, "seed": 39}),
        )

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "last.json").write_text(
        json.dumps({"checkpoint": "epoch-000001"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_matrix_resume_action(tmp_path, expected_context_sha256=context_sha)

