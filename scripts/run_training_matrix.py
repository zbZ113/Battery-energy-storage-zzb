"""Plan and execute the governed offline training matrix.

The command is deliberately fail-closed: disabled rows are reported with their
blocker and never contribute to completed counts.  ``plan`` never validates a
GPU or constructs a training task.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.core import TrainingMode  # noqa: E402
from quanxin_life.training.matrix import load_training_matrix  # noqa: E402
from quanxin_life.training.orchestrator import (  # noqa: E402
    build_matrix_plan_summary,
)

BLOCKED_EXIT = 42
EXECUTION_ERROR_EXIT = 44


def execute_ready_tasks(
    summary: dict[str, Any],
    *,
    executor: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Execute READY rows through an injected adapter and preserve blockers."""

    tasks: list[dict[str, Any]] = []
    for row in summary["tasks"]:
        task = dict(row)
        if task["status"] != "READY":
            tasks.append(task)
            continue
        try:
            result = executor(task)
        except Exception as exc:  # pragma: no cover - exercised by integration callers
            task["status"] = "FAILED"
            task["error"] = str(exc)
        else:
            task["status"] = "COMPLETED"
            if isinstance(result, dict):
                task["result"] = result
        tasks.append(task)
    summary = {**summary, "tasks": tasks}
    summary["ready_count"] = sum(item["status"] == "READY" for item in tasks)
    summary["blocked_count"] = sum(item["status"] == "BLOCKED" for item in tasks)
    summary["completed_count"] = sum(item["status"] == "COMPLETED" for item in tasks)
    summary["failed_count"] = sum(item["status"] == "FAILED" for item in tasks)
    if summary["failed_count"]:
        summary["status"] = "FAILED"
    elif summary["blocked_count"]:
        summary["status"] = "COMPLETED_WITH_BLOCKED"
    else:
        summary["status"] = "COMPLETED"
    return summary


def _run_advanced_task(task: dict[str, Any], *, mode: TrainingMode, root: Path) -> Any:
    """Dispatch a ready advanced task to the existing governed orchestrator."""

    family = str(task["model_family"])
    if family not in {
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    }:
        raise RuntimeError(
            f"BLOCKED_DEPENDENCY: no governed executor is registered for {family}"
        )
    if mode is TrainingMode.PLAN:
        raise RuntimeError("plan mode must not execute training")
    import torch

    from quanxin_life.training import advanced_orchestrator

    config = advanced_orchestrator._load_mode_config(root, mode.value)  # type: ignore[arg-type]
    result = advanced_orchestrator.execute_advanced_matr_three_batch_suite(
        project_root=root,
        config=config,
        device=torch.device("cuda:0"),
        seed=int(task["seed"]) if mode is TrainingMode.SMOKE else None,
    )
    return result


def run_matrix(
    mode: TrainingMode,
    *,
    root: Path = REPO_ROOT,
    matrix_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    matrix = load_training_matrix(matrix_path)
    summary = build_matrix_plan_summary(matrix, mode=mode)
    if mode is TrainingMode.PLAN:
        return summary, 0
    if summary["ready_count"] == 0:
        summary = {**summary, "status": "BLOCKED"}
        return summary, BLOCKED_EXIT

    from quanxin_life.training.device import validate_local_a100_binding

    validate_local_a100_binding(physical_index=1)
    advanced_result: dict[str, Any] = {}

    def execute_once(task: dict[str, Any]) -> Any:
        # The advanced orchestrator owns the complete candidate/cutoff matrix;
        # invoke it once and let each registry row reference the same evidence.
        family = str(task["model_family"])
        if family in {
            "cyclepatch_direct",
            "cyclepatch_batlinet",
            "current_hybrid",
            "hybridpatch_v2",
        }:
            if "result" not in advanced_result:
                advanced_result["result"] = _run_advanced_task(
                    task, mode=mode, root=root
                )
            return advanced_result["result"]
        return _run_advanced_task(task, mode=mode, root=root)

    return (
        execute_ready_tasks(
            summary,
            executor=execute_once,
        ),
        0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(item.value for item in TrainingMode))
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    mode = TrainingMode(args.mode)
    if args.plan_only:
        summary = build_matrix_plan_summary(
            load_training_matrix(args.matrix), mode=mode
        )
        exit_code = 0
    else:
        summary, exit_code = run_matrix(
            mode,
            root=args.project_root.resolve(),
            matrix_path=args.matrix,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
