"""Run or inspect the governed advanced MATR model suite."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.advanced_config import (  # noqa: E402
    AdvancedMatrThreeBatchRunConfig,
    build_advanced_run_matrix,
)
from quanxin_life.training.device import validate_local_a100_binding  # noqa: E402

DATASET_NOT_READY_EXIT = 42
FINAL_NOT_BOUND_EXIT = 43


def _load_config(mode: str) -> AdvancedMatrThreeBatchRunConfig:
    resolved_final = (
        REPO_ROOT
        / "runs"
        / "a100"
        / "matr-three-batch"
        / "advanced"
        / "selection"
        / "final_config_resolved.json"
    )
    config_name = "selection.json" if mode == "select" else f"{mode}.json"
    path = (
        resolved_final
        if mode == "final" and resolved_final.is_file() and not resolved_final.is_symlink()
        else REPO_ROOT / "configs" / "training" / "advanced" / config_name
    )
    return AdvancedMatrThreeBatchRunConfig.model_validate_json(path.read_bytes())


def _plan(config: AdvancedMatrThreeBatchRunConfig) -> dict[str, Any]:
    if config.mode == "select":
        matrix = build_advanced_run_matrix(config, selection_stage="selection_stage1")
    else:
        matrix = build_advanced_run_matrix(
            config,
            repository_root=REPO_ROOT if config.mode == "final" else None,
        )
    return {
        "status": "PLAN_READY",
        "dataset": "matr-three-batch",
        "mode": config.mode,
        "config_sha256": config.config_sha256,
        "task_count": len(matrix),
        "tasks": [item.model_dump(mode="json") for item in matrix],
    }


def _resolve_executor() -> Callable[..., Any]:
    try:
        from quanxin_life.training import advanced_orchestrator
    except ImportError as exc:
        raise RuntimeError(
            "advanced_orchestrator is unavailable; install the completed advanced "
            "training implementation before starting A100 work"
        ) from exc
    names = (
        "execute_advanced_matr_three_batch_suite",
        "run_advanced_matr_three_batch_suite",
        "execute_advanced_suite",
        "run_advanced_suite",
    )
    for name in names:
        candidate = getattr(advanced_orchestrator, name, None)
        if callable(candidate):
            return cast(Callable[..., Any], candidate)
    raise RuntimeError("advanced_orchestrator exposes none of: " + ", ".join(names))


def _invoke_executor(
    executor: Callable[..., Any],
    *,
    config: AdvancedMatrThreeBatchRunConfig,
    seed: int | None,
) -> Any:
    values: dict[str, Any] = {
        "project_root": REPO_ROOT,
        "repository_root": REPO_ROOT,
        "config": config,
        "device": torch.device("cuda:0"),
        "seed": seed,
        "mode": config.mode,
    }
    parameters = inspect.signature(executor).parameters
    kwargs = {name: value for name, value in values.items() if name in parameters}
    return executor(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=("matr-three-batch", "naumann-cycle", "naumann-calendar"),
    )
    parser.add_argument("mode", choices=("smoke", "select", "final"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=(38, 39, 40, 41, 42))
    args = parser.parse_args()

    if args.dataset != "matr-three-batch":
        print(
            json.dumps(
                {
                    "status": "DATASET_NOT_READY",
                    "dataset": args.dataset,
                    "reason": "approved advanced data and target semantics are not ready",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return DATASET_NOT_READY_EXIT

    config = _load_config(args.mode)
    if args.seed is not None and args.seed not in config.seeds:
        raise SystemExit(f"seed {args.seed} is not part of the {args.mode} configuration")
    if args.plan_only:
        try:
            print(json.dumps(_plan(config), ensure_ascii=False, sort_keys=True))
        except ValueError as exc:
            if config.mode == "final" and "selection" in str(exc).lower():
                print(
                    json.dumps(
                        {
                            "status": "FINAL_CONFIG_NOT_BOUND",
                            "mode": "final",
                            "reason": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return FINAL_NOT_BOUND_EXIT
            raise
        return 0

    validate_local_a100_binding(physical_index=1)
    result = _invoke_executor(_resolve_executor(), config=config, seed=args.seed)
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
