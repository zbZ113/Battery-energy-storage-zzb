"""Run or inspect one governed dataset model matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.device import validate_local_a100_binding  # noqa: E402
from quanxin_life.training.orchestrator import (  # noqa: E402
    execute_matr_suite,
    execute_matr_three_batch_suite,
)
from quanxin_life.training.suite import (  # noqa: E402
    MatrRunConfig,
    MatrThreeBatchRunConfig,
    build_run_matrix,
)

DATASET_NOT_READY_EXIT = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=(
            "matr",
            "matr-three-batch",
            "hust",
            "naumann-cycle",
            "naumann-calendar",
        ),
    )
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.dataset not in {"matr", "matr-three-batch"}:
        print(
            json.dumps(
                {
                    "status": "DATASET_NOT_READY",
                    "dataset": arguments.dataset,
                    "reason": "approved Canonical Parquet and target semantics are not ready",
                },
                sort_keys=True,
            )
        )
        return DATASET_NOT_READY_EXIT

    if arguments.dataset == "matr-three-batch":
        config_path = (
            REPO_ROOT
            / "configs"
            / "training"
            / f"matr_three_batch_{arguments.mode}.json"
        )
        config: MatrRunConfig | MatrThreeBatchRunConfig = (
            MatrThreeBatchRunConfig.model_validate_json(config_path.read_bytes())
        )
    else:
        config_path = REPO_ROOT / "configs" / "training" / f"matr_{arguments.mode}.json"
        config = MatrRunConfig.model_validate_json(config_path.read_bytes())
    matrix = build_run_matrix(config.suite)
    if arguments.plan_only:
        print(
            json.dumps(
                {
                    "status": "PLAN_READY",
                    "dataset": arguments.dataset,
                    "mode": arguments.mode,
                    "task_count": len(matrix),
                    "tasks": [item.model_dump(mode="json") for item in matrix],
                },
                sort_keys=True,
            )
        )
        return 0

    validate_local_a100_binding(physical_index=1)
    if isinstance(config, MatrThreeBatchRunConfig):
        aggregate = execute_matr_three_batch_suite(
            project_root=REPO_ROOT,
            config=config,
            device=torch.device("cuda:0"),
        )
    else:
        aggregate = execute_matr_suite(
            project_root=REPO_ROOT,
            config=config,
            device=torch.device("cuda:0"),
        )
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
