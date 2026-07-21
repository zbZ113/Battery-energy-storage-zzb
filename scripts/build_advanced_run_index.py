"""Build the byte-level index for a completed advanced A100 stage."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.advanced_config import AdvancedMatrThreeBatchRunConfig  # noqa: E402
from quanxin_life.training.advanced_orchestrator import _source_commit  # noqa: E402
from quanxin_life.training.advanced_outputs import (  # noqa: E402
    write_advanced_training_output_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("matr-three-batch",))
    parser.add_argument("mode", choices=("smoke", "select", "final"))
    args = parser.parse_args()

    resolved_final = (
        REPO_ROOT
        / "runs/a100/matr-three-batch/advanced/selection/final_config_resolved.json"
    )
    config_path = (
        resolved_final
        if args.mode == "final" and resolved_final.is_file()
        else REPO_ROOT
        / "configs/training/advanced"
        / ("selection.json" if args.mode == "select" else f"{args.mode}.json")
    )
    config = AdvancedMatrThreeBatchRunConfig.model_validate_json(config_path.read_bytes())
    output_root = REPO_ROOT / config.paths.run_root
    index = write_advanced_training_output_index(
        output_root,
        mode=args.mode,
        source_commit=_source_commit(REPO_ROOT),
        config_sha256=config.config_sha256,
        created_at=datetime.now(UTC),
    )
    print(json.dumps(index.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
