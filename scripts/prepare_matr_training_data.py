"""Create or verify the approved MATR inputs required by the A100 suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.preparation import prepare_matr_training_inputs  # noqa: E402
from quanxin_life.training.suite import MatrRunConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "final"))
    arguments = parser.parse_args()
    config_path = REPO_ROOT / "configs" / "training" / f"matr_{arguments.mode}.json"
    config = MatrRunConfig.model_validate_json(config_path.read_bytes())
    result = prepare_matr_training_inputs(project_root=REPO_ROOT, config=config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
