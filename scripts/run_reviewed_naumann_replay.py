"""Run one versioned, provenance-checked Naumann cycle GP replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quanxin_life.experiments.naumann_reviewed_run import (
    load_reviewed_naumann_replay_config,
    run_reviewed_naumann_cycle_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_reviewed_naumann_replay_config(arguments.config)
    result = run_reviewed_naumann_cycle_replay(
        arguments.repository_root,
        config,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
