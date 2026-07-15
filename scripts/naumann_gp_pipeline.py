"""Run the deterministic Naumann GP replay pipeline from a strict JSON request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quanxin_life.experiments.naumann_pipeline import (
    load_naumann_pipeline_request,
    run_naumann_gp_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    request = load_naumann_pipeline_request(arguments.request)
    result = run_naumann_gp_pipeline(request, output_dir=arguments.output_dir)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
