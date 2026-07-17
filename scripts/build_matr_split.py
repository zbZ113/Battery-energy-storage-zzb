"""Build a deterministic protocol/life-stratified MATR cell split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.data.matr_pipeline import (  # noqa: E402
    MatrBatchConversionReport,
    build_matr_split_evidence,
)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--quantile-count", type=int, default=4)
    arguments = parser.parse_args()

    conversion = MatrBatchConversionReport.model_validate_json(
        arguments.conversion_report.read_bytes()
    )
    evidence = build_matr_split_evidence(
        conversion,
        split_version=arguments.split_version,
        quantile_count=arguments.quantile_count,
        created_at=datetime.now(UTC),
    )
    _write_json_atomic(
        arguments.split_output,
        evidence.split_manifest.model_dump(mode="json"),
    )
    _write_json_atomic(arguments.evidence_output, evidence.model_dump(mode="json"))
    split = evidence.split_manifest
    print(
        f"split dataset={split.dataset_id} train={len(split.train)} "
        f"validation={len(split.validation)} calibration={len(split.calibration)} "
        f"test={len(split.test)} version={evidence.split_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
