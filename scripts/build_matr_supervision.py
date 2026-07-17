"""Build hash-bound MATR official-life and real SOH trajectory supervision."""

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

from quanxin_life.data.manifest import RawFileManifest  # noqa: E402
from quanxin_life.data.matr_pipeline import (  # noqa: E402
    MatrBatchConversionReport,
    build_matr_supervision_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon-cycle", type=int, default=500)
    arguments = parser.parse_args()

    manifest = RawFileManifest.model_validate_json(arguments.manifest.read_bytes())
    conversion = MatrBatchConversionReport.model_validate_json(
        arguments.conversion_report.read_bytes()
    )
    artifact = build_matr_supervision_artifact(
        raw_path=arguments.raw,
        raw_manifest=manifest,
        conversion_report=conversion,
        output_root=arguments.output_root,
        horizon_cycle=arguments.horizon_cycle,
        created_at=datetime.now(UTC),
    )
    payload = json.dumps(
        artifact.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_name(f".{arguments.report.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(arguments.report)
    print(
        f"built dataset={artifact.dataset_id} cells={artifact.cell_count} "
        f"horizon={artifact.horizon_cycle} rows={artifact.row_count} "
        f"sha256={artifact.parquet_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
