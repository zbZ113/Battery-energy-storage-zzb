"""Convert one reviewed MATR HDF5 batch to verified per-cell Parquet artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.data.manifest import RawFileManifest  # noqa: E402
from quanxin_life.data.matr_pipeline import convert_matr_batch  # noqa: E402


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
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-date", type=date.fromisoformat, required=True)
    parser.add_argument("--time-unit", choices=("seconds", "minutes"), required=True)
    parser.add_argument("--max-cycle-index", type=int, required=True)
    arguments = parser.parse_args()

    raw_manifest = RawFileManifest.model_validate_json(arguments.manifest.read_bytes())
    report = convert_matr_batch(
        raw_path=arguments.raw,
        raw_manifest=raw_manifest,
        output_root=arguments.output_root,
        batch_index=arguments.batch_index,
        batch_date=arguments.batch_date,
        time_unit=arguments.time_unit,
        max_cycle_index=arguments.max_cycle_index,
    )
    _write_json_atomic(arguments.report, report.model_dump(mode="json"))
    print(
        f"converted dataset={report.dataset_id} batch={report.batch_index} "
        f"cells={report.cell_count} rows={report.total_row_count} "
        f"report={arguments.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
