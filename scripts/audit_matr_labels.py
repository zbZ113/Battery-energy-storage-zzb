"""Audit official MATR life labels against the project's unified EOL80 rule."""

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
    audit_matr_eol80_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    raw_manifest = RawFileManifest.model_validate_json(arguments.manifest.read_bytes())
    conversion = MatrBatchConversionReport.model_validate_json(
        arguments.conversion_report.read_bytes()
    )
    audit = audit_matr_eol80_labels(
        raw_path=arguments.raw,
        raw_manifest=raw_manifest,
        conversion_report=conversion,
        created_at=datetime.now(UTC),
    )
    payload = json.dumps(
        audit.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(arguments.output)
    print(
        f"audited cells={audit.cell_count} unified_events={audit.unified_event_count} "
        f"right_censored={audit.unified_right_censored_count} output={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
