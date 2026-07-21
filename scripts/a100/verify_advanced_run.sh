#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/a100/verify_advanced_run.sh <output-root> [output-index.json]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="$(cd -- "$1" && pwd)"
INDEX="${2:-${OUTPUT_ROOT}/output_index.json}"

if [[ ! -f "${INDEX}" ]]; then
  echo "missing output index: ${INDEX}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
python - "${OUTPUT_ROOT}" "${INDEX}" <<'PY'
import sys
from pathlib import Path

from quanxin_life.training.advanced_outputs import (
    load_advanced_training_output_index,
    verify_advanced_training_output_index,
)

root = Path(sys.argv[1])
index = load_advanced_training_output_index(Path(sys.argv[2]))
verify_advanced_training_output_index(root, index)
print({"status": "VERIFIED", "mode": index.mode, "operations": index.operation_count, "files": len(index.files), "sha256": index.output_sha256})
PY
