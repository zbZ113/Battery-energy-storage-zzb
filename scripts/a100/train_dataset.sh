#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/a100/train_dataset.sh <matr|hust|naumann-cycle|naumann-calendar> <smoke|final>" >&2
  exit 2
fi

DATASET="$1"
MODE="$2"
if [[ "${MODE}" != "smoke" && "${MODE}" != "final" ]]; then
  echo "mode must be smoke or final" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${DATASET}" != "matr" ]]; then
  python scripts/run_training_suite.py "${DATASET}" "${MODE}" --plan-only
  exit $?
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

bash scripts/a100/preflight.sh
python scripts/run_training_suite.py matr "${MODE}"
