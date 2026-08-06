#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "plan" && "$1" != "smoke" && "$1" != "select" && "$1" != "final" ) ]]; then
  echo "usage: bash scripts/a100/launch_all_training.sh <plan|smoke|select|final>" >&2
  exit 2
fi

MODE="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Physical GPU 1 is intentionally exposed as process-local cuda:0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export MLFLOW_ALLOW_FILE_STORE=true
export PYTHONUNBUFFERED=1
unset DISPLAY

if [[ "${MODE}" == "plan" ]]; then
  exec python scripts/run_training_matrix.py plan
fi

bash scripts/a100/preflight.sh
exec python scripts/run_training_matrix.py "${MODE}"

