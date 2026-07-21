#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: bash scripts/a100/train_advanced_models.sh matr-three-batch <smoke|select|final> [--plan-only]" >&2
  exit 2
fi

DATASET="$1"
MODE="$2"
PLAN_ONLY="${3:-}"
if [[ "${DATASET}" != "matr-three-batch" ]]; then
  echo "advanced models currently support only matr-three-batch" >&2
  exit 42
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "select" && "${MODE}" != "final" ]]; then
  echo "mode must be smoke, select or final" >&2
  exit 2
fi
if [[ -n "${PLAN_ONLY}" && "${PLAN_ONLY}" != "--plan-only" ]]; then
  echo "third argument must be --plan-only" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export MLFLOW_ALLOW_FILE_STORE=true
export PYTHONUNBUFFERED=1
unset DISPLAY

if [[ "${PLAN_ONLY}" == "--plan-only" ]]; then
  exec python scripts/run_advanced_model_suite.py "${DATASET}" "${MODE}" --plan-only
fi

bash scripts/a100/preflight.sh
mkdir -p logs/a100
python scripts/prepare_advanced_matr_data.py "${MODE}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${A100_ADVANCED_LOG:-logs/a100/advanced-${MODE}-${timestamp}.log}"
touch "${log_file}"
echo "advanced run dataset=${DATASET} mode=${MODE} log=${log_file}" | tee -a "${log_file}"

if [[ "${MODE}" == "select" ]]; then
  SEEDS=(38 39 40)
  export PYTHONHASHSEED=38
  python scripts/run_advanced_model_suite.py "${DATASET}" "${MODE}" 2>&1 | tee -a "${log_file}"
  python scripts/build_advanced_run_index.py "${DATASET}" "${MODE}" 2>&1 | tee -a "${log_file}"
  echo "advanced run completed: ${log_file}" | tee -a "${log_file}"
  exit 0
fi

case "${MODE}" in
  smoke) SEEDS=(38) ;;
  final) SEEDS=(38 39 40 41 42) ;;
esac

for seed in "${SEEDS[@]}"; do
  export PYTHONHASHSEED="${seed}"
  echo "--- seed=${seed} ---" | tee -a "${log_file}"
  python scripts/run_advanced_model_suite.py "${DATASET}" "${MODE}" --seed "${seed}" 2>&1 | tee -a "${log_file}"
done

python scripts/build_advanced_run_index.py "${DATASET}" "${MODE}" 2>&1 | tee -a "${log_file}"
echo "advanced run completed: ${log_file}" | tee -a "${log_file}"
