#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export MLFLOW_ALLOW_FILE_STORE=true
export PIP_NO_INDEX=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
unset DISPLAY

[[ -f scripts/run_pbt_magnet_training.py ]] || {
  echo "missing scripts/run_pbt_magnet_training.py" >&2
  exit 42
}
[[ -d data/model_views ]] || { echo "missing data/model_views" >&2; exit 42; }
[[ -f scripts/a100/pbt_magnet_recovery_seed42.sha256 ]] || {
  echo "missing seed42 recovery patch checksum manifest" >&2
  exit 42
}
sha256sum -c scripts/a100/pbt_magnet_recovery_seed42.sha256

if [[ -z "${QUANXIN_SOURCE_COMMIT:-}" ]]; then
  QUANXIN_SOURCE_COMMIT="$(git rev-parse HEAD)"
  export QUANXIN_SOURCE_COMMIT
fi
if [[ ! "${QUANXIN_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "QUANXIN_SOURCE_COMMIT must be a 40-character lowercase Git commit" >&2
  exit 42
fi

python scripts/a100_device_guard.py --physical-index 1
A100_PREFLIGHT_OUTPUT=a100-pbt-magnet-recovery-seed42-preflight.json \
  python scripts/a100_preflight.py \
  --output a100-pbt-magnet-recovery-seed42-preflight.json

mkdir -p logs/a100
LOG_FILE="${A100_PBT_MAGNET_RECOVERY_LOG:-logs/a100/pbt-magnet-recovery-seed42.log}"
touch "${LOG_FILE}"

run_stage() {
  local stage="$1"
  local device="$2"
  python scripts/run_pbt_magnet_training.py \
    --models pbt,magnet \
    --stage "${stage}" \
    --seeds 42 \
    --device "${device}" \
    --recovery-seed42 \
    2>&1 | tee -a "${LOG_FILE}"
}

echo "seed42 recovery: isolated selection" | tee -a "${LOG_FILE}"
run_stage selection cuda:0
run_stage freeze-selection cpu
echo "seed42 recovery: isolated final" | tee -a "${LOG_FILE}"
run_stage final cuda:0
run_stage collect cpu

echo "RECOVERY_SUCCESS runs/a100/pbt_magnet_recovery_seed42_v2/summary" \
  | tee -a "${LOG_FILE}"
