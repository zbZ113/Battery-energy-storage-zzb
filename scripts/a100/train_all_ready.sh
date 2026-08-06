#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "plan" && "$1" != "smoke" && "$1" != "select" && "$1" != "final" ) ]]; then
  echo "usage: bash scripts/a100/train_all_ready.sh <plan|smoke|select|final>" >&2
  exit 2
fi

bash scripts/a100/launch_all_training.sh "$1"
