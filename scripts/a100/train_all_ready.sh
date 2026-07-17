#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "final" ) ]]; then
  echo "usage: bash scripts/a100/train_all_ready.sh <smoke|final>" >&2
  exit 2
fi

bash scripts/a100/train_dataset.sh matr-three-batch "$1"
