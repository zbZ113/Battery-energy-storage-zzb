#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_NAME="${QUANXIN_CONDA_ENV:-quanxin-a100}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found" >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -Fxq "${ENVIRONMENT_NAME}"; then
  echo "conda environment already exists: ${ENVIRONMENT_NAME}" >&2
  exit 2
fi

conda create -n "${ENVIRONMENT_NAME}" python=3.11.13 pip -y
conda activate "${ENVIRONMENT_NAME}"
python -m pip install --upgrade "pip>=26.1.2,<27" "setuptools>=83,<84" wheel
python -m pip install torch==2.12.0
python -m pip install --require-hashes -r requirements/a100-linux-py311.lock
python -m pip install -e . --no-deps
python -m pip check

python - <<'PY'
import platform
import torch

assert platform.python_version() == "3.11.13", platform.python_version()
assert torch.__version__.split("+")[0] == "2.12.0", torch.__version__
print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})
PY

echo "environment ready: conda activate ${ENVIRONMENT_NAME}"
