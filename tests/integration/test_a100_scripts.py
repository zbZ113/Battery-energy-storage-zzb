from pathlib import Path


def test_a100_preflight_binds_only_physical_gpu1() -> None:
    script = Path("scripts/a100/preflight.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "a100_device_guard.py --physical-index 1" in script


def test_dataset_shell_is_one_command_and_never_parallelizes_models() -> None:
    script = Path("scripts/a100/train_dataset.sh").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert 'python scripts/run_training_suite.py matr "${MODE}"' in script
    assert "xargs -P" not in script
    assert " wait" not in script


def test_a100_environment_is_hash_locked_and_does_not_install_cuda_toolkit() -> None:
    script = Path("scripts/a100/create_env.sh").read_text(encoding="utf-8")
    lock = Path("requirements/a100-linux-py311.lock").read_text(encoding="utf-8")

    assert "python=3.11.13" in script
    assert "torch==2.12.0" in script
    assert "--require-hashes -r requirements/a100-linux-py311.lock" in script
    assert "pip check" in script
    assert "cuda-toolkit" not in script.lower()
    assert "--hash=sha256:" in lock
    assert "xgboost==2.1.4" in lock
    assert "\ntorch==" not in lock
