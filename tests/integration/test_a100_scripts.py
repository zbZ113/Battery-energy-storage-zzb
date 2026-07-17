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
