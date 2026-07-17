from pathlib import Path


def test_a100_preflight_binds_only_physical_gpu1() -> None:
    script = Path("scripts/a100/preflight.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "a100_device_guard.py --physical-index 1" in script
