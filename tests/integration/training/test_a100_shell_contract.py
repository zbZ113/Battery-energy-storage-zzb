from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_unified_shell_pins_physical_gpu_one_and_keeps_plan_cpu_only() -> None:
    script = (ROOT / "scripts" / "a100" / "launch_all_training.sh").read_text(
        encoding="utf-8"
    )

    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert 'if [[ "${MODE}" == "plan" ]]' in script
    assert "run_training_matrix.py" in script
    assert "preflight.sh" in script


def test_legacy_all_ready_entry_delegates_to_unified_launcher() -> None:
    script = (ROOT / "scripts" / "a100" / "train_all_ready.sh").read_text(
        encoding="utf-8"
    )
    assert "launch_all_training.sh" in script

