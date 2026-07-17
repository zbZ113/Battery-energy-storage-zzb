import pytest

from quanxin_life.training.device import (
    PhysicalGpuSnapshot,
    VisibleCudaSnapshot,
    validate_a100_gpu1_binding,
)


def _physical(*, used_memory_mib: int = 17) -> PhysicalGpuSnapshot:
    return PhysicalGpuSnapshot(
        index=1,
        name="NVIDIA A100 80GB PCIe",
        total_memory_mib=81920,
        used_memory_mib=used_memory_mib,
    )


def _visible() -> VisibleCudaSnapshot:
    return VisibleCudaSnapshot(
        device_count=1,
        current_device=0,
        name="NVIDIA A100 80GB PCIe",
        total_memory_bytes=80 * 1024**3,
    )


def test_accepts_idle_physical_gpu1_mapped_to_torch_cuda0() -> None:
    validate_a100_gpu1_binding(
        physical=_physical(),
        visible=_visible(),
        cuda_visible_devices="1",
    )


@pytest.mark.parametrize(
    ("physical", "visible", "environment", "pattern"),
    [
        (_physical(used_memory_mib=1025), _visible(), "1", "memory use"),
        (_physical(), _visible(), "0", "CUDA_VISIBLE_DEVICES"),
        (
            _physical(),
            VisibleCudaSnapshot(
                device_count=2,
                current_device=0,
                name="NVIDIA A100 80GB PCIe",
                total_memory_bytes=80 * 1024**3,
            ),
            "1",
            "exactly one",
        ),
    ],
)
def test_rejects_unsafe_a100_binding(
    physical: PhysicalGpuSnapshot,
    visible: VisibleCudaSnapshot,
    environment: str,
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        validate_a100_gpu1_binding(
            physical=physical,
            visible=visible,
            cuda_visible_devices=environment,
        )
