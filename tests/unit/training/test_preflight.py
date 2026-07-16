from __future__ import annotations

from datetime import UTC, datetime

from quanxin_life.training.preflight import (
    GpuDevice,
    HardwareSnapshot,
    PackageSnapshot,
    build_training_preflight,
)


def test_gpu_preflight_is_ready_only_when_cuda_gpu_and_packages_are_available() -> None:
    report = build_training_preflight(
        generated_at=datetime(2026, 7, 16, tzinfo=UTC),
        hardware=HardwareSnapshot(
            platform="Linux-6.8-x86_64",
            python_version="3.11.13",
            cpu_count=32,
            total_memory_bytes=128 * 1024**3,
            free_disk_bytes=500 * 1024**3,
            torch_version="2.7.1",
            cuda_build_version="12.8",
            cuda_available=True,
            gpu_devices=(
                GpuDevice(
                    index=0,
                    name="NVIDIA A100-SXM4-80GB",
                    total_memory_bytes=80 * 1024**3,
                    compute_capability="8.0",
                ),
            ),
        ),
        packages=(
            PackageSnapshot(name="torch", required=True, installed_version="2.7.1"),
            PackageSnapshot(name="pyarrow", required=True, installed_version="23.0.1"),
        ),
    )

    assert report.ready_for_gpu_training is True
    assert report.reason_codes == ()
    assert "environment" not in report.model_dump(mode="json")


def test_gpu_preflight_explicitly_degrades_without_cuda_or_required_package() -> None:
    report = build_training_preflight(
        generated_at=datetime(2026, 7, 16, tzinfo=UTC),
        hardware=HardwareSnapshot(
            platform="Windows-11",
            python_version="3.11.13",
            cpu_count=8,
            total_memory_bytes=16 * 1024**3,
            free_disk_bytes=40 * 1024**3,
            torch_version="2.7.1+cpu",
            cuda_build_version=None,
            cuda_available=False,
            gpu_devices=(),
        ),
        packages=(
            PackageSnapshot(name="torch", required=True, installed_version="2.7.1+cpu"),
            PackageSnapshot(name="safetensors", required=True, installed_version=None),
        ),
    )

    assert report.ready_for_gpu_training is False
    assert report.reason_codes == (
        "CUDA_UNAVAILABLE",
        "GPU_DEVICE_UNAVAILABLE",
        "REQUIRED_PACKAGE_MISSING",
    )


def test_gpu_preflight_blocks_when_torch_metadata_exists_but_runtime_cannot_import() -> None:
    report = build_training_preflight(
        generated_at=datetime(2026, 7, 16, tzinfo=UTC),
        hardware=HardwareSnapshot(
            platform="Linux-6.8-x86_64",
            python_version="3.11.13",
            cpu_count=32,
            total_memory_bytes=128 * 1024**3,
            free_disk_bytes=500 * 1024**3,
            torch_version=None,
            cuda_build_version=None,
            cuda_available=False,
            gpu_devices=(),
        ),
        packages=(
            PackageSnapshot(name="torch", required=True, installed_version="2.7.1"),
        ),
    )

    assert report.status == "BLOCKED"
    assert "TORCH_RUNTIME_UNAVAILABLE" in report.reason_codes
