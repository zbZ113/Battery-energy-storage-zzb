"""Secret-safe A100 training environment preflight.

The collector deliberately reports a small allow-listed machine snapshot. It
never serializes environment variables, command lines, user names or host
names. A report describes capability only; it never starts training.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core.schemas import ContractModel


class PreflightStatus(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


class GpuDevice(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    total_memory_bytes: int = Field(gt=0)
    compute_capability: str = Field(pattern=r"^\d+\.\d+$")


class PackageSnapshot(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    required: bool
    installed_version: str | None = None


class HardwareSnapshot(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    cpu_count: int | None = Field(default=None, gt=0)
    total_memory_bytes: int | None = Field(default=None, gt=0)
    free_disk_bytes: int | None = Field(default=None, ge=0)
    torch_version: str | None = None
    cuda_build_version: str | None = None
    cuda_available: bool
    gpu_devices: tuple[GpuDevice, ...] = ()


class TrainingPreflightReport(ContractModel):
    """Allow-listed capability report safe to place in a training bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "a100-training-preflight-v1"
    generated_at: datetime
    hardware: HardwareSnapshot
    packages: tuple[PackageSnapshot, ...]
    git_commit: str | None = None
    git_dirty: bool | None = None
    status: PreflightStatus
    ready_for_gpu_training: bool
    reason_codes: tuple[str, ...]

    @field_validator("generated_at")
    @classmethod
    def generated_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value.astimezone(UTC)


def build_training_preflight(
    *,
    generated_at: datetime,
    hardware: HardwareSnapshot,
    packages: tuple[PackageSnapshot, ...],
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> TrainingPreflightReport:
    """Build a deterministic capability decision from collected facts."""

    reason_codes: list[str] = []
    if not hardware.python_version.startswith("3.11."):
        reason_codes.append("PYTHON_VERSION_UNSUPPORTED")
    if not hardware.cuda_available:
        reason_codes.append("CUDA_UNAVAILABLE")
    if not hardware.gpu_devices:
        reason_codes.append("GPU_DEVICE_UNAVAILABLE")
    torch_package = next((item for item in packages if item.name == "torch"), None)
    if (
        torch_package is not None
        and torch_package.installed_version is not None
        and hardware.torch_version is None
    ):
        reason_codes.append("TORCH_RUNTIME_UNAVAILABLE")
    if any(item.required and item.installed_version is None for item in packages):
        reason_codes.append("REQUIRED_PACKAGE_MISSING")
    if git_dirty is True:
        reason_codes.append("GIT_WORKTREE_DIRTY")

    blocking = {
        "PYTHON_VERSION_UNSUPPORTED",
        "REQUIRED_PACKAGE_MISSING",
        "TORCH_RUNTIME_UNAVAILABLE",
    }
    if any(code in blocking for code in reason_codes):
        status = PreflightStatus.BLOCKED
    elif reason_codes:
        status = PreflightStatus.DEGRADED
    else:
        status = PreflightStatus.READY
    return TrainingPreflightReport(
        generated_at=generated_at,
        hardware=hardware,
        packages=packages,
        git_commit=git_commit,
        git_dirty=git_dirty,
        status=status,
        ready_for_gpu_training=status is PreflightStatus.READY,
        reason_codes=tuple(reason_codes),
    )


def collect_local_training_preflight(
    *,
    repo_root: Path,
    required_packages: tuple[str, ...] = (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "xgboost",
        "torch",
        "safetensors",
    ),
    generated_at: datetime | None = None,
) -> TrainingPreflightReport:
    """Collect an allow-listed local snapshot without reading environment values."""

    packages = tuple(
        PackageSnapshot(
            name=name,
            required=True,
            installed_version=_package_version(name),
        )
        for name in required_packages
    )
    torch_info = _torch_snapshot()
    hardware = HardwareSnapshot(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        total_memory_bytes=_total_memory_bytes(),
        free_disk_bytes=shutil.disk_usage(repo_root.resolve()).free,
        torch_version=torch_info[0],
        cuda_build_version=torch_info[1],
        cuda_available=torch_info[2],
        gpu_devices=torch_info[3],
    )
    commit, dirty = _git_snapshot(repo_root)
    return build_training_preflight(
        generated_at=generated_at or datetime.now(UTC),
        hardware=hardware,
        packages=packages,
        git_commit=commit,
        git_dirty=dirty,
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_snapshot() -> tuple[str | None, str | None, bool, tuple[GpuDevice, ...]]:
    try:
        import torch
    except (ImportError, OSError):
        return None, None, False, ()

    cuda_available = bool(torch.cuda.is_available())
    devices: list[GpuDevice] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties: Any = torch.cuda.get_device_properties(index)
            devices.append(
                GpuDevice(
                    index=index,
                    name=str(properties.name),
                    total_memory_bytes=int(properties.total_memory),
                    compute_capability=f"{properties.major}.{properties.minor}",
                )
            )
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return str(torch.__version__), cuda_version, cuda_available, tuple(devices)


def _total_memory_bytes() -> int | None:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    page_count = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    if isinstance(page_size, int) and isinstance(page_count, int):
        return page_size * page_count
    return None


def _git_snapshot(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())
