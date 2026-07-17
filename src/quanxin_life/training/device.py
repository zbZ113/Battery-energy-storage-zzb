"""Fail-closed binding of physical GPU 1 to the sole visible PyTorch CUDA device."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
from pydantic import ConfigDict, Field

from quanxin_life.core.schemas import ContractModel

_MIN_A100_MEMORY_MIB = 75 * 1024
_MAX_STARTUP_MEMORY_MIB = 1024


class PhysicalGpuSnapshot(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    total_memory_mib: int = Field(gt=0)
    used_memory_mib: int = Field(ge=0)


class VisibleCudaSnapshot(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_count: int = Field(ge=0)
    current_device: int | None = Field(default=None, ge=0)
    name: str | None = None
    total_memory_bytes: int | None = Field(default=None, gt=0)


def validate_a100_gpu1_binding(
    *,
    physical: PhysicalGpuSnapshot,
    visible: VisibleCudaSnapshot,
    cuda_visible_devices: str | None,
) -> None:
    if physical.index != 1:
        raise ValueError("the approved training device must be physical GPU 1")
    if "A100" not in physical.name:
        raise ValueError("physical GPU 1 must be an NVIDIA A100")
    if physical.total_memory_mib < _MIN_A100_MEMORY_MIB:
        raise ValueError("physical GPU 1 must provide at least 75 GiB")
    if physical.used_memory_mib > _MAX_STARTUP_MEMORY_MIB:
        raise ValueError("physical GPU 1 memory use exceeds the 1024 MiB startup limit")
    if cuda_visible_devices != "1":
        raise ValueError("CUDA_VISIBLE_DEVICES must expose only physical GPU 1")
    if visible.device_count != 1:
        raise ValueError("PyTorch must see exactly one CUDA device")
    if visible.current_device != 0:
        raise ValueError("the sole visible PyTorch device must be cuda:0")
    if visible.name is None or "A100" not in visible.name:
        raise ValueError("the visible PyTorch device must be an NVIDIA A100")
    if (
        visible.total_memory_bytes is None
        or visible.total_memory_bytes < _MIN_A100_MEMORY_MIB * 1024**2
    ):
        raise ValueError("the visible PyTorch device must provide at least 75 GiB")


def collect_physical_gpu_snapshot(
    *,
    physical_index: int,
    nvidia_smi: str = "nvidia-smi",
) -> PhysicalGpuSnapshot:
    command = [
        nvidia_smi,
        f"--id={physical_index}",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi failed while inspecting physical GPU 1") from exc
    lines = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if len(lines) != 1:
        raise ValueError("nvidia-smi must return exactly one physical GPU row")
    parts = tuple(part.strip() for part in lines[0].split(","))
    if len(parts) != 4:
        raise ValueError("nvidia-smi returned an unexpected GPU row")
    try:
        return PhysicalGpuSnapshot(
            index=int(parts[0]),
            name=parts[1],
            total_memory_mib=int(parts[2]),
            used_memory_mib=int(parts[3]),
        )
    except ValueError as exc:
        raise ValueError("nvidia-smi returned invalid numeric GPU fields") from exc


def collect_visible_cuda_snapshot() -> VisibleCudaSnapshot:
    if not torch.cuda.is_available():
        return VisibleCudaSnapshot(device_count=0)
    count = torch.cuda.device_count()
    current = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(current)
    return VisibleCudaSnapshot(
        device_count=count,
        current_device=current,
        name=str(properties.name),
        total_memory_bytes=int(properties.total_memory),
    )


def validate_local_a100_binding(*, physical_index: int = 1) -> dict[str, object]:
    physical = collect_physical_gpu_snapshot(physical_index=physical_index)
    visible = collect_visible_cuda_snapshot()
    visible_environment = os.environ.get("CUDA_VISIBLE_DEVICES")
    validate_a100_gpu1_binding(
        physical=physical,
        visible=visible,
        cuda_visible_devices=visible_environment,
    )
    return {
        "status": "READY",
        "physical_gpu": physical.model_dump(mode="json"),
        "visible_cuda": visible.model_dump(mode="json"),
        "torch_device": "cuda:0",
    }


def require_existing_directory(path: Path) -> Path:
    """Small reusable guard for scripts writing device reports."""

    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError("report parent must be a regular directory")
    return resolved
