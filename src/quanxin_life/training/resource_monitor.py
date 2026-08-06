"""Best-effort resource records; unavailable telemetry remains null."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import ConfigDict, Field

from quanxin_life.core.schemas import ContractModel


class ResourceRecord(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    epoch: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    gpu_peak_memory_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cpu_percent: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    warning: str | None = None


def capture_resource(epoch: int, started: float | None = None) -> ResourceRecord:
    now = time.monotonic()
    elapsed = 0.0 if started is None else max(0.0, now - started)
    gpu = None
    warning = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            warning = "GPU_TELEMETRY_UNAVAILABLE"
    except (ImportError, RuntimeError):
        warning = "GPU_TELEMETRY_UNAVAILABLE"
    return ResourceRecord(
        timestamp_utc=datetime.now(UTC),
        epoch=epoch,
        elapsed_seconds=elapsed,
        gpu_peak_memory_mib=gpu,
        warning=warning,
    )


__all__ = ["ResourceRecord", "capture_resource"]
