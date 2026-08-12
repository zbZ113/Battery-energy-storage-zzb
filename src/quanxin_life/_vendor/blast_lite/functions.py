"""Minimal BLAST-Lite utility subset from upstream ``blast.utils.functions``."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def rescale_soc(soc: FloatArray, rescaling_factor: float) -> FloatArray:
    """Rescale an SOC vector by multiplying each SOC change."""
    delta_soc = np.diff(soc, prepend=soc[0]) * rescaling_factor
    rescaled = np.cumsum(delta_soc) + soc[0]
    if np.max(rescaled) > 1 or np.min(rescaled) < 0:
        rescaled = np.maximum(0, np.minimum(1, rescaled))
    return cast("FloatArray", rescaled)


__all__ = ["rescale_soc"]
