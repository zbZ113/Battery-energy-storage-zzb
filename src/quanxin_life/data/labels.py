from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cycle_index: int = Field(ge=0)
    soh: float = Field(ge=0)
    valid: bool = True


class EOL80Label(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(default=0.8, gt=0, lt=1)
    eol_cycle: int | None
    confirmation_cycles: tuple[int, ...] = ()
    right_censored: bool


def derive_eol80(
    points: Iterable[DiagnosticPoint], threshold: float = 0.8, confirmations: int = 3
) -> EOL80Label:
    """Derive EOL only after persistent low-SOH diagnostic observations."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    if confirmations < 1:
        raise ValueError("confirmations must be positive")

    valid_points = sorted(
        (point for point in points if point.valid), key=lambda point: point.cycle_index
    )
    for index in range(len(valid_points) - confirmations + 1):
        window = valid_points[index : index + confirmations]
        if all(point.soh <= threshold for point in window):
            cycles = tuple(point.cycle_index for point in window)
            return EOL80Label(
                threshold=threshold,
                eol_cycle=cycles[0],
                confirmation_cycles=cycles,
                right_censored=False,
            )
    return EOL80Label(threshold=threshold, eol_cycle=None, right_censored=True)

