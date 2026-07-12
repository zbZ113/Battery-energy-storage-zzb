from collections.abc import Iterable


def assert_cycles_within_cutoff(cycle_indices: Iterable[int], *, cutoff_cycle: int) -> None:
    if cutoff_cycle < 0:
        raise ValueError("cutoff_cycle must be non-negative")
    future = sorted({cycle for cycle in cycle_indices if cycle > cutoff_cycle})
    if future:
        raise ValueError(f"feature input contains cycles after cutoff {cutoff_cycle}: {future[:5]}")

