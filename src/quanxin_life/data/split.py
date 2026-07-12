import math
import random
from collections.abc import Iterable, Mapping, Sequence

from quanxin_life.data.schemas import SplitManifest, SplitName

DEFAULT_RATIOS = (0.60, 0.15, 0.10, 0.15)
DEFAULT_SEED = 20260712


def _target_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int, int]:
    if len(ratios) != 4 or any(ratio < 0 for ratio in ratios):
        raise ValueError("four non-negative split ratios are required")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    first_three = tuple(math.floor(total * ratio) for ratio in ratios[:3])
    return (*first_three, total - sum(first_three))


def build_cell_split(
    dataset_id: str,
    cell_ids: Iterable[str],
    *,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    preassigned: Mapping[str, Sequence[str]] | None = None,
) -> SplitManifest:
    """Create deterministic, mutually exclusive cell-level partitions."""
    cells = sorted(cell_ids)
    if not cells or any(not cell for cell in cells):
        raise ValueError("at least one non-empty cell identifier is required")
    if len(cells) != len(set(cells)):
        raise ValueError("duplicate cell identifier")

    assignments: dict[str, list[str]] = {name.value: [] for name in SplitName}
    claimed: set[str] = set()
    for raw_name, assigned_cells in (preassigned or {}).items():
        try:
            name = SplitName(raw_name).value
        except ValueError as exc:
            raise ValueError(f"unknown split name: {raw_name}") from exc
        for cell in assigned_cells:
            if cell not in cells:
                raise ValueError(f"preassigned cell is not in dataset: {cell}")
            if cell in claimed:
                raise ValueError(f"cell overlap in preassignment: {cell}")
            assignments[name].append(cell)
            claimed.add(cell)

    remaining = [cell for cell in cells if cell not in claimed]
    random.Random(seed).shuffle(remaining)
    targets = dict(zip((name.value for name in SplitName), _target_counts(len(cells), ratios)))
    for name in (member.value for member in SplitName):
        needed = max(targets[name] - len(assignments[name]), 0)
        assignments[name].extend(remaining[:needed])
        del remaining[:needed]
    assignments[SplitName.TEST.value].extend(remaining)

    return SplitManifest(
        dataset_id=dataset_id,
        seed=seed,
        train=tuple(sorted(assignments[SplitName.TRAIN.value])),
        validation=tuple(sorted(assignments[SplitName.VALIDATION.value])),
        calibration=tuple(sorted(assignments[SplitName.CALIBRATION.value])),
        test=tuple(sorted(assignments[SplitName.TEST.value])),
    )

