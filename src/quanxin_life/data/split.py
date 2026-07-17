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
    train = math.floor(total * ratios[0])
    validation = math.floor(total * ratios[1])
    calibration = math.floor(total * ratios[2])
    test = total - train - validation - calibration
    return train, validation, calibration, test


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
    targets = dict(
        zip(
            (name.value for name in SplitName),
            _target_counts(len(cells), ratios),
            strict=True,
        )
    )
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


def build_stratified_cell_split(
    dataset_id: str,
    cell_strata: Mapping[str, str],
    *,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> SplitManifest:
    """Create exact-size cell partitions while preserving sparse strata greedily."""

    if not cell_strata or any(not cell for cell in cell_strata):
        raise ValueError("at least one non-empty cell identifier is required")
    if any(not isinstance(stratum, str) or not stratum.strip() for stratum in cell_strata.values()):
        raise ValueError("every cell requires a non-empty stratum")

    names = tuple(member.value for member in SplitName)
    targets = dict(zip(names, _target_counts(len(cell_strata), ratios), strict=True))
    assignments: dict[str, list[str]] = {name: [] for name in names}
    by_stratum: dict[str, list[str]] = {}
    for cell_id, stratum in sorted(cell_strata.items()):
        by_stratum.setdefault(stratum, []).append(cell_id)

    generator = random.Random(seed)
    strata = sorted(by_stratum)
    generator.shuffle(strata)
    for stratum in strata:
        cells = sorted(by_stratum[stratum])
        generator.shuffle(cells)
        stratum_counts = {name: 0 for name in names}
        desired = {
            name: len(cells) * ratio for name, ratio in zip(names, ratios, strict=True)
        }
        tie_order = list(names)
        generator.shuffle(tie_order)
        tie_rank = {name: index for index, name in enumerate(tie_order)}

        for cell_id in cells:
            candidates = [
                name for name in names if len(assignments[name]) < targets[name]
            ]
            if not candidates:
                raise ValueError("split capacity was exhausted before all cells were assigned")

            scored_candidates: list[tuple[tuple[float, float, int], str]] = []
            for name in candidates:
                before = stratum_counts[name] - desired[name]
                after = stratum_counts[name] + 1 - desired[name]
                marginal_error = after * after - before * before
                remaining_share = (targets[name] - len(assignments[name])) / max(
                    targets[name], 1
                )
                scored_candidates.append(
                    ((marginal_error, -remaining_share, tie_rank[name]), name)
                )

            selected = min(scored_candidates)[1]
            assignments[selected].append(cell_id)
            stratum_counts[selected] += 1

    return SplitManifest(
        dataset_id=dataset_id,
        seed=seed,
        train=tuple(sorted(assignments[SplitName.TRAIN.value])),
        validation=tuple(sorted(assignments[SplitName.VALIDATION.value])),
        calibration=tuple(sorted(assignments[SplitName.CALIBRATION.value])),
        test=tuple(sorted(assignments[SplitName.TEST.value])),
    )

