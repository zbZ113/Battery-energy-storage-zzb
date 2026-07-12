import pytest

from quanxin_life.data.schemas import SplitManifest
from quanxin_life.data.split import build_cell_split


def test_split_is_deterministic_and_cell_disjoint() -> None:
    cells = [f"cell-{index:02d}" for index in range(20)]
    first = build_cell_split("MATR", cells)
    second = build_cell_split("MATR", list(reversed(cells)))

    assert first == second
    assert [len(first.train), len(first.validation), len(first.calibration), len(first.test)] == [12, 3, 2, 3]
    assert set(first.all_cells) == set(cells)


def test_manifest_rejects_a_cell_in_multiple_splits() -> None:
    with pytest.raises(ValueError, match="cell overlap"):
        SplitManifest(
            dataset_id="MATR",
            seed=20260712,
            train=("cell-1",),
            validation=("cell-1",),
            calibration=(),
            test=(),
        )


def test_explicit_preassignment_is_respected() -> None:
    result = build_cell_split(
        "MATR",
        [f"cell-{index}" for index in range(10)],
        preassigned={"test": ("cell-9",)},
    )

    assert "cell-9" in result.test
    assert "cell-9" not in result.train + result.validation + result.calibration

