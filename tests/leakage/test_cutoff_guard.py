import pytest

from quanxin_life.data.leakage import assert_cycles_within_cutoff


def test_cutoff_guard_rejects_future_cycles() -> None:
    with pytest.raises(ValueError, match="after cutoff"):
        assert_cycles_within_cutoff([1, 20, 21], cutoff_cycle=20)


def test_cutoff_guard_accepts_boundary_cycle() -> None:
    assert_cycles_within_cutoff([1, 20], cutoff_cycle=20)

