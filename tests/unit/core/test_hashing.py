import math

import pytest

from quanxin_life.core.hashing import sha256_canonical


def test_canonical_hash_is_stable() -> None:
    value = {"cell_id": "B0005", "cycles": [1, 2, 3], "valid": True}

    assert sha256_canonical(value) == sha256_canonical(value)
    assert len(sha256_canonical(value)) == 64


def test_mapping_key_order_does_not_change_hash() -> None:
    first = {"a": 1, "b": {"x": 2, "y": 3}}
    second = {"b": {"y": 3, "x": 2}, "a": 1}

    assert sha256_canonical(first) == sha256_canonical(second)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        sha256_canonical({"value": value})


def test_non_json_serializable_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="JSON"):
        sha256_canonical({"value": object()})
