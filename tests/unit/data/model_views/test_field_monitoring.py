import pytest

from quanxin_life.data.model_views.builder import validate_split_isolation
from quanxin_life.data.model_views.schemas import ModelViewRow


def test_field_system_cannot_cross_splits() -> None:
    rows = (
        ModelViewRow(
            entity_id="system-1",
            split="train",
            features={"x": 1.0},
            feature_mask={"x": True},
            evidence="NO_POINT_TARGET:SPLIT_TEST",
        ),
        ModelViewRow(
            entity_id="system-1",
            split="test",
            features={"x": 2.0},
            feature_mask={"x": True},
            evidence="NO_POINT_TARGET:SPLIT_TEST",
        ),
    )

    with pytest.raises(ValueError, match="crosses splits"):
        validate_split_isolation(rows)
