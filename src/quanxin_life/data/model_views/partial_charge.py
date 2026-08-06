"""Partial-charge feature view helpers."""

from quanxin_life.core import TrainingReadableSplit
from quanxin_life.data.model_views.schemas import ModelViewRow


def partial_charge_row(
    cell_id: str,
    *,
    split: TrainingReadableSplit,
    features: dict[str, float | None],
) -> ModelViewRow:
    return ModelViewRow(
        entity_id=cell_id,
        split=split,
        features=features,
        feature_mask={name: value is not None for name, value in features.items()},
        target=None,
        evidence="NO_POINT_TARGET:OBSERVED_PARTIAL_CHARGE",
    )


__all__ = ["partial_charge_row"]
