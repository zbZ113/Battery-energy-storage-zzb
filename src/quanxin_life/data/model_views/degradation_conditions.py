"""Condition-level degradation view helpers."""

from quanxin_life.core import TrainingReadableSplit
from quanxin_life.data.model_views.schemas import ModelViewRow


def condition_view_row(
    condition_id: str,
    *,
    split: TrainingReadableSplit,
    temperature_c: float,
    mean_soc: float,
) -> ModelViewRow:
    return ModelViewRow(
        entity_id=condition_id,
        split=split,
        features={"temperature_c": temperature_c, "mean_soc": mean_soc},
        feature_mask={"temperature_c": True, "mean_soc": True},
        target=None,
        evidence="NO_POINT_TARGET:OBSERVED_CONDITION",
    )


__all__ = ["condition_view_row"]
