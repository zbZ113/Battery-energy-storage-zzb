"""Early-life sequence view helpers."""

from quanxin_life.core import TrainingReadableSplit
from quanxin_life.data.model_views.schemas import ModelViewRow


def early_life_row(
    entity_id: str,
    *,
    split: TrainingReadableSplit,
    features: dict[str, float | None],
    target: float | None,
    right_censored: bool = False,
) -> ModelViewRow:
    return ModelViewRow(
        entity_id=entity_id,
        split=split,
        features=features,
        feature_mask={name: value is not None for name, value in features.items()},
        target=target,
        right_censored=right_censored,
    )


__all__ = ["early_life_row"]
