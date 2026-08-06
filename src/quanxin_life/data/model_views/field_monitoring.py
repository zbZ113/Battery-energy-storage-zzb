"""System-grouped field monitoring view helpers."""

from quanxin_life.core import TrainingReadableSplit
from quanxin_life.data.model_views.schemas import ModelViewRow


def field_monitoring_row(
    system_id: str,
    *,
    split: TrainingReadableSplit,
    features: dict[str, float | None],
) -> ModelViewRow:
    return ModelViewRow(
        entity_id=system_id,
        split=split,
        features=features,
        feature_mask={name: value is not None for name, value in features.items()},
        target=None,
        evidence="NO_POINT_TARGET:OBSERVED_NO_HEALTH_LABEL",
    )


__all__ = ["field_monitoring_row"]
