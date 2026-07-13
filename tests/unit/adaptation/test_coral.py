from __future__ import annotations

import math

import pytest

from quanxin_life.data.schemas import SplitManifest


def _source_split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("source-a", "source-b", "source-c"),
        validation=("source-validation",),
        calibration=("source-calibration",),
        test=("source-test",),
    )


def _target_split() -> SplitManifest:
    return SplitManifest(
        dataset_id="HUST",
        train=("target-a", "target-b", "target-c"),
        validation=("target-validation",),
        calibration=("target-calibration",),
        test=("target-test",),
    )


def _source_features() -> dict[str, dict[str, float]]:
    return {
        "source-a": {"capacity_drop": 0.01, "resistance_growth": 0.10},
        "source-b": {"capacity_drop": 0.02, "resistance_growth": 0.15},
        "source-c": {"capacity_drop": 0.03, "resistance_growth": 0.30},
    }


def _target_features() -> dict[str, dict[str, float]]:
    return {
        "target-a": {"capacity_drop": 0.10, "resistance_growth": 0.80},
        "target-b": {"capacity_drop": 0.20, "resistance_growth": 0.90},
        "target-c": {"capacity_drop": 0.30, "resistance_growth": 1.20},
    }


def _adapter():
    from quanxin_life.adaptation.coral import CORALFeatureAdapter

    return CORALFeatureAdapter(
        adapter_version="coral-v1",
        feature_version="early-cycle-v1",
        source_split_version="matr-split-v1",
        target_split_version="hust-split-v1",
        feature_names=("capacity_drop", "resistance_growth"),
        covariance_regularization=1e-4,
    )


def test_coral_uses_only_train_cell_features_and_aligns_source_mean_to_target_domain() -> None:
    adapter = _adapter().fit(
        source_features=_source_features(),
        source_split_manifest=_source_split(),
        target_adaptation_features=_target_features(),
        target_split_manifest=_target_split(),
    )

    transformed = adapter.transform_source_features(_source_features())
    feature_names = ("capacity_drop", "resistance_growth")
    target_means = [
        sum(values[name] for values in _target_features().values()) / len(_target_features())
        for name in feature_names
    ]
    transformed_means = [
        sum(values[name] for values in transformed.values()) / len(transformed)
        for name in feature_names
    ]

    assert transformed_means == pytest.approx(target_means, abs=1e-7)
    assert adapter.source_dataset_id == "MATR"
    assert adapter.target_dataset_id == "HUST"


def test_coral_rejects_target_test_cells_and_nonfinite_features() -> None:
    unsafe_target = {
        **_target_features(),
        "target-test": {"capacity_drop": 0.4, "resistance_growth": 1.3},
    }
    with pytest.raises(ValueError, match=r"must exactly match target split_manifest\.train"):
        _adapter().fit(
            source_features=_source_features(),
            source_split_manifest=_source_split(),
            target_adaptation_features=unsafe_target,
            target_split_manifest=_target_split(),
        )

    nonfinite_source = _source_features()
    nonfinite_source["source-a"]["capacity_drop"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        _adapter().fit(
            source_features=nonfinite_source,
            source_split_manifest=_source_split(),
            target_adaptation_features=_target_features(),
            target_split_manifest=_target_split(),
        )


def test_coral_is_deterministic_and_requires_exact_feature_schema() -> None:
    first = _adapter().fit(
        source_features=_source_features(),
        source_split_manifest=_source_split(),
        target_adaptation_features=_target_features(),
        target_split_manifest=_target_split(),
    )
    second = _adapter().fit(
        source_features=_source_features(),
        source_split_manifest=_source_split(),
        target_adaptation_features=_target_features(),
        target_split_manifest=_target_split(),
    )

    first_transformed = first.transform_source_features(_source_features())
    second_transformed = second.transform_source_features(_source_features())
    for cell_id, first_row in first_transformed.items():
        assert first_row == pytest.approx(second_transformed[cell_id], abs=1e-12)
    malformed = _source_features()
    malformed["source-a"] = {"capacity_drop": 0.01}
    with pytest.raises(ValueError, match="feature schema"):
        first.transform_source_features(malformed)
