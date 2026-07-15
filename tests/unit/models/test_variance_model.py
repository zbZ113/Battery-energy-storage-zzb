import math

import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.variance import DeltaQVarianceFeature
from quanxin_life.models import VarianceLifePredictor as ExportedVarianceLifePredictor
from quanxin_life.models.variance import VarianceLifePredictor


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("cell-a", "cell-b", "cell-c"),
        validation=("cell-d",),
        calibration=("cell-e",),
        test=("cell-f",),
    )


def _label(
    cell_id: str,
    observed_eol_cycle: int,
    *,
    right_censored: bool = False,
    cutoff_cycle: int = 20,
    feature_version: str = "delta-q-variance-v1",
    split_version: str = "matr-split-v1",
    data_version: str = "matr-data-v1",
) -> LifePrediction:
    return LifePrediction(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=cutoff_cycle,
        predicted_eol_cycle=float(observed_eol_cycle),
        observed_eol_cycle=None if right_censored else observed_eol_cycle,
        right_censored=right_censored,
        feature_version=feature_version,
        split_version=split_version,
        model_version="label-source-v1",
        data_version=data_version,
    )


def _feature(
    cell_id: str,
    variance: float | None,
    *,
    dataset_id: str = "MATR",
    cutoff_cycle: int = 20,
    feature_version: str = "delta-q-variance-v1",
    anchor_cycle: int = 1,
    comparison_cycle: int = 20,
) -> DeltaQVarianceFeature:
    return DeltaQVarianceFeature(
        dataset_id=dataset_id,
        cell_id=cell_id,
        feature_version=feature_version,
        cutoff_cycle=cutoff_cycle,
        anchor_cycle=anchor_cycle,
        comparison_cycle=comparison_cycle,
        valid_grid_point_count=100 if variance is not None else 0,
        delta_q_variance_ah2=variance,
        warnings=() if variance is not None else ("DELTA_Q_UNAVAILABLE",),
    )


def _predictor() -> VarianceLifePredictor:
    return VarianceLifePredictor(
        model_version="variance-eol80-v1",
        feature_version="delta-q-variance-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
        cutoff_cycle=20,
        anchor_cycle=1,
        comparison_cycle=20,
    )


def _fit() -> VarianceLifePredictor:
    return _predictor().fit(
        [_label("cell-a", 100), _label("cell-b", 1000), _label("cell-c", 10000)],
        training_features=[
            _feature("cell-a", 1e-6),
            _feature("cell-b", 1e-4),
            _feature("cell-c", 1e-2),
        ],
        split_manifest=_split(),
    )


def test_variance_predictor_is_lazily_exported() -> None:
    assert ExportedVarianceLifePredictor is VarianceLifePredictor


def test_fits_log_log_regression_and_predicts_eol80() -> None:
    prediction = _fit().predict(
        feature=_feature("cell-f", 1e-3),
        dataset_id="MATR",
        cell_id="cell-f",
        cutoff_cycle=20,
        feature_version="delta-q-variance-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert prediction.predicted_eol_cycle == pytest.approx(10**3.5)
    assert math.isfinite(prediction.predicted_eol_cycle)
    assert prediction.observed_eol_cycle is None
    assert prediction.right_censored is True
    assert prediction.model_version == "variance-eol80-v1"


def test_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        _predictor().predict(
            feature=_feature("cell-f", 1e-3),
            dataset_id="MATR",
            cell_id="cell-f",
            cutoff_cycle=20,
            feature_version="delta-q-variance-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )


@pytest.mark.parametrize(
    ("labels", "features", "pattern"),
    [
        (
            [_label("cell-d", 100), _label("cell-b", 1000)],
            [_feature("cell-d", 1e-6), _feature("cell-b", 1e-4)],
            "outside the train split",
        ),
        (
            [_label("cell-a", 100, right_censored=True), _label("cell-b", 1000)],
            [_feature("cell-a", 1e-6), _feature("cell-b", 1e-4)],
            "explicit observed EOL80",
        ),
        (
            [_label("cell-a", 100), _label("cell-b", 1000)],
            [_feature("cell-a", 1e-6)],
            "exactly match",
        ),
        (
            [_label("cell-a", 100), _label("cell-b", 1000)],
            [_feature("cell-a", 1e-6), _feature("cell-a", 1e-4)],
            "duplicate training feature cell_id",
        ),
        (
            [_label("cell-a", 100), _label("cell-a", 1000)],
            [_feature("cell-a", 1e-6), _feature("cell-b", 1e-4)],
            "duplicate training cell_id",
        ),
    ],
)
def test_fit_rejects_leakage_censoring_or_cell_set_errors(
    labels: list[LifePrediction],
    features: list[DeltaQVarianceFeature],
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _predictor().fit(labels, training_features=features, split_manifest=_split())


@pytest.mark.parametrize("variance", [None, 0.0])
def test_fit_rejects_unavailable_or_nonpositive_variance(variance: float | None) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _predictor().fit(
            [_label("cell-a", 100), _label("cell-b", 1000)],
            training_features=[_feature("cell-a", variance), _feature("cell-b", 1e-4)],
            split_manifest=_split(),
        )


@pytest.mark.parametrize(
    ("feature", "pattern"),
    [
        (_feature("cell-a", 1e-6, dataset_id="HUST"), "dataset_id"),
        (_feature("cell-a", 1e-6, cutoff_cycle=50), "cutoff_cycle"),
        (_feature("cell-a", 1e-6, feature_version="variance-v2"), "feature_version"),
        (_feature("cell-a", 1e-6, anchor_cycle=2), "anchor_cycle"),
        (_feature("cell-a", 1e-6, comparison_cycle=19), "comparison_cycle"),
    ],
)
def test_fit_rejects_feature_context_mismatch(
    feature: DeltaQVarianceFeature, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _predictor().fit(
            [_label("cell-a", 100), _label("cell-b", 1000)],
            training_features=[feature, _feature("cell-b", 1e-4)],
            split_manifest=_split(),
        )


def test_fit_requires_two_distinct_log_variance_values() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _predictor().fit(
            [_label("cell-a", 100), _label("cell-b", 1000)],
            training_features=[_feature("cell-a", 1e-4), _feature("cell-b", 1e-4)],
            split_manifest=_split(),
        )


def test_fit_rejects_bitwise_distinct_but_numerically_degenerate_variance() -> None:
    almost_identical_variance = 1e-4 * (1.0 + 1e-12)

    with pytest.raises(ValueError, match=r"ill-conditioned|insufficient numeric variation"):
        _predictor().fit(
            [_label("cell-a", 100), _label("cell-b", 1000)],
            training_features=[
                _feature("cell-a", 1e-4),
                _feature("cell-b", almost_identical_variance),
            ],
            split_manifest=_split(),
        )


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"dataset_id": "HUST"}, "dataset_id"),
        ({"cell_id": "other-cell"}, "cell_id"),
        ({"cutoff_cycle": 50}, "cutoff_cycle"),
        ({"feature_version": "variance-v2"}, "feature_version"),
        ({"split_version": "split-v2"}, "split_version"),
        ({"data_version": "data-v2"}, "data_version"),
    ],
)
def test_predict_rejects_explicit_context_mismatch(
    override: dict[str, str | int], pattern: str
) -> None:
    arguments: dict[str, object] = {
        "feature": _feature("cell-f", 1e-3),
        "dataset_id": "MATR",
        "cell_id": "cell-f",
        "cutoff_cycle": 20,
        "feature_version": "delta-q-variance-v1",
        "split_version": "matr-split-v1",
        "data_version": "matr-data-v1",
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=pattern):
        _fit().predict(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("feature", "pattern"),
    [
        (_feature("cell-f", None), "strictly positive"),
        (_feature("cell-f", 0.0), "strictly positive"),
        (_feature("cell-f", 1e-3, anchor_cycle=2), "anchor_cycle"),
        (_feature("cell-f", 1e-3, comparison_cycle=19), "comparison_cycle"),
    ],
)
def test_predict_rejects_invalid_feature(feature: DeltaQVarianceFeature, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        _fit().predict(
            feature=feature,
            dataset_id="MATR",
            cell_id="cell-f",
            cutoff_cycle=20,
            feature_version="delta-q-variance-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )


def test_predict_rejects_finite_result_before_cutoff() -> None:
    predictor = _predictor().fit(
        [_label("cell-a", 20), _label("cell-b", 21)],
        training_features=[_feature("cell-a", 1e-6), _feature("cell-b", 1e-4)],
        split_manifest=_split(),
    )

    with pytest.raises(RuntimeError, match="before the cutoff"):
        predictor.predict(
            feature=_feature("cell-f", 1e-12),
            dataset_id="MATR",
            cell_id="cell-f",
            cutoff_cycle=20,
            feature_version="delta-q-variance-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
