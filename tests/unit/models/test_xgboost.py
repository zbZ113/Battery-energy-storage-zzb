import math

import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models import XGBoostLifePredictor as ExportedXGBoostLifePredictor
from quanxin_life.models.xgboost import XGBoostLifePredictor

FEATURE_NAMES = ("capacity_delta_ah", "delta_q_variance_ah2")


def test_xgboost_export_is_lazy_but_available_when_ml_extra_is_installed() -> None:
    assert ExportedXGBoostLifePredictor is XGBoostLifePredictor


def _label(
    *,
    cell_id: str,
    observed_eol_cycle: int = 200,
    cutoff_cycle: int = 20,
    right_censored: bool = False,
    feature_version: str = "early-cycle-v1",
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


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("cell-a", "cell-b", "cell-c", "cell-d"),
        validation=("cell-e",),
        calibration=("cell-f",),
        test=("cell-g",),
    )


def _predictor() -> XGBoostLifePredictor:
    return XGBoostLifePredictor(
        model_version="xgboost-eol80-v1",
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
        cutoff_cycle=20,
        feature_names=FEATURE_NAMES,
    )


def _training_labels() -> list[LifePrediction]:
    return [
        _label(cell_id="cell-a", observed_eol_cycle=100),
        _label(cell_id="cell-b", observed_eol_cycle=180),
        _label(cell_id="cell-c", observed_eol_cycle=260),
        _label(cell_id="cell-d", observed_eol_cycle=340),
    ]


def _training_features() -> dict[str, dict[str, float]]:
    return {
        "cell-a": {"capacity_delta_ah": -0.02, "delta_q_variance_ah2": 0.01},
        "cell-b": {"capacity_delta_ah": -0.04, "delta_q_variance_ah2": 0.02},
        "cell-c": {"capacity_delta_ah": -0.07, "delta_q_variance_ah2": 0.04},
        "cell-d": {"capacity_delta_ah": -0.10, "delta_q_variance_ah2": 0.08},
    }


def test_fits_real_xgboost_on_train_cells_and_predicts_eol80() -> None:
    predictor = _predictor()
    predictor.fit(
        _training_labels(),
        training_features=_training_features(),
        split_manifest=_split(),
    )

    prediction = predictor.predict(
        dataset_id="MATR",
        cell_id="cell-g",
        features={"capacity_delta_ah": -0.08, "delta_q_variance_ah2": 0.05},
        cutoff_cycle=20,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert prediction.target.value == "eol80_cycle"
    assert prediction.right_censored is True
    assert prediction.observed_eol_cycle is None
    assert prediction.predicted_eol_cycle >= 20
    assert math.isfinite(prediction.predicted_eol_cycle)
    assert prediction.model_version == "xgboost-eol80-v1"


@pytest.mark.parametrize(
    ("labels", "features", "pattern"),
    [
        (
            [_label(cell_id="cell-e")],
            {"cell-e": {"capacity_delta_ah": -0.02, "delta_q_variance_ah2": 0.01}},
            "outside the train split",
        ),
        (
            [_label(cell_id="cell-a", right_censored=True)],
            {"cell-a": {"capacity_delta_ah": -0.02, "delta_q_variance_ah2": 0.01}},
            "explicit observed EOL80",
        ),
        (
            _training_labels(),
            {
                **_training_features(),
                "cell-g": {"capacity_delta_ah": -0.11, "delta_q_variance_ah2": 0.09},
            },
            "must exactly match",
        ),
        (
            _training_labels(),
            {
                **_training_features(),
                "cell-c": {"capacity_delta_ah": float("nan"), "delta_q_variance_ah2": 0.04},
            },
            "finite",
        ),
    ],
)
def test_fit_rejects_outside_censored_or_invalid_training_inputs(
    labels: list[LifePrediction], features: dict[str, dict[str, float]], pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _predictor().fit(labels, training_features=features, split_manifest=_split())


def test_fit_rejects_missing_or_unexpected_declared_feature_names() -> None:
    features = _training_features()
    features["cell-c"] = {"capacity_delta_ah": -0.07, "unexpected": 0.04}

    with pytest.raises(ValueError, match="feature schema"):
        _predictor().fit(_training_labels(), training_features=features, split_manifest=_split())


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"dataset_id": "HUST"}, "dataset_id"),
        ({"cutoff_cycle": 50}, "cutoff_cycle"),
        ({"feature_version": "early-cycle-v2"}, "feature_version"),
        ({"split_version": "matr-split-v2"}, "split_version"),
        ({"data_version": "matr-data-v2"}, "data_version"),
    ],
)
def test_predict_rejects_context_version_mismatch(
    override: dict[str, str | int], pattern: str
) -> None:
    predictor = _predictor().fit(
        _training_labels(),
        training_features=_training_features(),
        split_manifest=_split(),
    )
    arguments: dict[str, object] = {
        "dataset_id": "MATR",
        "cell_id": "cell-g",
        "features": {"capacity_delta_ah": -0.08, "delta_q_variance_ah2": 0.05},
        "cutoff_cycle": 20,
        "feature_version": "early-cycle-v1",
        "split_version": "matr-split-v1",
        "data_version": "matr-data-v1",
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=pattern):
        predictor.predict(**arguments)  # type: ignore[arg-type]


def test_predict_rejects_nonfinite_or_schema_mismatched_features() -> None:
    predictor = _predictor().fit(
        _training_labels(),
        training_features=_training_features(),
        split_manifest=_split(),
    )

    with pytest.raises(ValueError, match="feature schema"):
        predictor.predict(
            dataset_id="MATR",
            cell_id="cell-g",
            features={"capacity_delta_ah": -0.08},
            cutoff_cycle=20,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(
            dataset_id="MATR",
            cell_id="cell-g",
            features={"capacity_delta_ah": -0.08, "delta_q_variance_ah2": float("inf")},
            cutoff_cycle=20,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
