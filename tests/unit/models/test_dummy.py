import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.dummy import DummyLifePredictor


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
        train=("cell-a", "cell-b"),
        validation=("cell-c",),
        calibration=("cell-d",),
        test=("cell-e",),
    )


def _predictor() -> DummyLifePredictor:
    return DummyLifePredictor(
        model_version="dummy-eol80-v1",
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
        cutoff_cycle=20,
    )


def test_fit_uses_only_explicit_train_cell_eol80_labels() -> None:
    predictor = _predictor()
    predictor.fit(
        [
            _label(cell_id="cell-a", observed_eol_cycle=200),
            _label(cell_id="cell-b", observed_eol_cycle=300),
        ],
        split_manifest=_split(),
    )

    prediction = predictor.predict(
        dataset_id="MATR",
        cell_id="cell-e",
        cutoff_cycle=20,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert prediction.predicted_eol_cycle == pytest.approx(250.0)
    assert prediction.observed_eol_cycle is None
    assert prediction.right_censored is True
    assert prediction.feature_version == "early-cycle-v1"
    assert prediction.split_version == "matr-split-v1"
    assert prediction.model_version == "dummy-eol80-v1"
    assert prediction.data_version == "matr-data-v1"


@pytest.mark.parametrize(
    "labels, pattern",
    [
        (
            [_label(cell_id="cell-a"), _label(cell_id="cell-a", observed_eol_cycle=220)],
            "duplicate training cell_id",
        ),
        ([_label(cell_id="cell-e")], "outside the train split"),
        ([_label(cell_id="cell-a", right_censored=True)], "explicit observed EOL80"),
    ],
)
def test_fit_rejects_non_train_or_non_observed_labels(
    labels: list[LifePrediction], pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _predictor().fit(labels, split_manifest=_split())


def test_predict_rejects_cutoff_or_version_mismatch_after_fit() -> None:
    predictor = _predictor()
    predictor.fit(
        [_label(cell_id="cell-a"), _label(cell_id="cell-b", observed_eol_cycle=300)],
        split_manifest=_split(),
    )

    with pytest.raises(ValueError, match="cutoff_cycle"):
        predictor.predict(
            dataset_id="MATR",
            cell_id="cell-e",
            cutoff_cycle=50,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
    with pytest.raises(ValueError, match="feature_version"):
        predictor.predict(
            dataset_id="MATR",
            cell_id="cell-e",
            cutoff_cycle=20,
            feature_version="early-cycle-v2",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )


def test_predict_requires_a_fitted_training_baseline() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        _predictor().predict(
            dataset_id="MATR",
            cell_id="cell-e",
            cutoff_cycle=20,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
