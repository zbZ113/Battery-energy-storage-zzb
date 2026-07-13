import math

import pytest
import torch

from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.curve_tensor import CurveTensor
from quanxin_life.models import CPMLPLifePredictor as ExportedCPMLPLifePredictor
from quanxin_life.models.cpmlp import CPMLPLifePredictor, curve_tensor_to_tensors


def test_cpmlp_export_is_lazy_but_available_when_ml_extra_is_installed() -> None:
    assert ExportedCPMLPLifePredictor is CPMLPLifePredictor


def _curve(
    *,
    cell_id: str,
    capacity_scale: float = 1.0,
    feature_version: str = "discharge-curve-tensor-v1",
    malformed_observed_value: float | None = None,
    with_observed_curve: bool = True,
) -> CurveTensor:
    cycles = tuple(range(21))
    voltage_grid = (3.0, 3.1, 3.2)
    values: list[tuple[float | None, ...]] = [tuple(None for _ in voltage_grid) for _ in cycles]
    observed = [False for _ in cycles]
    if with_observed_curve:
        values[1] = (0.60 * capacity_scale, 0.30 * capacity_scale, 0.00)
        values[2] = (0.58 * capacity_scale, 0.29 * capacity_scale, 0.00)
        observed[1] = True
        observed[2] = True
    if malformed_observed_value is not None:
        values[1] = (0.60 * capacity_scale, malformed_observed_value, 0.00)
        observed[1] = True

    return CurveTensor(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        feature_version=feature_version,
        cycle_indices=cycles,
        voltage_grid_v=voltage_grid,
        values=tuple(values),
        observed_mask=tuple(observed),
    )


def _label(
    *,
    cell_id: str,
    observed_eol_cycle: int,
    feature_version: str = "discharge-curve-tensor-v1",
    right_censored: bool = False,
) -> LifePrediction:
    return LifePrediction(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        predicted_eol_cycle=float(observed_eol_cycle),
        observed_eol_cycle=None if right_censored else observed_eol_cycle,
        right_censored=right_censored,
        feature_version=feature_version,
        split_version="matr-split-v1",
        model_version="label-source-v1",
        data_version="matr-data-v1",
    )


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("cell-a", "cell-b", "cell-c", "cell-d"),
        validation=("cell-e",),
        calibration=("cell-f",),
        test=("cell-g",),
    )


def _predictor() -> CPMLPLifePredictor:
    return CPMLPLifePredictor(
        model_version="cpmlp-eol80-v1",
        feature_version="discharge-curve-tensor-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
        cutoff_cycle=20,
        curve_hidden_dim=8,
        aggregation_hidden_dim=8,
        epochs=20,
        learning_rate=0.03,
    )


def _training_labels() -> list[LifePrediction]:
    return [
        _label(cell_id="cell-a", observed_eol_cycle=90),
        _label(cell_id="cell-b", observed_eol_cycle=130),
        _label(cell_id="cell-c", observed_eol_cycle=190),
        _label(cell_id="cell-d", observed_eol_cycle=260),
    ]


def _training_curves() -> dict[str, CurveTensor]:
    return {
        "cell-a": _curve(cell_id="cell-a", capacity_scale=0.80),
        "cell-b": _curve(cell_id="cell-b", capacity_scale=0.90),
        "cell-c": _curve(cell_id="cell-c", capacity_scale=1.10),
        "cell-d": _curve(cell_id="cell-d", capacity_scale=1.20),
    }


def test_fit_uses_masked_curve_tensors_from_train_cells_and_predicts_eol80() -> None:
    predictor = _predictor().fit(
        _training_labels(), training_curves=_training_curves(), split_manifest=_split()
    )

    prediction = predictor.predict(
        curve=_curve(cell_id="cell-g", capacity_scale=1.0),
        cutoff_cycle=20,
        feature_version="discharge-curve-tensor-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert prediction.right_censored is True
    assert prediction.observed_eol_cycle is None
    assert prediction.predicted_eol_cycle >= 20
    assert math.isfinite(prediction.predicted_eol_cycle)
    assert prediction.model_version == "cpmlp-eol80-v1"


def test_tensor_adapter_preserves_missing_rows_as_nan_until_masked_by_network() -> None:
    values, observed_mask = curve_tensor_to_tensors(_curve(cell_id="cell-a"))

    assert observed_mask.dtype is torch.bool
    assert observed_mask[0].item() is False
    assert values[0].isnan().all().item() is True
    assert observed_mask[1].item() is True
    assert values[1].isfinite().all().item() is True


@pytest.mark.parametrize(
    ("curves", "labels", "pattern"),
    [
        (
            {
                **_training_curves(),
                "cell-g": _curve(cell_id="cell-g"),
            },
            _training_labels(),
            "must exactly match",
        ),
        (
            {
                **_training_curves(),
                "cell-c": _curve(cell_id="cell-c", malformed_observed_value=float("nan")),
            },
            _training_labels(),
            "non-finite",
        ),
        (
            {
                **_training_curves(),
                "cell-c": _curve(cell_id="cell-c", with_observed_curve=False),
            },
            _training_labels(),
            "at least one observed curve",
        ),
        (
            _training_curves(),
            [_label(cell_id="cell-a", observed_eol_cycle=90, right_censored=True)],
            "explicit observed EOL80",
        ),
        (
            {"cell-a": _curve(cell_id="cell-a")},
            [
                _label(cell_id="cell-a", observed_eol_cycle=90),
                _label(cell_id="cell-a", observed_eol_cycle=100),
            ],
            "duplicate training cell_id",
        ),
    ],
)
def test_fit_rejects_invalid_training_cohorts(
    curves: dict[str, CurveTensor], labels: list[LifePrediction], pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _predictor().fit(labels, training_curves=curves, split_manifest=_split())


def test_predict_rejects_curve_feature_version_mismatch() -> None:
    predictor = _predictor().fit(
        _training_labels(), training_curves=_training_curves(), split_manifest=_split()
    )

    with pytest.raises(ValueError, match="feature_version"):
        predictor.predict(
            curve=_curve(cell_id="cell-g", feature_version="discharge-curve-tensor-v2"),
            cutoff_cycle=20,
            feature_version="discharge-curve-tensor-v2",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )


def test_fit_rejects_curve_context_version_mismatch() -> None:
    curves = _training_curves()
    curves["cell-c"] = _curve(cell_id="cell-c", feature_version="discharge-curve-tensor-v2")

    with pytest.raises(ValueError, match="feature_version"):
        _predictor().fit(_training_labels(), training_curves=curves, split_manifest=_split())


def test_predict_rejects_mismatched_curve_axis() -> None:
    predictor = _predictor().fit(
        _training_labels(), training_curves=_training_curves(), split_manifest=_split()
    )
    incompatible_curve = CurveTensor(
        dataset_id="MATR",
        cell_id="cell-g",
        cutoff_cycle=20,
        feature_version="discharge-curve-tensor-v1",
        cycle_indices=tuple(range(21)),
        voltage_grid_v=(3.0, 3.2),
        values=tuple(
            (0.60, 0.00) if cycle_index == 1 else (None, None) for cycle_index in range(21)
        ),
        observed_mask=tuple(cycle_index == 1 for cycle_index in range(21)),
    )

    with pytest.raises(ValueError, match="voltage_grid_v"):
        predictor.predict(
            curve=incompatible_curve,
            cutoff_cycle=20,
            feature_version="discharge-curve-tensor-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )
