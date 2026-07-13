import pytest

from quanxin_life.core import LifePrediction
from quanxin_life.models.metrics import evaluate_eol80_predictions


def _prediction(
    *,
    cell_id: str,
    predicted_eol_cycle: float,
    observed_eol_cycle: int,
    dataset_id: str = "MATR",
    cutoff_cycle: int = 20,
    feature_version: str = "early-cycle-v1",
    split_version: str = "matr-split-v1",
    model_version: str = "xgboost-eol80-v1",
    data_version: str = "matr-data-v1",
    right_censored: bool = False,
) -> LifePrediction:
    return LifePrediction(
        dataset_id=dataset_id,
        cell_id=cell_id,
        cutoff_cycle=cutoff_cycle,
        predicted_eol_cycle=predicted_eol_cycle,
        observed_eol_cycle=None if right_censored else observed_eol_cycle,
        right_censored=right_censored,
        feature_version=feature_version,
        split_version=split_version,
        model_version=model_version,
        data_version=data_version,
    )


def test_evaluates_unique_observed_cells_with_shared_context() -> None:
    metrics = evaluate_eol80_predictions(
        [
            _prediction(cell_id="cell-a", predicted_eol_cycle=100.0, observed_eol_cycle=110),
            _prediction(cell_id="cell-b", predicted_eol_cycle=230.0, observed_eol_cycle=220),
        ]
    )

    assert metrics.evaluated_cell_count == 2
    assert metrics.mae_cycle == pytest.approx(10.0)
    assert metrics.rmse_cycle == pytest.approx(10.0)
    assert metrics.mape_percent == pytest.approx((10 / 110 + 10 / 220) * 50)
    assert metrics.r2 == pytest.approx(1 - 200 / 6050)
    assert metrics.warnings == []


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"dataset_id": "HUST"}, "dataset_id"),
        ({"cutoff_cycle": 50}, "cutoff_cycle"),
        ({"feature_version": "early-cycle-v2"}, "feature_version"),
        ({"split_version": "matr-split-v2"}, "split_version"),
        ({"model_version": "xgboost-eol80-v2"}, "model_version"),
        ({"data_version": "matr-data-v2"}, "data_version"),
    ],
)
def test_rejects_mixed_evaluation_context(override: dict[str, str | int], pattern: str) -> None:
    predictions = [
        _prediction(cell_id="cell-a", predicted_eol_cycle=100.0, observed_eol_cycle=110),
        _prediction(
            cell_id="cell-b",
            predicted_eol_cycle=230.0,
            observed_eol_cycle=220,
            **override,
        ),
    ]

    with pytest.raises(ValueError, match=pattern):
        evaluate_eol80_predictions(predictions)


def test_rejects_duplicate_cell_id() -> None:
    predictions = [
        _prediction(cell_id="cell-a", predicted_eol_cycle=100.0, observed_eol_cycle=110),
        _prediction(cell_id="cell-a", predicted_eol_cycle=230.0, observed_eol_cycle=220),
    ]

    with pytest.raises(ValueError, match="duplicate cell_id"):
        evaluate_eol80_predictions(predictions)


def test_rejects_right_censored_prediction() -> None:
    with pytest.raises(ValueError, match="right-censored"):
        evaluate_eol80_predictions(
            [
                _prediction(
                    cell_id="cell-a",
                    predicted_eol_cycle=100.0,
                    observed_eol_cycle=110,
                    right_censored=True,
                )
            ]
        )


def test_returns_undefined_r2_warning_for_constant_observed_labels() -> None:
    metrics = evaluate_eol80_predictions(
        [
            _prediction(cell_id="cell-a", predicted_eol_cycle=100.0, observed_eol_cycle=110),
            _prediction(cell_id="cell-b", predicted_eol_cycle=120.0, observed_eol_cycle=110),
        ]
    )

    assert metrics.r2 is None
    assert metrics.warnings == ["R2_UNDEFINED_CONSTANT_OBSERVED_EOL80"]


def test_rejects_empty_evaluation_cohort() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_eol80_predictions([])
