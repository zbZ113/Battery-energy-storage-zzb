import math
from itertools import pairwise

import pytest

from quanxin_life.data.schemas import SplitManifest


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("cell-a", "cell-b", "cell-c", "cell-d"),
        validation=("cell-e",),
        calibration=("cell-f",),
        test=("cell-g",),
    )


def _sample(
    *,
    cell_id: str,
    initial_soh: float,
    degradation_scale: float,
):
    from quanxin_life.models.hybrid_degradation import TrajectoryTrainingSample

    target_cycles = tuple(range(21, 61))
    return TrajectoryTrainingSample(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        observed_cycles=(0, 10, 20),
        observed_soh=(initial_soh, initial_soh - 0.01, initial_soh - 0.02),
        target_cycles=target_cycles,
        target_soh=tuple(
            initial_soh - 0.02 - degradation_scale * (cycle - 20) for cycle in target_cycles
        ),
        condition_features={
            "temperature_c": 25.0 + degradation_scale * 100.0,
            "mean_soc": 0.5,
        },
        feature_version="early-cycle-v1",
        data_version="matr-data-v1",
    )


def _predictor():
    from quanxin_life.models.hybrid_degradation import HybridDegradationPredictor

    return HybridDegradationPredictor(
        model_version="hybrid-degradation-v1",
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
        cutoff_cycle=20,
        prediction_cycles=tuple(range(21, 61)),
        condition_feature_names=("temperature_c", "mean_soc"),
        hidden_dim=8,
        epochs=30,
        learning_rate=0.03,
    )


def _training_samples():
    return [
        _sample(cell_id="cell-a", initial_soh=1.0, degradation_scale=0.003),
        _sample(cell_id="cell-b", initial_soh=1.0, degradation_scale=0.004),
        _sample(cell_id="cell-c", initial_soh=0.99, degradation_scale=0.005),
        _sample(cell_id="cell-d", initial_soh=0.99, degradation_scale=0.006),
    ]


def test_hybrid_predictor_emits_structurally_monotone_trajectory_from_early_inputs() -> None:
    predictor = _predictor().fit(_training_samples(), split_manifest=_split())
    prediction = predictor.predict(
        dataset_id="MATR",
        cell_id="cell-g",
        observed_cycles=(0, 10, 20),
        observed_soh=(1.0, 0.99, 0.98),
        condition_features={"temperature_c": 28.0, "mean_soc": 0.5},
        cutoff_cycle=20,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert prediction.prediction_cycles == tuple(range(21, 61))
    assert len(prediction.predicted_soh) == len(prediction.prediction_cycles)
    assert all(math.isfinite(value) for value in prediction.predicted_soh)
    assert all(
        current <= previous
        for previous, current in pairwise(prediction.predicted_soh)
    )
    assert prediction.model_version == "hybrid-degradation-v1"
    assert prediction.derived_rul_cycle is None or prediction.derived_rul_cycle >= 1


def test_eol80_crossing_and_rul_are_derived_from_trajectory_not_an_independent_head() -> None:
    from quanxin_life.models.hybrid_degradation import derive_eol80_crossing

    crossing = derive_eol80_crossing(
        prediction_cycles=(21, 22, 23, 24),
        predicted_soh=(0.83, 0.81, 0.80, 0.79),
        cutoff_cycle=20,
    )

    assert crossing.eol80_cycle == 23
    assert crossing.derived_rul_cycle == 3


def test_nonuniform_prediction_cycles_use_their_actual_cycle_spacing() -> None:
    from quanxin_life.models.hybrid_degradation import normalise_prediction_cycle_positions

    positions = normalise_prediction_cycle_positions(
        prediction_cycles=(21, 22, 60), cutoff_cycle=20
    )

    assert positions == pytest.approx((1.0 / 40.0, 2.0 / 40.0, 1.0))


def test_prediction_contract_rejects_eol_crossing_inconsistent_with_trajectory() -> None:
    from quanxin_life.models.hybrid_degradation import EOL80Crossing, SOHTrajectoryPrediction

    with pytest.raises(ValueError, match="must be derived from predicted_soh"):
        SOHTrajectoryPrediction(
            dataset_id="MATR",
            cell_id="cell-g",
            cutoff_cycle=20,
            prediction_cycles=(21, 22, 23),
            predicted_soh=(0.83, 0.80, 0.79),
            eol80_crossing=EOL80Crossing(cutoff_cycle=20),
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            model_version="hybrid-degradation-v1",
            data_version="matr-data-v1",
        )


def test_predict_rejects_cell_already_at_eol80_at_cutoff() -> None:
    predictor = _predictor().fit(_training_samples(), split_manifest=_split())

    with pytest.raises(ValueError, match="already reached EOL80"):
        predictor.predict(
            dataset_id="MATR",
            cell_id="cell-g",
            observed_cycles=(0, 10, 20),
            observed_soh=(1.0, 0.90, 0.79),
            condition_features={"temperature_c": 28.0, "mean_soc": 0.5},
            cutoff_cycle=20,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            data_version="matr-data-v1",
        )


def test_training_rejects_future_target_leakage_and_non_train_cells() -> None:
    from quanxin_life.models.hybrid_degradation import TrajectoryTrainingSample

    leaky = _sample(cell_id="cell-a", initial_soh=1.0, degradation_scale=0.003)
    with pytest.raises(ValueError, match="strictly after cutoff_cycle"):
        TrajectoryTrainingSample(
            **{
                **leaky.model_dump(),
                "target_cycles": (20, *leaky.target_cycles[1:]),
            }
        )

    with pytest.raises(ValueError, match="must exactly match"):
        samples = [
            *_training_samples()[:-1],
            _sample(cell_id="cell-g", initial_soh=0.99, degradation_scale=0.006),
        ]
        _predictor().fit(
            samples,
            split_manifest=_split(),
        )


def test_fixed_seed_produces_repeatable_predictions() -> None:
    first = _predictor().fit(_training_samples(), split_manifest=_split()).predict(
        dataset_id="MATR",
        cell_id="cell-g",
        observed_cycles=(0, 10, 20),
        observed_soh=(1.0, 0.99, 0.98),
        condition_features={"temperature_c": 28.0, "mean_soc": 0.5},
        cutoff_cycle=20,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )
    second = _predictor().fit(_training_samples(), split_manifest=_split()).predict(
        dataset_id="MATR",
        cell_id="cell-g",
        observed_cycles=(0, 10, 20),
        observed_soh=(1.0, 0.99, 0.98),
        condition_features={"temperature_c": 28.0, "mean_soc": 0.5},
        cutoff_cycle=20,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        data_version="matr-data-v1",
    )

    assert first.predicted_soh == pytest.approx(second.predicted_soh, abs=1e-7)
