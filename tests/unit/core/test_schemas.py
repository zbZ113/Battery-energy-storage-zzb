from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.core.enums import PredictionTarget, SourceKind
from quanxin_life.core.schemas import (
    AnalysisState,
    CellMetadata,
    ConformalCalibration,
    CycleLifePrediction,
    LifePrediction,
    LifetimeMetrics,
    PredictionInterval,
    ProvenanceRecord,
    ToolResult,
)


def provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="nasa-pcoe",
        source_kind=SourceKind.OBSERVED,
        uri="https://example.test/B0005.csv",
        sha256="a" * 64,
        description="Observed cycling data",
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
    )


def valid_tool_result(**overrides: object) -> ToolResult:
    data: dict[str, object] = {
        "result_id": str(uuid4()),
        "tool_name": "soh_estimator",
        "tool_version": "1.0.0",
        "input_hash": "b" * 64,
        "values": {"soh": 0.91},
        "provenance": [provenance()],
        "created_at": datetime(2026, 7, 12, tzinfo=UTC),
    }
    data.update(overrides)
    return ToolResult.model_validate(data)


def test_tool_result_rejects_empty_provenance() -> None:
    with pytest.raises(ValidationError):
        valid_tool_result(provenance=[])


@pytest.mark.parametrize("bad_hash", ["ABC", "A" * 64, "0" * 63, "g" * 64])
def test_tool_result_rejects_invalid_input_hash(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        valid_tool_result(input_hash=bad_hash)


@pytest.mark.parametrize("field", ["values", "uncertainty"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_tool_result_rejects_non_finite_numbers(field: str, non_finite: float) -> None:
    with pytest.raises(ValidationError):
        valid_tool_result(**{field: {"nested": [non_finite]}})


@pytest.mark.parametrize("capacity", [0, -1])
def test_cell_metadata_requires_positive_capacities(capacity: float) -> None:
    data = {
        "dataset_id": "nasa",
        "cell_id": "B0005",
        "chemistry": "Li-ion",
        "nominal_capacity_ah": 2.0,
        "reference_capacity_ah": 2.0,
        "source_uri": "https://example.test/B0005.csv",
        "source_sha256": "c" * 64,
        "schema_version": "1.0.0",
    }
    data["nominal_capacity_ah"] = capacity

    with pytest.raises(ValidationError):
        CellMetadata.model_validate(data)


def test_reference_capacity_may_be_unknown_during_ingestion() -> None:
    metadata = CellMetadata(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        reference_capacity_ah=None,
        source_uri="https://data.matr.io/1/",
        source_sha256="c" * 64,
        schema_version="1.0.0",
    )

    assert metadata.reference_capacity_ah is None


@pytest.mark.parametrize("capacity", [0, -1])
def test_reference_capacity_is_positive_when_known(capacity: float) -> None:
    with pytest.raises(ValidationError):
        CellMetadata(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            reference_capacity_ah=capacity,
            source_uri="https://data.matr.io/1/",
            source_sha256="c" * 64,
            schema_version="1.0.0",
        )


def test_cell_metadata_ingestion_parameters_must_be_json_safe() -> None:
    with pytest.raises(ValidationError):
        CellMetadata(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="https://data.matr.io/1/",
            source_sha256="c" * 64,
            schema_version="1.0.0",
            adapter_version="matr-hdf5-v1.0.0",
            ingestion_parameters={"invalid": object()},
        )


def test_life_prediction_requires_explicit_eol80_metadata() -> None:
    prediction = LifePrediction(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cutoff_cycle=100,
        target=PredictionTarget.EOL80_CYCLE,
        predicted_eol_cycle=845.5,
        observed_eol_cycle=None,
        right_censored=True,
        feature_version="early-cycle-v1",
        split_version="matr-v1",
        model_version="dummy-v1",
        data_version="matr-v1",
    )

    assert prediction.derived_rul == pytest.approx(745.5)
    assert prediction.observed_eol_cycle is None
    assert prediction.right_censored is True


def test_cycle_life_prediction_preserves_matr_official_target_semantics() -> None:
    prediction = CycleLifePrediction(
        dataset_id="MATR",
        cell_id="MATR_b3c0",
        cutoff_cycle=150,
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        predicted_cycle=1000.0,
        observed_cycle=1009,
        right_censored=False,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        model_version="xgboost-v1",
        data_version="matr-2018-04-12-v1",
    )

    assert prediction.target is PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert prediction.derived_remaining_cycles == pytest.approx(850.0)


def test_life_prediction_rejects_matr_official_label_as_eol80() -> None:
    with pytest.raises(ValidationError, match="EOL80"):
        LifePrediction(
            dataset_id="MATR",
            cell_id="MATR_b3c0",
            cutoff_cycle=150,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            predicted_eol_cycle=1000.0,
            observed_eol_cycle=1009,
            right_censored=False,
            feature_version="early-cycle-v1",
            split_version="matr-split-v1",
            model_version="xgboost-v1",
            data_version="matr-2018-04-12-v1",
        )


def test_life_prediction_rejects_non_eol_target() -> None:
    with pytest.raises(ValidationError):
        LifePrediction(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            cutoff_cycle=100,
            target="unsupported",
            predicted_eol_cycle=99.0,
            observed_eol_cycle=99,
            right_censored=False,
            feature_version="early-cycle-v1",
            split_version="matr-v1",
            model_version="dummy-v1",
            data_version="matr-v1",
        )


def test_life_prediction_rejects_an_eol_before_the_cutoff() -> None:
    with pytest.raises(ValidationError, match="cannot precede cutoff_cycle"):
        LifePrediction(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            cutoff_cycle=100,
            target=PredictionTarget.EOL80_CYCLE,
            predicted_eol_cycle=99.0,
            observed_eol_cycle=None,
            right_censored=True,
            feature_version="early-cycle-v1",
            split_version="matr-v1",
            model_version="dummy-v1",
            data_version="matr-v1",
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_life_prediction_rejects_non_finite_eol_prediction(non_finite: float) -> None:
    with pytest.raises(ValidationError):
        LifePrediction(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            cutoff_cycle=100,
            target=PredictionTarget.EOL80_CYCLE,
            predicted_eol_cycle=non_finite,
            observed_eol_cycle=None,
            right_censored=True,
            feature_version="early-cycle-v1",
            split_version="matr-v1",
            model_version="dummy-v1",
            data_version="matr-v1",
        )


def test_lifetime_metrics_rejects_non_finite_values_and_empty_evaluation() -> None:
    with pytest.raises(ValidationError):
        LifetimeMetrics(
            evaluated_cell_count=0,
            mae_cycle=float("inf"),
            rmse_cycle=10.0,
            mape_percent=2.0,
            r2=None,
        )


def test_conformal_contract_requires_finite_quantile_and_ordered_interval() -> None:
    calibration = ConformalCalibration(
        target=PredictionTarget.EOL80_CYCLE,
        alpha=0.1,
        residual_quantile_cycle=12.5,
        calibration_cell_count=10,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        model_version="xgboost-eol80-v1",
        data_version="matr-data-v1",
    )
    interval = PredictionInterval(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cutoff_cycle=100,
        target=PredictionTarget.EOL80_CYCLE,
        point_prediction_cycle=250.0,
        lower_eol_cycle=237.5,
        upper_eol_cycle=262.5,
        calibration=calibration,
    )

    assert interval.coverage_target == pytest.approx(0.9)
    with pytest.raises(ValidationError):
        PredictionInterval(
            dataset_id="MATR",
            cell_id="MATR_b1c0",
            cutoff_cycle=100,
            target=PredictionTarget.EOL80_CYCLE,
            point_prediction_cycle=250.0,
            lower_eol_cycle=260.0,
            upper_eol_cycle=255.0,
            calibration=calibration,
        )


def test_created_at_is_normalized_to_utc() -> None:
    result = valid_tool_result(created_at=datetime(2026, 7, 12, 8, tzinfo=UTC))

    assert result.created_at.tzinfo is UTC
    assert result.created_at.utcoffset().total_seconds() == 0


def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_tool_result(created_at=datetime(2026, 7, 12, 8))


def test_mutable_defaults_are_isolated() -> None:
    first = AnalysisState(request_id=str(uuid4()), status="pending")
    second = AnalysisState(request_id=str(uuid4()), status="pending")

    first.warnings.append("first-only")

    assert second.warnings == []
