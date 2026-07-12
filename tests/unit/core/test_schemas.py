from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.core.enums import SourceKind
from quanxin_life.core.schemas import (
    AnalysisState,
    CellMetadata,
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
