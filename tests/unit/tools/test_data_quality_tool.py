from __future__ import annotations

from datetime import UTC, datetime

from quanxin_life.core import ProvenanceRecord, SourceKind
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.tools import ToolRegistry


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="matr-fixture",
            source_kind=SourceKind.OBSERVED,
            uri="file:///fixtures/matr-b1c0.parquet",
            sha256="b" * 64,
            description="Fixture records for deterministic quality validation",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _record(sample_index: int, time_s: float) -> CycleRecord:
    return CycleRecord(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cycle_index=0,
        sample_index=sample_index,
        time_s=time_s,
        voltage_v=3.2,
        current_a=-1.0,
        temperature_c=25.0,
    )


def test_registered_data_quality_tool_returns_auditable_quality_result() -> None:
    from quanxin_life.tools.data_quality import (
        ValidateBatteryDataToolInput,
        register_validate_battery_data_tool,
    )

    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    tool_input = ValidateBatteryDataToolInput(
        records=(_record(0, 0.0), _record(1, 1.0)),
        data_version="matr-fixture-v1",
        feature_version="raw-cycle-v1",
        provenance=_provenance(),
        validated_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    result = registry.execute("validate_battery_data", tool_input)

    assert result.tool_name == "validate_battery_data"
    assert result.tool_version == "data-quality-tool-v1"
    assert result.model_version == "data-quality-rule-engine-v1"
    assert result.data_version == "matr-fixture-v1"
    assert result.feature_version == "raw-cycle-v1"
    assert result.values["dataset_id"] == "MATR"
    assert result.values["blocked"] is False
    assert result.values["quality_score"] == 1.0
    assert result.values["issue_count"] == 0
    assert result.provenance == list(_provenance())


def test_data_quality_tool_exposes_empty_input_as_blocking_evidence() -> None:
    from quanxin_life.tools.data_quality import (
        ValidateBatteryDataToolInput,
        execute_validate_battery_data_tool,
    )

    result = execute_validate_battery_data_tool(
        ValidateBatteryDataToolInput(
            records=(),
            data_version="matr-fixture-v1",
            feature_version="raw-cycle-v1",
            provenance=_provenance(),
            validated_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
    )

    assert result.values["blocked"] is True
    assert result.values["issues"][0]["code"] == "EMPTY_CELL"
    assert "EMPTY_CELL" in result.warnings
