from __future__ import annotations

import pytest
from pydantic import ValidationError

from quanxin_life.data.canonical import CanonicalNumericValue


def test_numeric_value_binds_physical_provenance() -> None:
    value = CanonicalNumericValue(
        record_id="cell-1-cycle-1-capacity",
        table_type="cell_cycle_telemetry",
        quantity_name="discharge_capacity",
        value=1.05,
        unit="Ah",
        quality_status="VALID",
        source_file="data/raw/MATR/v1/batch.mat",
        source_sha256="a" * 64,
        adapter_version="matr-canonical-v1",
    )

    assert value.source_sha256 == "a" * 64
    assert value.unit.value == "Ah"


@pytest.mark.parametrize(
    ("field", "value"),
    [("table_type", "unknown_table"), ("unit", "mystery_unit")],
)
def test_unknown_table_types_and_units_are_rejected(field: str, value: str) -> None:
    payload = {
        "record_id": "r1",
        "table_type": "condition_observations",
        "quantity_name": "relative_capacity",
        "value": 0.9,
        "unit": "ratio",
        "quality_status": "VALID",
        "source_file": "data/raw/NAUMANN_CYCLE/v1/file.mat",
        "source_sha256": "a" * 64,
        "adapter_version": "naumann-v1",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        CanonicalNumericValue.model_validate(payload)
