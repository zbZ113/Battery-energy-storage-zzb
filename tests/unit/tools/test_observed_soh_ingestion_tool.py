"""Contracts for trusted newly observed SOH evidence ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.tools.registry import StandardToolName, ToolRegistry


def _provenance(*, source_kind: SourceKind = SourceKind.NEWLY_OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="bms-diagnostic-batch-20260714",
        source_kind=source_kind,
        uri="trusted-store://bms/diagnostic-batch-20260714",
        sha256=sha256_canonical({"source": "bms-diagnostic-batch-20260714"}),
        description="Verified diagnostic discharge measurements for tool-contract tests",
        created_at=datetime(2026, 7, 14, 8, 0, tzinfo=UTC),
    )


def _batch(
    *,
    measured_at: tuple[datetime, datetime] | None = None,
    source_kind: SourceKind = SourceKind.NEWLY_OBSERVED,
):
    from quanxin_life.tools.observed_soh_ingestion import (
        VerifiedNewlyObservedMeasurement,
        VerifiedObservationBatch,
    )

    first_time, second_time = measured_at or (
        datetime(2026, 7, 14, 8, 0, tzinfo=UTC),
        datetime(2026, 7, 14, 8, 5, tzinfo=UTC),
    )
    measurements = (
        VerifiedNewlyObservedMeasurement(
            measurement_id=str(uuid4()),
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=50,
            discharge_capacity_ah=9.5,
            reference_capacity_ah=10.0,
            measured_at=first_time,
            source_record_hash=sha256_canonical({"record": "trusted-50"}),
        ),
        VerifiedNewlyObservedMeasurement(
            measurement_id=str(uuid4()),
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=100,
            discharge_capacity_ah=9.0,
            reference_capacity_ah=10.0,
            measured_at=second_time,
            source_record_hash=sha256_canonical({"record": "trusted-100"}),
        ),
    )
    return VerifiedObservationBatch(
        measurement_batch_id="trusted-batch-20260714",
        measurements=measurements,
        reference_capacity_method="metadata_reference_capacity",
        data_version="synthetic-data-v1",
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        provenance=(_provenance(source_kind=source_kind),),
    )


class _Resolver:
    def __init__(self, batch):
        self.batch = batch
        self.calls: list[str] = []

    def resolve_verified_observation_batch(self, measurement_batch_id: str):
        self.calls.append(measurement_batch_id)
        if measurement_batch_id != self.batch.measurement_batch_id:
            raise ValueError("trusted measurement batch was not found")
        return self.batch


def _input():
    from quanxin_life.tools.observed_soh_ingestion import NewlyObservedSOHIngestionInput

    return NewlyObservedSOHIngestionInput(measurement_batch_id="trusted-batch-20260714")


def test_observed_soh_ingestion_rejects_blank_trusted_batch_identifier() -> None:
    from quanxin_life.tools.observed_soh_ingestion import NewlyObservedSOHIngestionInput

    with pytest.raises(ValueError, match="must not be blank"):
        NewlyObservedSOHIngestionInput(measurement_batch_id="   ")


def test_observed_soh_ingestion_resolves_trusted_batch_and_calculates_standard_evidence() -> None:
    from quanxin_life.tools.observed_soh_ingestion import register_ingest_newly_observed_soh_tool

    resolver = _Resolver(_batch())
    registry = ToolRegistry()
    register_ingest_newly_observed_soh_tool(
        registry,
        resolver=resolver,
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )
    tool_input = _input()

    result = registry.execute(StandardToolName.INGEST_NEWLY_OBSERVED_SOH, tool_input)

    assert resolver.calls == ["trusted-batch-20260714"]
    assert result.tool_name == StandardToolName.INGEST_NEWLY_OBSERVED_SOH.value
    assert result.values["artifact_type"] == "quanxin_life.newly_observed_soh_evidence.v1"
    artifact = result.values["artifact"]
    assert artifact["observation_count"] == 2
    assert artifact["observations"][0]["soh"] == 0.95
    assert artifact["observations"][1]["soh"] == 0.9
    assert artifact["observations"][0]["source_kind"] == "NEWLY_OBSERVED"
    assert artifact["observations"][0]["source_record_hash"] == sha256_canonical(
        {"record": "trusted-50"}
    )
    assert result.created_at == datetime(2026, 7, 14, 9, tzinfo=UTC)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("soh", 0.95),
        ("measurements", []),
        ("provenance", []),
        ("global_trajectory", {"cycles": [50, 100]}),
        ("data_version", "caller-controlled-v1"),
    ),
)
def test_observed_soh_ingestion_rejects_caller_measurements_predictions_and_provenance(
    field_name: str,
    value: object,
) -> None:
    from quanxin_life.tools.observed_soh_ingestion import NewlyObservedSOHIngestionInput

    payload = _input().model_dump(mode="python")
    payload[field_name] = value

    with pytest.raises(ValueError, match="Extra inputs"):
        NewlyObservedSOHIngestionInput.model_validate(payload)


def test_observed_soh_ingestion_rejects_untrusted_batch_time_reversal_and_wrong_provenance(
) -> None:
    from quanxin_life.tools.observed_soh_ingestion import (
        NewlyObservedSOHIngestionInput,
        VerifiedObservationBatch,
        execute_ingest_newly_observed_soh_tool,
    )

    with pytest.raises(ValueError, match="not found"):
        execute_ingest_newly_observed_soh_tool(
            NewlyObservedSOHIngestionInput(measurement_batch_id="unknown-trusted-batch"),
            resolver=_Resolver(_batch()),
            clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

    trusted_batch = _batch()
    reversed_batch = VerifiedObservationBatch.model_construct(
        measurement_batch_id=trusted_batch.measurement_batch_id,
        measurements=(
            trusted_batch.measurements[0],
            trusted_batch.measurements[1].model_copy(
                update={"measured_at": datetime(2026, 7, 14, 7, 59, tzinfo=UTC)}
            ),
        ),
        reference_capacity_method=trusted_batch.reference_capacity_method,
        data_version=trusted_batch.data_version,
        feature_version=trusted_batch.feature_version,
        split_version=trusted_batch.split_version,
        provenance=trusted_batch.provenance,
    )
    with pytest.raises(ValueError, match="must not move backward"):
        execute_ingest_newly_observed_soh_tool(
            NewlyObservedSOHIngestionInput(measurement_batch_id=reversed_batch.measurement_batch_id),
            resolver=_Resolver(reversed_batch),
            clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

    wrong_provenance_batch = VerifiedObservationBatch.model_construct(
        measurement_batch_id=trusted_batch.measurement_batch_id,
        measurements=trusted_batch.measurements,
        reference_capacity_method=trusted_batch.reference_capacity_method,
        data_version=trusted_batch.data_version,
        feature_version=trusted_batch.feature_version,
        split_version=trusted_batch.split_version,
        provenance=(_provenance(source_kind=SourceKind.OBSERVED),),
    )
    with pytest.raises(ValueError, match="NEWLY_OBSERVED"):
        execute_ingest_newly_observed_soh_tool(
            NewlyObservedSOHIngestionInput(
                measurement_batch_id=wrong_provenance_batch.measurement_batch_id
            ),
            resolver=_Resolver(wrong_provenance_batch),
            clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
        )


def test_observed_soh_ingestion_allows_fewer_than_three_trusted_measurements() -> None:
    from quanxin_life.tools.observed_soh_ingestion import (
        NewlyObservedSOHIngestionInput,
        VerifiedObservationBatch,
        execute_ingest_newly_observed_soh_tool,
    )

    batch = _batch()
    short_batch = VerifiedObservationBatch(
        measurement_batch_id=batch.measurement_batch_id,
        measurements=batch.measurements[:2],
        reference_capacity_method=batch.reference_capacity_method,
        data_version=batch.data_version,
        feature_version=batch.feature_version,
        split_version=batch.split_version,
        provenance=batch.provenance,
    )
    result = execute_ingest_newly_observed_soh_tool(
        NewlyObservedSOHIngestionInput(measurement_batch_id=short_batch.measurement_batch_id),
        resolver=_Resolver(short_batch),
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )

    assert result.values["artifact"]["observation_count"] == 2
