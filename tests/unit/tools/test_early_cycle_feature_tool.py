"""Contracts for trusted early-cycle trajectory evidence extraction."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.tools.registry import StandardToolName, ToolRegistry


def _metadata(*, reference_capacity_ah: float | None = 10.0) -> CellMetadata:
    return CellMetadata(
        dataset_id="synthetic-lfp",
        cell_id="cell-feature-tool-01",
        chemistry="LFP/graphite",
        nominal_capacity_ah=10.0,
        reference_capacity_ah=reference_capacity_ah,
        source_uri="trusted-store://features/cell-feature-tool-01",
        source_sha256=sha256_canonical({"fixture": "cell-feature-tool-01"}),
        schema_version="cell-metadata-v1",
    )


def _records() -> tuple[CycleRecord, ...]:
    return (
        CycleRecord(
            dataset_id="synthetic-lfp",
            cell_id="cell-feature-tool-01",
            cycle_index=0,
            sample_index=0,
            time_s=0.0,
            voltage_v=3.1,
            current_a=-1.0,
            temperature_c=25.0,
            discharge_capacity_ah=10.0,
            diagnostic=True,
        ),
        CycleRecord(
            dataset_id="synthetic-lfp",
            cell_id="cell-feature-tool-01",
            cycle_index=10,
            sample_index=0,
            time_s=10.0,
            voltage_v=3.1,
            current_a=-1.0,
            temperature_c=25.5,
            discharge_capacity_ah=9.9,
            diagnostic=True,
        ),
        CycleRecord(
            dataset_id="synthetic-lfp",
            cell_id="cell-feature-tool-01",
            cycle_index=20,
            sample_index=0,
            time_s=20.0,
            voltage_v=3.1,
            current_a=-1.0,
            temperature_c=26.0,
            discharge_capacity_ah=9.8,
            diagnostic=True,
        ),
    )


def _provenance(*, source_kind: SourceKind = SourceKind.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="synthetic-feature-records",
        source_kind=source_kind,
        uri="trusted-store://features/synthetic-feature-records",
        sha256=sha256_canonical({"fixture": "feature-records"}),
        description="Verified canonical early-cycle records used only in tool-contract tests",
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def _batch(*, reference_capacity_ah: float | None = 10.0):
    from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

    return VerifiedEarlyCycleBatch(
        record_batch_id="trusted-record-batch-20260714",
        records=_records(),
        metadata=_metadata(reference_capacity_ah=reference_capacity_ah),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="synthetic-data-v1",
        split_version="synthetic-split-v1",
        source_manifest_hash=sha256_canonical({"manifest": "trusted-feature-batch"}),
        provenance=(_provenance(),),
    )


class _Resolver:
    def __init__(self, batch):
        self.batch = batch
        self.calls: list[str] = []

    def resolve_verified_early_cycle_batch(self, record_batch_id: str):
        self.calls.append(record_batch_id)
        if record_batch_id != self.batch.record_batch_id:
            raise ValueError("trusted early-cycle record batch was not found")
        return self.batch


def _input():
    from quanxin_life.tools.early_cycle_features import ExtractEarlyCycleFeaturesToolInput

    return ExtractEarlyCycleFeaturesToolInput(record_batch_id="trusted-record-batch-20260714")


def test_feature_tool_resolves_trusted_batch_and_derives_cutoff_safe_standard_evidence() -> None:
    from quanxin_life.tools.early_cycle_features import register_extract_early_cycle_features_tool

    resolver = _Resolver(_batch())
    registry = ToolRegistry()
    register_extract_early_cycle_features_tool(
        registry,
        resolver=resolver,
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )

    result = registry.execute(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES, _input())

    assert resolver.calls == ["trusted-record-batch-20260714"]
    assert result.values["artifact_type"] == "quanxin_life.early_cycle_trajectory_evidence.v1"
    artifact = result.values["artifact"]
    assert artifact["observed_cycles"] == [0, 10, 20]
    assert artifact["observed_soh"] == pytest.approx([1.0, 0.99, 0.98])
    assert artifact["reference_capacity_method"] == "metadata_reference_capacity"
    assert artifact["feature_version"] == "early-cycle-v1"
    assert all(value is not None for value in artifact["condition_features"].values())
    assert artifact["source_manifest_hash"] == sha256_canonical(
        {"manifest": "trusted-feature-batch"}
    )
    assert result.data_version == artifact["data_version"]
    assert result.feature_version == artifact["feature_version"]
    assert "trajectory_input" not in result.values
    assert result.created_at == datetime(2026, 7, 14, 9, tzinfo=UTC)


def test_feature_tool_derives_reference_capacity_from_trusted_early_diagnostic_records_only(
) -> None:
    from quanxin_life.tools.early_cycle_features import execute_extract_early_cycle_features_tool

    result = execute_extract_early_cycle_features_tool(
        _input(),
        resolver=_Resolver(_batch(reference_capacity_ah=None)),
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )

    artifact = result.values["artifact"]
    assert artifact["reference_capacity_ah"] == pytest.approx(9.9)
    assert artifact["reference_capacity_method"] == "first_valid_diagnostic_median"


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("records", []),
        ("metadata", {}),
        ("provenance", []),
        ("data_version", "caller-controlled-v1"),
        ("feature_config", {"cutoff_cycle": 100}),
        ("observed_soh", [1.0, 0.99, 0.98]),
        ("condition_features", {"temperature_mean_c": 25.0}),
        ("prediction_cycles", [50, 100]),
    ),
)
def test_feature_tool_rejects_caller_raw_data_versions_and_derived_values(
    field_name: str,
    value: object,
) -> None:
    from quanxin_life.tools.early_cycle_features import ExtractEarlyCycleFeaturesToolInput

    payload = _input().model_dump(mode="python")
    payload[field_name] = value

    with pytest.raises(ValueError, match="Extra inputs"):
        ExtractEarlyCycleFeaturesToolInput.model_validate(payload)


def test_feature_tool_revalidates_bypassed_resolver_batch_and_rejects_unknown_batch() -> None:
    from quanxin_life.tools.early_cycle_features import (
        ExtractEarlyCycleFeaturesToolInput,
        VerifiedEarlyCycleBatch,
        execute_extract_early_cycle_features_tool,
    )

    trusted_batch = _batch()
    future_record = trusted_batch.records[-1].model_copy(update={"cycle_index": 21})
    bypassed_batch = VerifiedEarlyCycleBatch.model_construct(
        record_batch_id=trusted_batch.record_batch_id,
        records=(*trusted_batch.records[:-1], future_record),
        metadata=trusted_batch.metadata,
        feature_config=trusted_batch.feature_config,
        data_version=trusted_batch.data_version,
        split_version=trusted_batch.split_version,
        source_manifest_hash=trusted_batch.source_manifest_hash,
        provenance=trusted_batch.provenance,
    )
    with pytest.raises(ValueError, match="after the feature cutoff"):
        execute_extract_early_cycle_features_tool(
            ExtractEarlyCycleFeaturesToolInput(record_batch_id=bypassed_batch.record_batch_id),
            resolver=_Resolver(bypassed_batch),
            clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="not found"):
        execute_extract_early_cycle_features_tool(
            ExtractEarlyCycleFeaturesToolInput(record_batch_id="unknown-trusted-record-batch"),
            resolver=_Resolver(trusted_batch),
            clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
        )


def test_feature_tool_is_not_available_without_a_context_bound_resolver() -> None:
    from quanxin_life.tools.bootstrap import create_available_tool_registry

    schemas = create_available_tool_registry().list_schemas()

    assert StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES not in {
        schema.tool_name for schema in schemas
    }
