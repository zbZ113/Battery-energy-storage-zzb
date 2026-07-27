from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.application.invocation_context import (
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    CellMetadata,
    ProvenanceRecord,
    SourceKind,
    UserRole,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.features.multichannel_cycle import (
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)
from quanxin_life.tools.advanced_input import (
    ADVANCED_INPUT_EVIDENCE_TYPE,
    ADVANCED_INPUT_TRANSFORM_VERSION,
    PREPARE_ADVANCED_INPUT_TOOL_VERSION,
    AdvancedInputEvidence,
    PrepareAdvancedInputToolInput,
    decode_advanced_input_result,
    execute_prepare_advanced_input_tool,
    register_project_prepare_advanced_input_tool,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch
from quanxin_life.tools.registry import (
    StandardToolName,
    ToolAuthorizationError,
    ToolExecutionScope,
    ToolRegistry,
)


def _records() -> tuple[CycleRecord, ...]:
    records: list[CycleRecord] = []
    for cycle in range(21):
        for sample, current, capacity, voltage in (
            (0, 1.0, 0.0, 3.0),
            (1, 1.1, 0.5, 3.5),
            (2, 1.2, 1.0, 4.0),
            (3, -0.8, 0.0, 4.1),
            (4, -0.9, 0.5, 3.6),
            (5, -1.0, 1.0, 3.1),
        ):
            records.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id="cell-advanced-input",
                    cycle_index=cycle,
                    sample_index=sample,
                    time_s=float(sample),
                    voltage_v=voltage,
                    current_a=current,
                    temperature_c=25.0,
                    charge_capacity_ah=capacity if current > 0 else None,
                    discharge_capacity_ah=capacity if current < 0 else None,
                    diagnostic=cycle == 20,
                )
            )
    return tuple(records)


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="advanced-input-records",
        source_kind=SourceKind.OBSERVED,
        uri="trusted-store://advanced-input/records",
        sha256="a" * 64,
        description="Verified cutoff-bounded canonical records",
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )


def _batch() -> VerifiedEarlyCycleBatch:
    return VerifiedEarlyCycleBatch(
        record_batch_id=str(uuid4()),
        records=_records(),
        metadata=CellMetadata(
            dataset_id="MATR",
            cell_id="cell-advanced-input",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            reference_capacity_ah=1.0,
            source_uri="trusted-store://advanced-input/metadata",
            source_sha256="b" * 64,
            schema_version="cell-metadata-v1",
            official_life_label=900,
            official_life_label_name="MATR_cycle_life",
        ),
        feature_config=EarlyCycleFeatureConfig(
            cutoff_cycle=20,
            feature_version="cyclepatch-multichannel-v1",
        ),
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        source_manifest_hash="c" * 64,
        provenance=(_provenance(),),
    )


def _context() -> VerifiedProjectInvocationContext:
    return VerifiedProjectInvocationContext(
        project_id="project-1",
        actor_user_id="user-1",
        actor_session_id="session-1",
        actor_role=UserRole.ADMIN,
        invocation_source=ProjectInvocationSource.HTTP,
        _authorization_tag="d" * 64,
    )


class _BatchResolver:
    def __init__(self, batch: VerifiedEarlyCycleBatch) -> None:
        self.batch = batch
        self.calls: list[tuple[str, str]] = []

    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        self.calls.append((context.project_id, record_batch_id))
        return self.batch


class _ContextValidator:
    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        return context


def test_project_advanced_input_returns_only_audited_sequence_evidence() -> None:
    batch = _batch()
    transform_config = MultichannelCycleConfig(
        cutoff_cycle=20,
        feature_version="cyclepatch-multichannel-v1",
    )
    raw = build_early_cycle_sequence(
        batch.records,
        config=transform_config,
        data_version=batch.data_version,
    )
    batch_resolver = _BatchResolver(batch)
    input_value = PrepareAdvancedInputToolInput(
        record_batch_id=batch.record_batch_id,
    )

    result = execute_prepare_advanced_input_tool(
        input_value,
        context=_context(),
        batch_resolver=batch_resolver,
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )

    assert result.tool_name == StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value
    assert result.tool_version == PREPARE_ADVANCED_INPUT_TOOL_VERSION
    assert result.model_version == ADVANCED_INPUT_TRANSFORM_VERSION
    assert result.data_version == batch.data_version
    assert result.feature_version == transform_config.feature_version
    assert result.input_hash == sha256_canonical(input_value.model_dump(mode="json"))
    assert set(result.values) == {"artifact_type", "artifact"}
    assert result.values["artifact_type"] == ADVANCED_INPUT_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert set(artifact) == {
        "cell_id",
        "condition_names",
        "cutoff_cycle",
        "dataset_id",
        "normalization_version",
        "phase_names",
        "raw_sequence_input_sha256",
        "record_batch_id",
        "source_manifest_hash",
        "split_version",
        "transform_config_sha256",
        "variable_names",
    }
    assert artifact["record_batch_id"] == batch.record_batch_id
    assert artifact["raw_sequence_input_sha256"] == raw.input_hash
    assert artifact["transform_config_sha256"] == sha256_canonical(
        asdict(transform_config)
    )
    assert artifact["normalization_version"] == "none"
    assert "values" not in artifact
    assert "prediction" not in repr(result.values).casefold()
    assert "official_life_label" not in repr(result.values)
    assert "model_artifact" not in repr(result.values)
    assert result.uncertainty is None
    assert result.warnings == []
    assert batch_resolver.calls == [("project-1", batch.record_batch_id)]


def test_advanced_input_registration_is_project_only() -> None:
    batch = _batch()
    registry = ToolRegistry(project_context_validator=_ContextValidator())
    register_project_prepare_advanced_input_tool(
        registry,
        batch_resolver=_BatchResolver(batch),
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )

    assert registry.list_schemas() == ()
    assert [
        item.tool_name
        for item in registry.list_schemas(
            execution_scope=ToolExecutionScope.PROJECT
        )
    ] == [StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES]
    with pytest.raises(ToolAuthorizationError, match="project-scoped"):
        registry.execute(
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            {"record_batch_id": batch.record_batch_id},
        )


def test_advanced_input_hash_changes_with_verified_observation() -> None:
    original = _batch()
    changed_records = list(original.records)
    changed_records[0] = changed_records[0].model_copy(
        update={"voltage_v": changed_records[0].voltage_v + 0.01}
    )
    changed = original.model_copy(update={"records": tuple(changed_records)})
    input_value = PrepareAdvancedInputToolInput(
        record_batch_id=original.record_batch_id
    )

    first = execute_prepare_advanced_input_tool(
        input_value,
        context=_context(),
        batch_resolver=_BatchResolver(original),
    )
    second = execute_prepare_advanced_input_tool(
        input_value,
        context=_context(),
        batch_resolver=_BatchResolver(changed),
    )

    assert (
        first.values["artifact"]["raw_sequence_input_sha256"]
        != second.values["artifact"]["raw_sequence_input_sha256"]
    )


def test_advanced_input_evidence_hashes_are_stable_for_same_verified_batch() -> None:
    batch = _batch()
    input_value = PrepareAdvancedInputToolInput(record_batch_id=batch.record_batch_id)

    first = execute_prepare_advanced_input_tool(
        input_value,
        context=_context(),
        batch_resolver=_BatchResolver(batch),
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )
    second = execute_prepare_advanced_input_tool(
        input_value,
        context=_context(),
        batch_resolver=_BatchResolver(batch),
        clock=lambda: datetime(2026, 7, 26, 11, 0, tzinfo=UTC),
    )

    assert first.result_id != second.result_id
    assert first.created_at != second.created_at
    assert first.input_hash == second.input_hash
    assert first.tool_version == second.tool_version
    assert first.model_version == second.model_version
    assert first.data_version == second.data_version
    assert first.feature_version == second.feature_version
    assert first.provenance == second.provenance
    assert first.values == second.values


def test_advanced_input_rejects_non_uuid_or_mismatched_binding() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="server-issued UUID"):
        PrepareAdvancedInputToolInput(record_batch_id="client-selected-batch")

    input_value = PrepareAdvancedInputToolInput(record_batch_id=str(uuid4()))
    with pytest.raises(ValueError, match="mismatched binding"):
        execute_prepare_advanced_input_tool(
            input_value,
            context=_context(),
            batch_resolver=_BatchResolver(batch),
        )


def test_advanced_input_rejects_non_matr_verified_batch() -> None:
    batch = _batch()
    other_dataset = "OTHER"
    records = tuple(
        record.model_copy(update={"dataset_id": other_dataset})
        for record in batch.records
    )
    metadata = batch.metadata.model_copy(update={"dataset_id": other_dataset})
    non_matr = VerifiedEarlyCycleBatch.model_validate(
        batch.model_copy(
            update={"records": records, "metadata": metadata}
        ).model_dump(mode="json")
    )

    with pytest.raises(ValueError, match="requires MATR data"):
        execute_prepare_advanced_input_tool(
            PrepareAdvancedInputToolInput(
                record_batch_id=non_matr.record_batch_id,
            ),
            context=_context(),
            batch_resolver=_BatchResolver(non_matr),
        )


def test_advanced_input_export_preserves_cold_api_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import quanxin_life.api.service; "
                "blocked = {'numpy', 'scipy', 'torch'}; "
                "loaded = sorted(name for name in blocked if name in sys.modules); "
                "assert not loaded, loaded; "
                "print('API_IMPORT_OK')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "API_IMPORT_OK"


def test_advanced_input_result_decodes_as_strict_reusable_evidence() -> None:
    batch = _batch()
    result = execute_prepare_advanced_input_tool(
        PrepareAdvancedInputToolInput(record_batch_id=batch.record_batch_id),
        context=_context(),
        batch_resolver=_BatchResolver(batch),
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )

    evidence = decode_advanced_input_result(result)

    assert isinstance(evidence, AdvancedInputEvidence)
    assert evidence.record_batch_id == batch.record_batch_id
    assert evidence.dataset_id == "MATR"
    assert evidence.cell_id == batch.metadata.cell_id
    assert evidence.cutoff_cycle == 20


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("tool_version", "legacy-tool-v1", "tool_version"),
        ("model_version", "legacy-transform-v1", "model_version"),
        ("input_hash", "f" * 64, "input_hash"),
    ],
)
def test_advanced_input_decoder_rejects_outer_contract_drift(
    field_name: str,
    replacement: str,
    message: str,
) -> None:
    batch = _batch()
    result = execute_prepare_advanced_input_tool(
        PrepareAdvancedInputToolInput(record_batch_id=batch.record_batch_id),
        context=_context(),
        batch_resolver=_BatchResolver(batch),
    )

    with pytest.raises(ValueError, match=message):
        decode_advanced_input_result(
            result.model_copy(update={field_name: replacement})
        )
