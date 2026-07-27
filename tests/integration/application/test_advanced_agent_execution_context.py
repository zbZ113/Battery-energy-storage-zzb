from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4, uuid5

import pytest

from quanxin_life.application.advanced_agent_execution_context import (
    AdvancedAgentExecutionContextError,
    AdvancedAgentExecutionContextResolver,
)
from quanxin_life.application.advanced_runtime import (
    RULInferenceAdapter,
    SOHInferenceAdapter,
    VerifiedAdvancedRuntime,
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactKind,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.audit.project_ledger import (
    MaterializedProjectResult,
    ProjectMaterializationCommit,
)
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    AgentIntent,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    AdvancedCalibrationSampleBinding,
    AgentRun,
    Dataset,
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
    ProvenanceRecordRow,
    RecordBatchBinding,
    SessionRecord,
    ToolResultRecord,
    User,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
    ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch
from quanxin_life.tools.registry import StandardToolName

NOW = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
PROJECT_ID = "1bf86f22-440c-4742-964a-2dc0ec9a0df6"
USER_ID = "c981caab-a773-4971-9933-8eae49a9a76f"
SESSION_ID = "fa60c5d4-7511-4a22-9f85-9233646899be"
RUN_ID = "06517af2-6eb9-454e-921f-326c5f22908e"
DATASET_ID = "7ae286c5-d321-4804-9541-a1900e88b706"
RECORD_BATCH_ID = "236992f9-3970-47f3-82bd-d4daf538b523"
TARGET_CELL_ID = "target-cell"
CUTOFF = 100
DATA_VERSION = "matr-three-batch-v1"
FEATURE_VERSION = "cyclepatch-multichannel-v1"
SPLIT_VERSION = "matr-cell-split-v1"
SOURCE_REGISTRATION_ID = "matr-three-batch-final-v1"
SOURCE_IDENTITY_SHA256 = "1" * 64
NORMALIZATION_SHA256 = "2" * 64


@dataclass(frozen=True, slots=True)
class _Route:
    task: AdvancedModelTask
    role: AdvancedModelRouteRole
    materialization_id: str
    artifact_id: str
    decision_event_id: str
    artifact_sha256: str
    ledger_sha256: str
    model_version: str
    artifact_kind: DeepArtifactKind
    output_target: AdvancedOutputTarget


RUL_ROUTE = _Route(
    task=AdvancedModelTask.RUL,
    role=AdvancedModelRouteRole.COVERAGE,
    materialization_id="e58c7b96-780f-49c7-a29a-45fe317277a9",
    artifact_id="e4f76193-62e7-4cc7-87cc-bf88162f49a4",
    decision_event_id="18721b71-0b7b-493f-aa27-b501a3f8b20e",
    artifact_sha256="3" * 64,
    ledger_sha256="4" * 64,
    model_version="cyclepatch-direct-v1",
    artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
    output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
)
SOH_ROUTE = _Route(
    task=AdvancedModelTask.SOH,
    role=AdvancedModelRouteRole.MEAN_ACCURACY,
    materialization_id="3ca26b89-159e-4adb-a704-293450089602",
    artifact_id="ef9ca4c0-cf41-4242-b235-10c3c896eebb",
    decision_event_id="01d7fb60-af6d-49ef-a645-3d77f61a0195",
    artifact_sha256="5" * 64,
    ledger_sha256="6" * 64,
    model_version="current-hybrid-v1",
    artifact_kind=DeepArtifactKind.CURRENT_HYBRID,
    output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
)


class _RecordBatchResolver:
    def __init__(self, batch: VerifiedEarlyCycleBatch) -> None:
        self.batch = batch
        self.calls: list[tuple[str, str]] = []

    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        self.calls.append((context.project_id, record_batch_id))
        if context.project_id != PROJECT_ID or record_batch_id != RECORD_BATCH_ID:
            raise LookupError("record batch is not visible")
        return self.batch


class _RuntimeResolver:
    def __init__(self, runtimes: tuple[VerifiedAdvancedRuntime, ...]) -> None:
        self.runtimes = {
            (runtime.task, runtime.cutoff_cycle, runtime.role): runtime
            for runtime in runtimes
        }
        self.calls: list[
            tuple[AdvancedModelTask, int, AdvancedModelRouteRole]
        ] = []

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedAdvancedRuntime:
        self.calls.append((task, cutoff_cycle, role))
        if context.project_id != PROJECT_ID:
            raise LookupError("runtime is not visible")
        return self.runtimes[(task, cutoff_cycle, role)]


class _Delegate:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str, str]] = []

    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        del run_id, project_id, dataset_id
        raise AssertionError("Advanced resolver must verify the bound record batch")

    def resolve_context(
        self,
        *,
        run_id: str,
        project_id: str,
        reference: str,
    ) -> object:
        self.context_calls.append((run_id, project_id, reference))
        if reference == "context.rul_task":
            return AdvancedModelTask.RUL.value
        raise KeyError(reference)


@dataclass(slots=True)
class _Fixture:
    session_factory: SessionFactory
    record_batch_resolver: _RecordBatchResolver
    runtime_resolver: _RuntimeResolver
    delegate: _Delegate
    resolver: AdvancedAgentExecutionContextResolver
    results: dict[AdvancedModelTask, tuple[str, ...]]


@pytest.fixture
def fixture(tmp_path) -> _Fixture:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'agent-context.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    batch = _batch(TARGET_CELL_ID)
    record_batch_resolver = _RecordBatchResolver(batch)
    runtimes = (_runtime(RUL_ROUTE), _runtime(SOH_ROUTE))
    runtime_resolver = _RuntimeResolver(runtimes)
    delegate = _Delegate()
    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    _seed_base(session_factory)
    _seed_route_and_materialization(session_factory, RUL_ROUTE)
    _seed_route_and_materialization(session_factory, SOH_ROUTE)
    ledger = SqlProjectAuditLedger(
        session_factory,
        context_validator=context_service,
        clock=lambda: NOW,
    )
    results = {
        AdvancedModelTask.RUL: ledger.commit_advanced_calibration_materialization(
            _commit(RUL_ROUTE, _rul_samples(RUL_ROUTE))
        ).result_ids,
        AdvancedModelTask.SOH: ledger.commit_advanced_calibration_materialization(
            _commit(SOH_ROUTE, _soh_samples(SOH_ROUTE))
        ).result_ids,
    }
    resolver = AdvancedAgentExecutionContextResolver(
        session_factory,
        context_service=context_service,
        record_batch_resolver=record_batch_resolver,
        runtime_resolver=runtime_resolver,
        delegate=delegate,
    )
    return _Fixture(
        session_factory=session_factory,
        record_batch_resolver=record_batch_resolver,
        runtime_resolver=runtime_resolver,
        delegate=delegate,
        resolver=resolver,
        results=results,
    )


def test_resolves_exact_ready_rul_and_soh_ids_from_bound_target_batch(
    fixture: _Fixture,
) -> None:
    dataset_artifact = fixture.resolver.resolve_dataset_artifact(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        dataset_id=RECORD_BATCH_ID,
    )
    rul_ids = fixture.resolver.resolve_context(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        reference="context.rul_calibration_sample_result_ids",
    )
    soh_ids = fixture.resolver.resolve_context(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        reference="context.soh_calibration_sample_result_ids",
    )

    assert dataset_artifact == RECORD_BATCH_ID
    assert rul_ids == fixture.results[AdvancedModelTask.RUL]
    assert soh_ids == fixture.results[AdvancedModelTask.SOH]
    assert fixture.record_batch_resolver.calls == [
        (PROJECT_ID, RECORD_BATCH_ID),
        (PROJECT_ID, RECORD_BATCH_ID),
        (PROJECT_ID, RECORD_BATCH_ID),
    ]
    assert fixture.runtime_resolver.calls == [
        (AdvancedModelTask.RUL, CUTOFF, AdvancedModelRouteRole.COVERAGE),
        (AdvancedModelTask.SOH, CUTOFF, AdvancedModelRouteRole.MEAN_ACCURACY),
    ]


def test_returns_result_ids_by_persisted_ordinal_not_row_order(
    fixture: _Fixture,
) -> None:
    expected = fixture.results[AdvancedModelTask.RUL]
    with fixture.session_factory.begin() as session:
        bindings = list(
            session.query(AdvancedCalibrationSampleBinding)
            .filter_by(materialization_id=RUL_ROUTE.materialization_id)
            .all()
        )
        bindings[0].created_at = NOW + timedelta(minutes=2)
        bindings[1].created_at = NOW - timedelta(minutes=2)

    resolved = fixture.resolver.resolve_context(
        run_id=RUN_ID,
        project_id=PROJECT_ID,
        reference="context.rul_calibration_sample_result_ids",
    )

    assert resolved == expected


def test_populates_only_server_owned_calibration_references(
    fixture: _Fixture,
) -> None:
    assert (
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_task",
        )
        == AdvancedModelTask.RUL.value
    )
    assert fixture.delegate.context_calls == [
        (RUN_ID, PROJECT_ID, "context.rul_task")
    ]
    assert not fixture.delegate.context_calls or all(
        reference
        not in {
            "context.rul_calibration_sample_result_ids",
            "context.soh_calibration_sample_result_ids",
        }
        for _, _, reference in fixture.delegate.context_calls
    )


def test_rejects_target_cell_that_is_part_of_calibration_cohort(
    fixture: _Fixture,
) -> None:
    _set_target_cell(fixture, "rul-cal-0")

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"target|cohort"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        )


def test_rejects_missing_exact_ready_materialization(fixture: _Fixture) -> None:
    with fixture.session_factory.begin() as session:
        row = session.get(
            AdvancedCalibrationMaterialization,
            RUL_ROUTE.materialization_id,
        )
        assert row is not None
        row.decision_event_id = SOH_ROUTE.decision_event_id

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"READY|materialization"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        )


def test_rejects_stale_materialization(fixture: _Fixture) -> None:
    with fixture.session_factory.begin() as session:
        row = session.get(
            AdvancedCalibrationMaterialization,
            RUL_ROUTE.materialization_id,
        )
        assert row is not None
        row.status = AdvancedCalibrationMaterializationStatus.STALE.value
        row.failure_code = "ACTIVE_ROUTE_CHANGED"

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"READY|STALE"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        )


def test_rejects_missing_sample_binding(fixture: _Fixture) -> None:
    with fixture.session_factory.begin() as session:
        binding = (
            session.query(AdvancedCalibrationSampleBinding)
            .filter_by(materialization_id=RUL_ROUTE.materialization_id, ordinal=1)
            .one()
        )
        session.delete(binding)

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"sample|count|manifest"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        )


def test_rejects_duplicate_sample_provenance(fixture: _Fixture) -> None:
    result_id = fixture.results[AdvancedModelTask.RUL][0]
    with fixture.session_factory.begin() as session:
        original = (
            session.query(ProvenanceRecordRow)
            .filter_by(tool_result_id=result_id)
            .order_by(ProvenanceRecordRow.source_kind)
            .first()
        )
        assert original is not None
        session.add(
            ProvenanceRecordRow(
                id=str(uuid4()),
                tool_result_id=result_id,
                source_id=original.source_id,
                source_kind=original.source_kind,
                uri=original.uri,
                sha256=original.sha256,
                description=original.description,
                created_at=original.created_at,
            )
        )

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"provenance|content"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        )


def test_rejects_tampered_tool_result_or_binding(fixture: _Fixture) -> None:
    result_id = fixture.results[AdvancedModelTask.SOH][0]
    with fixture.session_factory.begin() as session:
        result = session.get(ToolResultRecord, result_id)
        assert result is not None
        values = dict(result.values_json)
        artifact = dict(values["artifact"])
        artifact["cell_id"] = "tampered-cell"
        values["artifact"] = artifact
        result.values_json = values

    with pytest.raises(AdvancedAgentExecutionContextError, match=r"ToolResult|binding|content"):
        fixture.resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.soh_calibration_sample_result_ids",
        )


def _seed_base(session_factory: SessionFactory) -> None:
    intent = AgentIntent(
        intent_id=str(uuid4()),
        project_id=PROJECT_ID,
        goal="analyze one frozen MATR cell",
        dataset_ids=(RECORD_BATCH_ID,),
        requested_outputs=("advanced_single_cell_analysis",),
        created_at=NOW,
    )
    with session_factory.begin() as session:
        session.add(
            User(
                id=USER_ID,
                username="agent-context@example.test",
                credential_hash="$argon2id$test-only",
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=SESSION_ID,
                user_id=USER_ID,
                token_hash="7" * 64,
                status=SessionStatus.ACTIVE.value,
                created_at=NOW,
                expires_at=NOW + timedelta(hours=2),
            )
        )
        session.add(
            Project(
                id=PROJECT_ID,
                owner_user_id=USER_ID,
                name="Advanced Agent context",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Dataset(
                id=DATASET_ID,
                project_id=PROJECT_ID,
                name="Frozen target cell",
                data_version=DATA_VERSION,
                schema_version="canonical-cycle-v1",
                status="FROZEN",
                manifest_uri="dataset://target/manifest",
                manifest_sha256="8" * 64,
                created_at=NOW,
                frozen_at=NOW,
            )
        )
        session.add(
            RecordBatchBinding(
                id=RECORD_BATCH_ID,
                binding_schema_version="record-batch-binding-v1",
                content_batch_id="sha256:" + "9" * 64,
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID,
                source_manifest_sha256="a" * 64,
                registration_sha256="b" * 64,
                content_dataset_id="MATR",
                dataset_schema_version="canonical-cycle-v1",
                cell_id=TARGET_CELL_ID,
                cutoff_cycle=CUTOFF,
                data_version=DATA_VERSION,
                split_version=SPLIT_VERSION,
                feature_version=FEATURE_VERSION,
                created_by_user_id=USER_ID,
                created_at=NOW,
            )
        )
        session.add(
            AgentRun(
                id=RUN_ID,
                project_id=PROJECT_ID,
                session_id=SESSION_ID,
                created_by_user_id=USER_ID,
                idempotency_key_hash="c" * 64,
                request_hash="d" * 64,
                status="RUNNING",
                planning_mode="FALLBACK",
                intent_json=intent.model_dump(mode="json"),
                plan_json={"schema_version": "test-only-agent-plan"},
                plan_hash="e" * 64,
                execution_plan_hash=None,
                created_at=NOW,
                updated_at=NOW,
            )
        )


def _seed_route_and_materialization(
    session_factory: SessionFactory,
    route: _Route,
) -> None:
    with session_factory.begin() as session:
        session.add(
            ModelArtifact(
                id=route.artifact_id,
                project_id=PROJECT_ID,
                artifact_format="safetensors",
                object_uri=f"artifact://advanced-model/{route.artifact_id}",
                sha256=route.artifact_sha256,
                status="VERIFIED",
                created_at=NOW,
            )
        )
        session.add(
            ModelManifest(
                id=str(uuid4()),
                artifact_id=route.artifact_id,
                model_version=route.model_version,
                manifest_uri=f"artifact://advanced-model/{route.artifact_id}/manifest",
                manifest_sha256=route.artifact_sha256,
                metadata_json={"schema_version": "deep-model-artifact-v2"},
                created_at=NOW,
            )
        )
        session.add(
            ModelRouteActivationEvent(
                id=route.decision_event_id,
                project_id=PROJECT_ID,
                stream_sequence=1,
                task=route.task.value,
                cutoff_cycle=CUTOFF,
                route_role=route.role.value,
                decision_type="ACTIVATE",
                artifact_id=route.artifact_id,
                artifact_sha256=route.artifact_sha256,
                manifest_sha256=route.artifact_sha256,
                deployment_bundle_manifest_sha256="f" * 64,
                route_provenance_sha256="0" * 64,
                rollback_target_event_id=None,
                previous_event_sha256="1" * 64,
                event_sha256=route.ledger_sha256,
                actor_user_id=USER_ID,
                reason="Activate test route",
                idempotency_key_sha256=sha256_canonical(
                    {"activation": route.task.value}
                ),
                request_sha256=sha256_canonical(
                    {"activation_request": route.task.value}
                ),
                created_at=NOW,
            )
        )
        session.add(
            ModelRouteActivationStreamHead(
                project_id=PROJECT_ID,
                task=route.task.value,
                cutoff_cycle=CUTOFF,
                route_role=route.role.value,
                head_event_id=route.decision_event_id,
                head_sequence=1,
                head_event_sha256=route.ledger_sha256,
                updated_at=NOW,
            )
        )
        session.add(
            AdvancedCalibrationMaterialization(
                id=route.materialization_id,
                project_id=PROJECT_ID,
                task=route.task.value,
                cutoff_cycle=CUTOFF,
                route_role=route.role.value,
                status=AdvancedCalibrationMaterializationStatus.RUNNING.value,
                data_version=DATA_VERSION,
                split_version=SPLIT_VERSION,
                feature_version=FEATURE_VERSION,
                artifact_id=route.artifact_id,
                artifact_manifest_sha256=route.artifact_sha256,
                normalization_statistics_sha256=NORMALIZATION_SHA256,
                decision_event_id=route.decision_event_id,
                ledger_sequence_number=1,
                ledger_head_sha256=route.ledger_sha256,
                source_registration_id=SOURCE_REGISTRATION_ID,
                source_identity_sha256=SOURCE_IDENTITY_SHA256,
                sample_manifest_sha256=None,
                sample_count=0,
                idempotency_key_sha256=sha256_canonical(
                    {"route": route.task.value}
                ),
                request_sha256=sha256_canonical(
                    {"request": route.task.value}
                ),
                created_by_user_id=USER_ID,
                created_by_session_id=SESSION_ID,
                created_by_role=UserRole.ADMIN.value,
                created_at=NOW,
                started_at=NOW,
                completed_at=None,
                failure_code=None,
                claim_token_sha256=sha256_canonical(
                    {"claim_token": route.task.value}
                ),
                claim_attempt=1,
                claim_lease_expires_at=NOW + timedelta(minutes=15),
            )
        )


def _runtime(route: _Route) -> VerifiedAdvancedRuntime:
    common = {
        "project_id": PROJECT_ID,
        "task": route.task,
        "cutoff_cycle": CUTOFF,
        "role": route.role,
        "output_target": route.output_target,
        "artifact_kind": route.artifact_kind,
        "dataset_id": "MATR",
        "data_version": DATA_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "normalization_sha256": NORMALIZATION_SHA256,
        "artifact_id": route.artifact_id,
        "artifact_manifest_sha256": route.artifact_sha256,
        "model_version": route.model_version,
        "decision_event_id": route.decision_event_id,
        "ledger_sequence_number": 1,
        "ledger_head_sha256": route.ledger_sha256,
    }
    if route.task is AdvancedModelTask.RUL:
        return VerifiedRULRuntime(
            **common,
            inference=cast(RULInferenceAdapter, object()),
        )
    return VerifiedSOHRuntime(
        **common,
        inference=cast(SOHInferenceAdapter, object()),
    )


def _commit(
    route: _Route,
    samples: tuple[MaterializedProjectResult, ...],
) -> ProjectMaterializationCommit:
    return ProjectMaterializationCommit(
        materialization_id=route.materialization_id,
        project_id=PROJECT_ID,
        task=route.task,
        cutoff_cycle=CUTOFF,
        route_role=route.role,
        data_version=DATA_VERSION,
        split_version=SPLIT_VERSION,
        feature_version=FEATURE_VERSION,
        artifact_id=route.artifact_id,
        artifact_manifest_sha256=route.artifact_sha256,
        model_version=route.model_version,
        normalization_statistics_sha256=NORMALIZATION_SHA256,
        decision_event_id=route.decision_event_id,
        ledger_sequence_number=1,
        ledger_head_sha256=route.ledger_sha256,
        source_registration_id=SOURCE_REGISTRATION_ID,
        source_identity_sha256=SOURCE_IDENTITY_SHA256,
        request_sha256=sha256_canonical({"request": route.task.value}),
        claim_token=route.task.value,
        claim_attempt=1,
        claim_lease_expires_at=NOW + timedelta(minutes=15),
        samples=samples,
    )


def _rul_samples(route: _Route) -> tuple[MaterializedProjectResult, ...]:
    return tuple(
        _sample(
            route,
            ordinal=ordinal,
            cell_id=f"rul-cal-{ordinal}",
            artifact_extra={
                "point_prediction_cycle": 600.0 + ordinal,
                "observed_cycle": 610 + ordinal,
            },
        )
        for ordinal in range(2)
    )


def _soh_samples(route: _Route) -> tuple[MaterializedProjectResult, ...]:
    cycles = list(range(CUTOFF + 1, 501))
    return tuple(
        _sample(
            route,
            ordinal=ordinal,
            cell_id=f"soh-cal-{ordinal}",
            artifact_extra={
                "prediction_cycles": cycles,
                "predicted_soh": [
                    1.0 - index * 0.0005 for index in range(len(cycles))
                ],
                "observed_soh": [
                    0.99 - index * 0.0005 for index in range(len(cycles))
                ],
                "finite_horizon_only": True,
                "horizon_end_cycle": 500,
            },
        )
        for ordinal in range(2)
    )


def _sample(
    route: _Route,
    *,
    ordinal: int,
    cell_id: str,
    artifact_extra: dict[str, object],
) -> MaterializedProjectResult:
    artifact = {
        "task": route.task.value,
        "route_role": route.role.value,
        "output_target": route.output_target.value,
        "artifact_kind": _conformal_artifact_kind(route.artifact_kind),
        "artifact_id": route.artifact_id,
        "artifact_manifest_sha256": route.artifact_sha256,
        "model_version": route.model_version,
        "dataset_id": "MATR",
        "cutoff_cycle": CUTOFF,
        "data_version": DATA_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "normalization_statistics_sha256": NORMALIZATION_SHA256,
        "decision_event_id": route.decision_event_id,
        "ledger_sequence_number": 1,
        "ledger_head_sha256": route.ledger_sha256,
        "materialization_id": route.materialization_id,
        "source_registration_id": SOURCE_REGISTRATION_ID,
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "split_partition": "calibration",
        "cell_id": cell_id,
        **artifact_extra,
    }
    result_id = str(
        uuid5(
            UUID(route.materialization_id),
            sha256_canonical(
                {"ordinal": ordinal, "cell_id": cell_id, "artifact": artifact}
            ),
        )
    )
    tool_name = (
        StandardToolName.PREDICT_CYCLE_LIFE
        if route.task is AdvancedModelTask.RUL
        else StandardToolName.PREDICT_SOH_TRAJECTORY
    )
    evidence_type = (
        ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE
        if route.task is AdvancedModelTask.RUL
        else ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE
    )
    result = ToolResult(
        result_id=result_id,
        tool_name=tool_name.value,
        tool_version="advanced-calibration-sample-producer-v1",
        model_version=route.model_version,
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        input_hash=sha256_canonical(artifact),
        values={"artifact_type": evidence_type, "artifact": artifact},
        uncertainty=None,
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id=(
                    f"advanced-calibration-source-{SOURCE_REGISTRATION_ID}"
                ),
                source_kind=SourceKind.OBSERVED,
                uri=(
                    f"calibration-source://{SOURCE_REGISTRATION_ID}/"
                    f"{SOURCE_IDENTITY_SHA256}"
                ),
                sha256=SOURCE_IDENTITY_SHA256,
                description="Verified server-owned MATR calibration supervision",
                created_at=NOW,
            ),
            ProvenanceRecord(
                source_id=f"advanced-model-{route.artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{route.artifact_id}",
                sha256=route.artifact_sha256,
                description="Verified active Advanced runtime",
                created_at=NOW,
            ),
        ],
        created_at=NOW,
    )
    return MaterializedProjectResult(
        ordinal=ordinal,
        cell_id=cell_id,
        result=result,
    )


def _conformal_artifact_kind(artifact_kind: DeepArtifactKind) -> str:
    return {
        DeepArtifactKind.CYCLEPATCH_DIRECT: "cyclepatch_direct",
        DeepArtifactKind.CYCLEPATCH_BATLINET: "cyclepatch_batlinet",
        DeepArtifactKind.HYBRIDPATCH_V2: "hybridpatch_v2",
        DeepArtifactKind.CURRENT_HYBRID: "current_hybrid",
    }[artifact_kind]


def _batch(cell_id: str) -> VerifiedEarlyCycleBatch:
    records = tuple(
        CycleRecord(
            dataset_id="MATR",
            cell_id=cell_id,
            cycle_index=cycle,
            sample_index=index,
            time_s=float(index),
            voltage_v=3.2,
            current_a=-1.0,
            temperature_c=25.0,
            charge_capacity_ah=1.1,
            discharge_capacity_ah=capacity,
            internal_resistance_ohm=0.01,
            diagnostic=True,
            valid=True,
        )
        for index, (cycle, capacity) in enumerate(((1, 1.1), (CUTOFF, 1.0)))
    )
    return VerifiedEarlyCycleBatch(
        record_batch_id=RECORD_BATCH_ID,
        records=records,
        metadata={
            "dataset_id": "MATR",
            "cell_id": cell_id,
            "chemistry": "LFP/graphite",
            "nominal_capacity_ah": 1.1,
            "source_uri": f"test://target/{cell_id}",
            "source_sha256": "a" * 64,
            "schema_version": "cycle-record-v1",
        },
        feature_config=EarlyCycleFeatureConfig(
            cutoff_cycle=CUTOFF,
            feature_version=FEATURE_VERSION,
        ),
        data_version=DATA_VERSION,
        split_version=SPLIT_VERSION,
        source_manifest_hash="a" * 64,
        provenance=(
            ProvenanceRecord(
                source_id=f"target-{cell_id}",
                source_kind=SourceKind.OBSERVED,
                uri=f"test://target/{cell_id}",
                sha256="a" * 64,
                description="Verified target record batch",
                created_at=NOW,
            ),
        ),
    )


def _set_target_cell(fixture: _Fixture, cell_id: str) -> None:
    fixture.record_batch_resolver.batch = _batch(cell_id)
    with fixture.session_factory.begin() as session:
        binding = session.get(RecordBatchBinding, RECORD_BATCH_ID)
        assert binding is not None
        binding.cell_id = cell_id
