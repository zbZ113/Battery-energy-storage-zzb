from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4, uuid5

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from quanxin_life.audit import AuditLedgerError, SqlProjectAuditLedger
from quanxin_life.audit.project_ledger import (
    MaterializedProjectResult,
    ProjectMaterializationCommit,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    AdvancedCalibrationSampleBinding,
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
    User,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName

NOW = datetime(2026, 7, 27, 4, 0, tzinfo=UTC)
MATERIALIZATION_ID = "1e8a55ad-4fd8-48d8-8dc8-b7f02c459507"
PROJECT_ID = "0166c272-9e52-4f77-882b-031c8557924b"
USER_ID = "3ffbd300-b982-463f-8129-d667293f0b19"
SESSION_ID = "e0b0b2e7-f583-49ff-8641-fd53c2054f65"
ARTIFACT_ID = "40b87568-7c8e-481f-a284-945480d62fbc"
DECISION_EVENT_ID = "cf2632ec-e75f-4ba1-943d-f8b71e34727e"
CLAIM_TOKEN = "calibration-claim-token-v1"
CLAIM_TOKEN_SHA256 = sha256_canonical({"claim_token": CLAIM_TOKEN})
CLAIM_ATTEMPT = 1
CLAIM_LEASE = NOW + timedelta(minutes=15)
REQUEST_SHA256 = "1" * 64
SOURCE_IDENTITY_SHA256 = "2" * 64
ARTIFACT_MANIFEST_SHA256 = "3" * 64
NORMALIZATION_SHA256 = "4" * 64
LEDGER_HEAD_SHA256 = "5" * 64


@dataclass(frozen=True, slots=True)
class _Fixture:
    session_factory: SessionFactory
    ledger: SqlProjectAuditLedger


@pytest.fixture
def fixture(tmp_path: Path) -> _Fixture:
    engine = create_engine_from_config(
        DatabaseConfig(
            url=f"sqlite+pysqlite:///{tmp_path / 'advanced-materialization.sqlite3'}"
        )
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    with session_factory.begin() as session:
        session.add(
            User(
                id=USER_ID,
                username="calibration-admin@example.test",
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
                token_hash="6" * 64,
                status=SessionStatus.ACTIVE.value,
                created_at=NOW,
                expires_at=NOW + timedelta(hours=1),
            )
        )
        session.add(
            Project(
                id=PROJECT_ID,
                owner_user_id=USER_ID,
                name="Advanced calibration integration",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            ModelArtifact(
                id=ARTIFACT_ID,
                project_id=PROJECT_ID,
                artifact_format="safetensors",
                object_uri=f"artifact://advanced-model/{ARTIFACT_ID}",
                sha256=ARTIFACT_MANIFEST_SHA256,
                status="VERIFIED",
                created_at=NOW,
            )
        )
        session.add(
            ModelManifest(
                id=str(uuid4()),
                artifact_id=ARTIFACT_ID,
                model_version="cyclepatch-direct-v1",
                manifest_uri=f"artifact://advanced-model/{ARTIFACT_ID}/manifest",
                manifest_sha256=ARTIFACT_MANIFEST_SHA256,
                metadata_json={"schema_version": "deep-model-artifact-v2"},
                created_at=NOW,
            )
        )
        session.add(
            ModelRouteActivationEvent(
                id=DECISION_EVENT_ID,
                project_id=PROJECT_ID,
                stream_sequence=3,
                task=AdvancedModelTask.RUL.value,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE.value,
                decision_type="ACTIVATE",
                artifact_id=ARTIFACT_ID,
                artifact_sha256=ARTIFACT_MANIFEST_SHA256,
                manifest_sha256=ARTIFACT_MANIFEST_SHA256,
                deployment_bundle_manifest_sha256="7" * 64,
                route_provenance_sha256="8" * 64,
                rollback_target_event_id=None,
                previous_event_sha256="9" * 64,
                event_sha256=LEDGER_HEAD_SHA256,
                actor_user_id=USER_ID,
                reason="Approve calibration runtime",
                idempotency_key_sha256="a" * 64,
                request_sha256="b" * 64,
                created_at=NOW,
            )
        )
        session.add(
            ModelRouteActivationStreamHead(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL.value,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE.value,
                head_event_id=DECISION_EVENT_ID,
                head_sequence=3,
                head_event_sha256=LEDGER_HEAD_SHA256,
                updated_at=NOW,
            )
        )
        session.add(
            AdvancedCalibrationMaterialization(
                id=MATERIALIZATION_ID,
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL.value,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE.value,
                status=AdvancedCalibrationMaterializationStatus.RUNNING.value,
                data_version="matr-three-batch-v1",
                split_version="matr-three-batch-split-v1",
                feature_version="advanced-feature-v1",
                artifact_id=ARTIFACT_ID,
                artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
                normalization_statistics_sha256=NORMALIZATION_SHA256,
                decision_event_id=DECISION_EVENT_ID,
                ledger_sequence_number=3,
                ledger_head_sha256=LEDGER_HEAD_SHA256,
                source_registration_id="matr-three-batch-final-v1",
                source_identity_sha256=SOURCE_IDENTITY_SHA256,
                sample_manifest_sha256=None,
                sample_count=0,
                idempotency_key_sha256="c" * 64,
                request_sha256=REQUEST_SHA256,
                created_by_user_id=USER_ID,
                created_by_session_id=SESSION_ID,
                created_by_role=UserRole.ADMIN.value,
                created_at=NOW,
                started_at=NOW,
                completed_at=None,
                failure_code=None,
                claim_token_sha256=CLAIM_TOKEN_SHA256,
                claim_attempt=CLAIM_ATTEMPT,
                claim_lease_expires_at=CLAIM_LEASE,
            )
        )
    from quanxin_life.application.invocation_context import (
        ProjectInvocationContextService,
    )

    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    principal = AuthPrincipal(
        user_id=USER_ID,
        session_id=SESSION_ID,
        username="calibration-admin@example.test",
        role=UserRole.ADMIN,
        must_change_password=False,
    )
    context_service.resolve_http(principal, PROJECT_ID)
    return _Fixture(
        session_factory=session_factory,
        ledger=SqlProjectAuditLedger(
            session_factory,
            context_validator=context_service,
            clock=lambda: NOW,
        ),
    )


def test_atomically_commits_samples_bindings_manifest_and_ready_state(
    fixture: _Fixture,
) -> None:
    samples = _samples()

    receipt = fixture.ledger.commit_advanced_calibration_materialization(
        _commit(samples)
    )

    assert receipt.materialization_id == MATERIALIZATION_ID
    assert receipt.sample_count == 2
    assert receipt.result_ids == tuple(item.result.result_id for item in samples)
    assert len(receipt.sample_manifest_sha256) == 64
    with fixture.session_factory() as session:
        materialization = session.get(
            AdvancedCalibrationMaterialization,
            MATERIALIZATION_ID,
        )
        assert materialization is not None
        assert materialization.status == "READY"
        assert materialization.sample_count == 2
        assert (
            materialization.sample_manifest_sha256
            == receipt.sample_manifest_sha256
        )
        assert materialization.claim_token_sha256 is None
        assert materialization.claim_lease_expires_at is None
        assert materialization.claim_attempt == CLAIM_ATTEMPT
        assert materialization.completed_at == NOW
        assert _count(session, ToolResultRecord) == 2
        assert _count(session, ProvenanceRecordRow) == 4
        assert _count(session, ProjectToolResultBindingRecord) == 2
        assert _count(session, AdvancedCalibrationSampleBinding) == 2
        bindings = tuple(
            session.scalars(
                select(AdvancedCalibrationSampleBinding).order_by(
                    AdvancedCalibrationSampleBinding.ordinal
                )
            )
        )
        assert tuple(item.ordinal for item in bindings) == (0, 1)
        assert tuple(item.cell_id for item in bindings) == ("cell-a", "cell-b")
        assert all(len(item.sample_sha256) == 64 for item in bindings)


def test_tool_results_are_inserted_before_fk_dependent_sample_bindings(
    fixture: _Fixture,
) -> None:
    engine = fixture.session_factory.kw["bind"]
    insert_statements: list[str] = []

    def record_insert_order(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if statement.lstrip().upper().startswith("INSERT"):
            insert_statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_insert_order)
    try:
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit(_samples())
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_insert_order)

    tool_result_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO tool_results" in statement
    )
    sample_binding_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO advanced_calibration_sample_bindings" in statement
    )
    assert tool_result_position < sample_binding_position


def test_invalid_sample_rolls_back_every_result_and_binding(
    fixture: _Fixture,
) -> None:
    first, second = _samples()
    invalid = MaterializedProjectResult(
        ordinal=second.ordinal,
        cell_id=second.cell_id,
        result=second.result.model_copy(
            update={"data_version": "wrong-data-version"}
        ),
    )

    with pytest.raises(AuditLedgerError, match="identity"):
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit((first, invalid))
        )

    _assert_running_without_samples(fixture)


def test_strict_sample_artifact_failure_rolls_back_every_row(
    fixture: _Fixture,
) -> None:
    first, second = _samples()
    artifact = dict(second.result.values["artifact"])
    artifact["client_supplied_value"] = 123
    invalid = MaterializedProjectResult(
        ordinal=second.ordinal,
        cell_id=second.cell_id,
        result=second.result.model_copy(
            update={
                "values": {
                    "artifact_type": (
                        ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE
                    ),
                    "artifact": artifact,
                }
            }
        ),
    )

    with pytest.raises(AuditLedgerError, match="sample"):
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit((first, invalid))
        )

    _assert_running_without_samples(fixture)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"claim_attempt": 2}, "claim"),
        ({"claim_lease_expires_at": NOW - timedelta(seconds=1)}, "claim"),
        ({"request_sha256": "f" * 64}, "identity"),
    ],
)
def test_rejects_stale_claim_or_frozen_identity(
    fixture: _Fixture,
    update: dict[str, object],
    message: str,
) -> None:
    commit = _commit(_samples())

    with pytest.raises(AuditLedgerError, match=message):
        fixture.ledger.commit_advanced_calibration_materialization(
            replace(commit, **update)
        )

    _assert_running_without_samples(fixture)


def test_route_head_change_rejects_commit_without_partial_rows(
    fixture: _Fixture,
) -> None:
    with fixture.session_factory.begin() as session:
        head = session.get(
            ModelRouteActivationStreamHead,
            {
                "project_id": PROJECT_ID,
                "task": AdvancedModelTask.RUL.value,
                "cutoff_cycle": 100,
                "route_role": AdvancedModelRouteRole.COVERAGE.value,
            },
        )
        assert head is not None
        head.head_event_sha256 = "f" * 64

    with pytest.raises(AuditLedgerError, match="route"):
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit(_samples())
        )

    _assert_running_without_samples(fixture)


@pytest.mark.parametrize("scope", ["project", "user", "session"])
def test_inactive_project_or_revoked_admin_scope_rejects_commit(
    fixture: _Fixture,
    scope: str,
) -> None:
    with fixture.session_factory.begin() as session:
        if scope == "project":
            project = session.get(Project, PROJECT_ID)
            assert project is not None
            project.status = "ARCHIVED"
        elif scope == "user":
            user = session.get(User, USER_ID)
            assert user is not None
            user.status = UserStatus.DISABLED.value
        else:
            actor_session = session.get(SessionRecord, SESSION_ID)
            assert actor_session is not None
            actor_session.status = SessionStatus.REVOKED.value
            actor_session.revoked_at = NOW

    with pytest.raises(AuditLedgerError, match=r"project|ADMIN session"):
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit(_samples())
        )

    _assert_running_without_samples(fixture)


def test_ready_retry_is_idempotent_but_different_content_conflicts(
    fixture: _Fixture,
) -> None:
    samples = _samples()
    first = fixture.ledger.commit_advanced_calibration_materialization(
        _commit(samples)
    )

    repeated = fixture.ledger.commit_advanced_calibration_materialization(
        _commit(samples)
    )

    assert repeated == first
    changed_result = samples[1].result.model_copy(
        update={
            "values": {
                **samples[1].result.values,
                "artifact": {
                    **samples[1].result.values["artifact"],
                    "point_prediction_cycle": 799.0,
                },
            }
        }
    )
    with pytest.raises(AuditLedgerError, match="content"):
        fixture.ledger.commit_advanced_calibration_materialization(
            _commit(
                (
                    samples[0],
                    MaterializedProjectResult(
                        ordinal=1,
                        cell_id="cell-b",
                        result=changed_result,
                    ),
                )
            )
        )
    with fixture.session_factory() as session:
        assert _count(session, ToolResultRecord) == 2
        assert _count(session, AdvancedCalibrationSampleBinding) == 2


def test_concurrent_exact_commits_resolve_one_ready_materialization(
    fixture: _Fixture,
) -> None:
    commit = _commit(_samples())
    barrier = Barrier(2)

    def run_commit() -> object:
        barrier.wait()
        return fixture.ledger.commit_advanced_calibration_materialization(
            commit
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(executor.map(lambda _: run_commit(), range(2)))

    assert receipts[0] == receipts[1]
    with fixture.session_factory() as session:
        assert _count(session, ToolResultRecord) == 2
        assert _count(session, AdvancedCalibrationSampleBinding) == 2


def test_ready_binding_tampering_is_rejected_on_retry(
    fixture: _Fixture,
) -> None:
    commit = _commit(_samples())
    receipt = fixture.ledger.commit_advanced_calibration_materialization(
        commit
    )
    with fixture.session_factory.begin() as session:
        binding = session.get(
            ProjectToolResultBindingRecord,
            receipt.result_ids[0],
        )
        assert binding is not None
        binding.actor_role = UserRole.MEMBER.value

    with pytest.raises(AuditLedgerError, match=r"READY|content"):
        fixture.ledger.commit_advanced_calibration_materialization(commit)


def _commit(
    samples: tuple[MaterializedProjectResult, ...],
) -> ProjectMaterializationCommit:
    return ProjectMaterializationCommit(
        materialization_id=MATERIALIZATION_ID,
        project_id=PROJECT_ID,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        route_role=AdvancedModelRouteRole.COVERAGE,
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-split-v1",
        feature_version="advanced-feature-v1",
        artifact_id=ARTIFACT_ID,
        artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        model_version="cyclepatch-direct-v1",
        normalization_statistics_sha256=NORMALIZATION_SHA256,
        decision_event_id=DECISION_EVENT_ID,
        ledger_sequence_number=3,
        ledger_head_sha256=LEDGER_HEAD_SHA256,
        source_registration_id="matr-three-batch-final-v1",
        source_identity_sha256=SOURCE_IDENTITY_SHA256,
        request_sha256=REQUEST_SHA256,
        claim_token=CLAIM_TOKEN,
        claim_attempt=CLAIM_ATTEMPT,
        claim_lease_expires_at=CLAIM_LEASE,
        samples=samples,
    )


def _samples() -> tuple[MaterializedProjectResult, ...]:
    return tuple(
        _sample(ordinal, cell_id, observed, predicted)
        for ordinal, (cell_id, observed, predicted) in enumerate(
            (
                ("cell-a", 611, 620.0),
                ("cell-b", 702, 710.0),
            )
        )
    )


def _sample(
    ordinal: int,
    cell_id: str,
    observed_cycle: int,
    predicted_cycle: float,
) -> MaterializedProjectResult:
    artifact = {
        "task": AdvancedModelTask.RUL.value,
        "route_role": AdvancedModelRouteRole.COVERAGE.value,
        "output_target": "matr_official_cycle_life",
        "artifact_kind": "cyclepatch_direct",
        "artifact_id": ARTIFACT_ID,
        "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
        "model_version": "cyclepatch-direct-v1",
        "dataset_id": "MATR",
        "cutoff_cycle": 100,
        "data_version": "matr-three-batch-v1",
        "feature_version": "advanced-feature-v1",
        "split_version": "matr-three-batch-split-v1",
        "normalization_statistics_sha256": NORMALIZATION_SHA256,
        "decision_event_id": DECISION_EVENT_ID,
        "ledger_sequence_number": 3,
        "ledger_head_sha256": LEDGER_HEAD_SHA256,
        "materialization_id": MATERIALIZATION_ID,
        "source_registration_id": "matr-three-batch-final-v1",
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "split_partition": "calibration",
        "cell_id": cell_id,
        "point_prediction_cycle": predicted_cycle,
        "observed_cycle": observed_cycle,
    }
    result_id = str(
        uuid5(
            UUID(MATERIALIZATION_ID),
            sha256_canonical(
                {
                    "ordinal": ordinal,
                    "cell_id": cell_id,
                    "artifact": artifact,
                }
            ),
        )
    )
    created_at = NOW
    result = ToolResult(
        result_id=result_id,
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version="advanced-calibration-sample-producer-v1",
        model_version="cyclepatch-direct-v1",
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        input_hash=sha256_canonical(artifact),
        values={
            "artifact_type": ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            "artifact": artifact,
        },
        uncertainty=None,
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="advanced-calibration-source-matr-three-batch-final-v1",
                source_kind=SourceKind.OBSERVED,
                uri=(
                    "calibration-source://matr-three-batch-final-v1/"
                    f"{SOURCE_IDENTITY_SHA256}"
                ),
                sha256=SOURCE_IDENTITY_SHA256,
                description="Verified calibration supervision",
                created_at=created_at,
            ),
            ProvenanceRecord(
                source_id=f"advanced-model-{ARTIFACT_ID}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{ARTIFACT_ID}",
                sha256=ARTIFACT_MANIFEST_SHA256,
                description="Verified active runtime",
                created_at=created_at,
            ),
        ],
        created_at=created_at,
    )
    return MaterializedProjectResult(
        ordinal=ordinal,
        cell_id=cell_id,
        result=result,
    )


def _assert_running_without_samples(fixture: _Fixture) -> None:
    with fixture.session_factory() as session:
        materialization = session.get(
            AdvancedCalibrationMaterialization,
            MATERIALIZATION_ID,
        )
        assert materialization is not None
        assert materialization.status == "RUNNING"
        assert materialization.sample_count == 0
        assert materialization.sample_manifest_sha256 is None
        assert _count(session, ToolResultRecord) == 0
        assert _count(session, ProvenanceRecordRow) == 0
        assert _count(session, ProjectToolResultBindingRecord) == 0
        assert _count(session, AdvancedCalibrationSampleBinding) == 0


def _count(session: Session, model: type[object]) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)
