from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import quanxin_life.application.advanced_calibration_jobs as jobs_module
from quanxin_life.application.advanced_calibration_materialization import (
    AdvancedCalibrationMaterializationRequest,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit.project_ledger import (
    MaterializedProjectResult,
    ProjectMaterializationCommit,
    ProjectMaterializationReceipt,
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
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
    SessionRecord,
    User,
)

NOW = datetime(2026, 7, 27, 7, 0, tzinfo=UTC)
PROJECT_ID = "0166c272-9e52-4f77-882b-031c8557924b"
USER_ID = "3ffbd300-b982-463f-8129-d667293f0b19"
SESSION_ID = "e0b0b2e7-f583-49ff-8641-fd53c2054f65"
ARTIFACT_ID = "40b87568-7c8e-481f-a284-945480d62fbc"
DECISION_EVENT_ID = "cf2632ec-e75f-4ba1-943d-f8b71e34727e"
MATERIALIZATION_ID = "1e8a55ad-4fd8-48d8-8dc8-b7f02c459507"
SOURCE_REGISTRATION_ID = "matr-three-batch-final-v1"
ARTIFACT_MANIFEST_SHA256 = "3" * 64
NORMALIZATION_SHA256 = "4" * 64
LEDGER_HEAD_SHA256 = "5" * 64
SOURCE_IDENTITY_SHA256 = "2" * 64
CLAIM_TOKEN = "advanced-calibration-claim-token-v2"
CLAIM_TOKEN_SHA256 = sha256_canonical({"claim_token": CLAIM_TOKEN})
LEASE_TTL = timedelta(minutes=15)


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self._value = value
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._value

    def set(self, value: datetime) -> None:
        with self._lock:
            self._value = value


@dataclass(frozen=True, slots=True)
class _PreparedIdentity:
    data_version: str = "matr-three-batch-v1"
    split_version: str = "matr-three-batch-split-v1"
    feature_version: str = "advanced-feature-v1"
    artifact_id: str = ARTIFACT_ID
    artifact_manifest_sha256: str = ARTIFACT_MANIFEST_SHA256
    model_version: str = "cyclepatch-direct-v1"
    normalization_statistics_sha256: str = NORMALIZATION_SHA256
    decision_event_id: str = DECISION_EVENT_ID
    ledger_sequence_number: int = 3
    ledger_head_sha256: str = LEDGER_HEAD_SHA256
    source_registration_id: str = SOURCE_REGISTRATION_ID
    source_identity_sha256: str = SOURCE_IDENTITY_SHA256


class _PreparationResolver:
    def __init__(self) -> None:
        self.calls: list[
            tuple[VerifiedProjectInvocationContext, AdvancedCalibrationMaterializationRequest]
        ] = []

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
    ) -> _PreparedIdentity:
        self.calls.append((context, request))
        return _PreparedIdentity(source_registration_id=request.source_registration_id)


class _Producer:
    def __init__(
        self,
        *,
        samples: tuple[MaterializedProjectResult, ...] = (),
        before_return: Callable[[], None] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.samples = samples
        self.before_return = before_return
        self.error = error
        self.calls: list[
            tuple[
                VerifiedProjectInvocationContext,
                str,
                AdvancedCalibrationMaterializationRequest,
                datetime,
            ]
        ] = []

    def produce(
        self,
        *,
        context: VerifiedProjectInvocationContext,
        materialization_id: str,
        request: AdvancedCalibrationMaterializationRequest,
        created_at: datetime,
    ) -> tuple[MaterializedProjectResult, ...]:
        self.calls.append((context, materialization_id, request, created_at))
        if self.before_return is not None:
            self.before_return()
        if self.error is not None:
            raise self.error
        return self.samples


class _CapturingMaterializer:
    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        completed_at: datetime,
    ) -> None:
        self._session_factory = session_factory
        self._completed_at = completed_at
        self.commits: list[ProjectMaterializationCommit] = []

    def commit_advanced_calibration_materialization(
        self,
        commit: ProjectMaterializationCommit,
    ) -> ProjectMaterializationReceipt:
        self.commits.append(commit)
        result_ids = tuple(item.result.result_id for item in commit.samples)
        manifest_sha256 = sha256_canonical(
            {
                "materialization_id": commit.materialization_id,
                "result_ids": result_ids,
            }
        )
        with self._session_factory.begin() as session:
            row = session.get(
                AdvancedCalibrationMaterialization,
                commit.materialization_id,
            )
            assert row is not None
            assert row.claim_token_sha256 == sha256_canonical(
                {"claim_token": commit.claim_token}
            )
            assert row.claim_attempt == commit.claim_attempt
            row.status = AdvancedCalibrationMaterializationStatus.READY.value
            row.sample_count = len(commit.samples)
            row.sample_manifest_sha256 = manifest_sha256
            row.completed_at = self._completed_at
            row.failure_code = None
            row.claim_token_sha256 = None
            row.claim_lease_expires_at = None
        return ProjectMaterializationReceipt(
            materialization_id=commit.materialization_id,
            sample_count=len(commit.samples),
            sample_manifest_sha256=manifest_sha256,
            result_ids=result_ids,
        )


@dataclass(frozen=True, slots=True)
class _Fixture:
    session_factory: SessionFactory
    context_service: ProjectInvocationContextService
    context: VerifiedProjectInvocationContext
    request: AdvancedCalibrationMaterializationRequest
    preparation_resolver: _PreparationResolver

    def service(self) -> Any:
        service_type = _required_symbol("AdvancedCalibrationMaterializationService")
        return service_type(
            self.session_factory,
            preparation_resolver=self.preparation_resolver,
            clock=lambda: NOW,
            id_factory=lambda: MATERIALIZATION_ID,
        )

    def worker(
        self,
        *,
        producer: _Producer,
        materializer: _CapturingMaterializer,
        token: str = CLAIM_TOKEN,
    ) -> Any:
        worker_type = _required_symbol("AdvancedCalibrationMaterializationWorker")
        return worker_type(
            self.session_factory,
            context_service=self.context_service,
            producer=producer,
            materializer=materializer,
            clock=lambda: NOW,
            token_factory=lambda: token,
            lease_ttl=LEASE_TTL,
        )


@pytest.fixture
def fixture(tmp_path: Path) -> _Fixture:
    engine = create_engine_from_config(
        DatabaseConfig(
            url=f"sqlite+pysqlite:///{tmp_path / 'advanced-calibration-jobs.sqlite3'}"
        )
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    with session_factory.begin() as session:
        _seed_project_route(session)
    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    context = context_service.resolve_http(
        AuthPrincipal(
            user_id=USER_ID,
            session_id=SESSION_ID,
            username="calibration-admin@example.test",
            role=UserRole.ADMIN,
            must_change_password=False,
        ),
        PROJECT_ID,
    )
    return _Fixture(
        session_factory=session_factory,
        context_service=context_service,
        context=context,
        request=AdvancedCalibrationMaterializationRequest(
            project_id=PROJECT_ID,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=100,
            route_role=AdvancedModelRouteRole.COVERAGE,
            source_registration_id=SOURCE_REGISTRATION_ID,
        ),
        preparation_resolver=_PreparationResolver(),
    )


def test_create_is_identity_only_idempotent_and_exposes_safe_queries(
    fixture: _Fixture,
) -> None:
    service = fixture.service()

    created = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )
    repeated = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )

    record_type = _required_symbol("AdvancedCalibrationMaterializationRecord")
    assert isinstance(created, record_type)
    assert repeated == created
    assert created.materialization_id == MATERIALIZATION_ID
    assert created.project_id == PROJECT_ID
    assert created.status is AdvancedCalibrationMaterializationStatus.PENDING
    assert service.get(
        fixture.context,
        materialization_id=MATERIALIZATION_ID,
    ) == created
    assert service.list(fixture.context) == (created,)
    payload = _record_payload(created)
    assert not {
        "samples",
        "sample_ids",
        "result_ids",
        "cell_ids",
        "observed",
        "predicted",
        "paths",
    }.intersection(payload)
    with fixture.session_factory() as session:
        rows = tuple(session.scalars(select(AdvancedCalibrationMaterialization)))
        assert len(rows) == 1
        assert len(rows[0].idempotency_key_sha256) == 64
        assert rows[0].idempotency_key_sha256 != "prepare-rul-coverage-100"
        assert len(rows[0].request_sha256) == 64


def test_create_rejects_reusing_an_idempotency_key_for_another_request(
    fixture: _Fixture,
) -> None:
    service = fixture.service()
    service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )
    changed = fixture.request.model_copy(
        update={"source_registration_id": "matr-three-batch-final-v2"}
    )

    with pytest.raises(
        _materialization_error(),
        match="idempotency",
    ):
        service.create(
            fixture.context,
            changed,
            idempotency_key="prepare-rul-coverage-100",
        )

    with fixture.session_factory() as session:
        assert _count(session, AdvancedCalibrationMaterialization) == 1


def test_idempotent_retry_returns_original_before_resolving_mutable_preparation(
    fixture: _Fixture,
) -> None:
    service = fixture.service()
    created = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )

    def reject_changed_server_state(
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
    ) -> _PreparedIdentity:
        del context, request
        raise RuntimeError("active route changed after the first response")

    fixture.preparation_resolver.resolve = reject_changed_server_state  # type: ignore[method-assign]

    repeated = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )

    assert repeated == created


def test_status_queries_mark_ready_evidence_stale_after_route_change(
    fixture: _Fixture,
) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.READY,
        claim_attempt=1,
    )
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

    detail = fixture.service().get(
        fixture.context,
        materialization_id=MATERIALIZATION_ID,
    )
    collection = fixture.service().list(fixture.context)

    assert detail.status is AdvancedCalibrationMaterializationStatus.STALE
    assert detail.failure_code == "ACTIVE_ROUTE_CHANGED"
    assert collection == (detail,)


def test_idempotent_retry_marks_original_ready_record_stale_after_route_change(
    fixture: _Fixture,
) -> None:
    service = fixture.service()
    created = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )
    with fixture.session_factory.begin() as session:
        row = session.get(
            AdvancedCalibrationMaterialization,
            created.materialization_id,
        )
        assert row is not None
        row.status = AdvancedCalibrationMaterializationStatus.READY.value
        row.started_at = NOW
        row.completed_at = NOW
        row.sample_count = 1
        row.sample_manifest_sha256 = "d" * 64
        row.claim_attempt = 1
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

    repeated = service.create(
        fixture.context,
        fixture.request,
        idempotency_key="prepare-rul-coverage-100",
    )

    assert repeated.materialization_id == created.materialization_id
    assert repeated.status is AdvancedCalibrationMaterializationStatus.STALE
    assert repeated.failure_code == "ACTIVE_ROUTE_CHANGED"


def test_worker_claims_pending_before_calling_the_producer(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    observed: dict[str, object] = {}

    def observe_claim() -> None:
        with fixture.session_factory() as session:
            row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
            assert row is not None
            observed.update(
                status=row.status,
                claim_attempt=row.claim_attempt,
                claim_token_sha256=row.claim_token_sha256,
                lease=row.claim_lease_expires_at,
            )

    producer = _Producer(samples=_samples(), before_return=observe_claim)
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert observed == {
        "status": AdvancedCalibrationMaterializationStatus.RUNNING.value,
        "claim_attempt": 1,
        "claim_token_sha256": CLAIM_TOKEN_SHA256,
        "lease": NOW + LEASE_TTL,
    }


def test_worker_rejects_a_live_running_claim_as_busy(fixture: _Fixture) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.RUNNING,
        claim_token="live-worker-token",
        claim_attempt=1,
        lease_expires_at=NOW + timedelta(minutes=1),
    )
    producer = _Producer(samples=_samples())
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    with pytest.raises(
        _required_symbol("AdvancedCalibrationMaterializationBusyError"),
        match=r"claim|lease|running",
    ):
        fixture.worker(
            producer=producer,
            materializer=materializer,
        ).execute(materialization_id=MATERIALIZATION_ID)

    assert producer.calls == []
    assert materializer.commits == []


def test_worker_default_lease_expires_before_busy_retries_are_exhausted(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    observed: dict[str, datetime] = {}

    def observe_default_lease() -> None:
        with fixture.session_factory() as session:
            row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
            assert row is not None
            assert row.claim_lease_expires_at is not None
            observed["lease"] = row.claim_lease_expires_at

    worker_type = _required_symbol("AdvancedCalibrationMaterializationWorker")
    status = worker_type(
        fixture.session_factory,
        context_service=fixture.context_service,
        producer=_Producer(samples=_samples(), before_return=observe_default_lease),
        materializer=_CapturingMaterializer(
            fixture.session_factory,
            completed_at=NOW,
        ),
        clock=lambda: NOW,
        token_factory=lambda: CLAIM_TOKEN,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert observed["lease"] <= NOW + timedelta(seconds=95)


def test_worker_heartbeats_keep_a_live_long_running_claim_fenced(
    fixture: _Fixture,
) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.PENDING,
    )
    producer_started = Event()
    release_producer = Event()
    clock = _MutableClock(NOW)
    first_result: list[AdvancedCalibrationMaterializationStatus] = []
    first_error: list[BaseException] = []

    def block_during_production() -> None:
        producer_started.set()
        assert release_producer.wait(timeout=3)

    worker_type = _required_symbol("AdvancedCalibrationMaterializationWorker")
    first_worker = worker_type(
        fixture.session_factory,
        context_service=fixture.context_service,
        producer=_Producer(
            samples=_samples(),
            before_return=block_during_production,
        ),
        materializer=_CapturingMaterializer(
            fixture.session_factory,
            completed_at=NOW,
        ),
        clock=clock,
        token_factory=lambda: CLAIM_TOKEN,
        lease_ttl=timedelta(minutes=1),
        heartbeat_interval=timedelta(milliseconds=5),
    )

    def execute_first_worker() -> None:
        try:
            first_result.append(
                first_worker.execute(materialization_id=MATERIALIZATION_ID)
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            first_error.append(exc)

    thread = Thread(target=execute_first_worker)
    thread.start()
    assert producer_started.wait(timeout=3)

    clock.set(NOW + timedelta(seconds=30))
    deadline = monotonic() + 3
    renewed_lease: datetime | None = None
    while monotonic() < deadline:
        with fixture.session_factory() as session:
            row = session.get(
                AdvancedCalibrationMaterialization,
                MATERIALIZATION_ID,
            )
            assert row is not None
            renewed_lease = row.claim_lease_expires_at
        if renewed_lease == NOW + timedelta(seconds=90):
            break
        Event().wait(0.01)
    assert renewed_lease == NOW + timedelta(seconds=90)

    clock.set(NOW + timedelta(seconds=70))
    second_worker = worker_type(
        fixture.session_factory,
        context_service=fixture.context_service,
        producer=_Producer(samples=_samples()),
        materializer=_CapturingMaterializer(
            fixture.session_factory,
            completed_at=NOW,
        ),
        clock=clock,
        token_factory=lambda: "replacement-token",
        lease_ttl=timedelta(minutes=1),
        heartbeat_interval=timedelta(milliseconds=5),
    )
    with pytest.raises(
        _required_symbol("AdvancedCalibrationMaterializationBusyError"),
        match=r"claim|lease|running",
    ):
        second_worker.execute(materialization_id=MATERIALIZATION_ID)

    release_producer.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert first_error == []
    assert first_result == [AdvancedCalibrationMaterializationStatus.READY]


def test_worker_recovers_an_expired_running_claim_with_a_new_fence(
    fixture: _Fixture,
) -> None:
    old_token = "expired-worker-token"
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.RUNNING,
        claim_token=old_token,
        claim_attempt=1,
        lease_expires_at=NOW - timedelta(seconds=1),
    )
    producer = _Producer(samples=_samples())
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert len(materializer.commits) == 1
    commit = materializer.commits[0]
    assert commit.claim_token == CLAIM_TOKEN
    assert commit.claim_attempt == 2
    assert commit.claim_lease_expires_at == NOW + LEASE_TTL
    assert sha256_canonical({"claim_token": old_token}) != CLAIM_TOKEN_SHA256


def test_worker_retries_a_failed_materialization_with_a_new_attempt(
    fixture: _Fixture,
) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.FAILED,
        claim_attempt=1,
        failure_code="RuntimeError",
    )
    producer = _Producer(samples=_samples())
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert materializer.commits[0].claim_attempt == 2
    with fixture.session_factory() as session:
        row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
        assert row is not None
        assert row.failure_code is None


def test_worker_returns_ready_without_replaying_production(
    fixture: _Fixture,
) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.READY,
        claim_attempt=1,
    )
    producer = _Producer(samples=_samples())
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert producer.calls == []
    assert materializer.commits == []


def test_worker_returns_stale_without_replaying_production(
    fixture: _Fixture,
) -> None:
    _seed_materialization(
        fixture,
        status=AdvancedCalibrationMaterializationStatus.STALE,
        claim_attempt=1,
        failure_code="ACTIVE_ROUTE_CHANGED",
    )
    producer = _Producer(samples=_samples())
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.STALE
    assert producer.calls == []
    assert materializer.commits == []


def test_worker_failure_cannot_overwrite_a_newer_fenced_claim(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    newer_token = "replacement-worker-token"
    newer_token_sha256 = sha256_canonical({"claim_token": newer_token})

    def replace_claim() -> None:
        with fixture.session_factory.begin() as session:
            row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
            assert row is not None
            row.claim_token_sha256 = newer_token_sha256
            row.claim_attempt = 2
            row.claim_lease_expires_at = NOW + timedelta(minutes=30)

    producer = _Producer(
        before_return=replace_claim,
        error=RuntimeError("sensitive producer detail must not be persisted"),
    )
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    with pytest.raises(
        _materialization_error(),
        match=r"claim|fence|stale",
    ) as exc_info:
        fixture.worker(
            producer=producer,
            materializer=materializer,
        ).execute(materialization_id=MATERIALIZATION_ID)

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__
    assert "sensitive producer detail" not in str(exc_info.value)
    with fixture.session_factory() as session:
        row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
        assert row is not None
        assert row.status == AdvancedCalibrationMaterializationStatus.RUNNING.value
        assert row.claim_token_sha256 == newer_token_sha256
        assert row.claim_attempt == 2
        assert row.failure_code is None


def test_worker_cannot_mark_failure_after_its_lease_expires(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    clock_values = iter((NOW, NOW, NOW + LEASE_TTL + timedelta(seconds=1)))
    worker_type = _required_symbol("AdvancedCalibrationMaterializationWorker")
    worker = worker_type(
        fixture.session_factory,
        context_service=fixture.context_service,
        producer=_Producer(error=RuntimeError("expired owner")),
        materializer=_CapturingMaterializer(
            fixture.session_factory,
            completed_at=NOW,
        ),
        clock=lambda: next(clock_values),
        token_factory=lambda: CLAIM_TOKEN,
        lease_ttl=LEASE_TTL,
    )

    with pytest.raises(_materialization_error(), match=r"claim|fence|stale"):
        worker.execute(materialization_id=MATERIALIZATION_ID)

    with fixture.session_factory() as session:
        row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
        assert row is not None
        assert row.status == AdvancedCalibrationMaterializationStatus.RUNNING.value
        assert row.claim_token_sha256 == CLAIM_TOKEN_SHA256
        assert row.claim_attempt == 1
        assert row.failure_code is None


def test_worker_marks_an_owned_producer_failure_without_persisting_its_message(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    producer = _Producer(
        error=RuntimeError("sensitive producer detail must not be persisted")
    )
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.FAILED
    with fixture.session_factory() as session:
        row = session.get(AdvancedCalibrationMaterialization, MATERIALIZATION_ID)
        assert row is not None
        assert row.status == AdvancedCalibrationMaterializationStatus.FAILED.value
        assert row.failure_code == "RuntimeError"
        assert "sensitive" not in row.failure_code
        assert row.claim_token_sha256 is None
        assert row.claim_lease_expires_at is None
        assert row.claim_attempt == 1


def test_worker_builds_the_atomic_commit_from_frozen_state_and_producer_output(
    fixture: _Fixture,
) -> None:
    _seed_materialization(fixture, status=AdvancedCalibrationMaterializationStatus.PENDING)
    samples = _samples()
    producer = _Producer(samples=samples)
    materializer = _CapturingMaterializer(fixture.session_factory, completed_at=NOW)

    status = fixture.worker(
        producer=producer,
        materializer=materializer,
    ).execute(materialization_id=MATERIALIZATION_ID)

    assert status is AdvancedCalibrationMaterializationStatus.READY
    assert len(producer.calls) == 1
    produced_context, produced_id, produced_request, produced_at = producer.calls[0]
    assert produced_context.project_id == PROJECT_ID
    assert produced_context.actor_role is UserRole.ADMIN
    assert produced_id == MATERIALIZATION_ID
    assert produced_request == fixture.request
    assert produced_at == NOW
    assert materializer.commits == [
        ProjectMaterializationCommit(
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
            source_registration_id=SOURCE_REGISTRATION_ID,
            source_identity_sha256=SOURCE_IDENTITY_SHA256,
            request_sha256=_request_sha256(fixture.request),
            claim_token=CLAIM_TOKEN,
            claim_attempt=1,
            claim_lease_expires_at=NOW + LEASE_TTL,
            samples=samples,
        )
    ]


def _seed_project_route(session: Session) -> None:
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
            name="Advanced calibration jobs",
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


def _seed_materialization(
    fixture: _Fixture,
    *,
    status: AdvancedCalibrationMaterializationStatus,
    claim_token: str | None = None,
    claim_attempt: int = 0,
    lease_expires_at: datetime | None = None,
    failure_code: str | None = None,
) -> None:
    started_at: datetime | None = None
    completed_at: datetime | None = None
    sample_count = 0
    sample_manifest_sha256: str | None = None
    claim_token_sha256: str | None = None
    if status is AdvancedCalibrationMaterializationStatus.RUNNING:
        assert claim_token is not None
        started_at = NOW - timedelta(minutes=5)
        claim_token_sha256 = sha256_canonical({"claim_token": claim_token})
    elif status is AdvancedCalibrationMaterializationStatus.FAILED:
        started_at = NOW - timedelta(minutes=5)
        completed_at = NOW - timedelta(minutes=1)
    elif status in {
        AdvancedCalibrationMaterializationStatus.READY,
        AdvancedCalibrationMaterializationStatus.STALE,
    }:
        started_at = NOW - timedelta(minutes=5)
        completed_at = NOW - timedelta(minutes=1)
        sample_count = 1
        sample_manifest_sha256 = "d" * 64
    with fixture.session_factory.begin() as session:
        session.add(
            AdvancedCalibrationMaterialization(
                id=MATERIALIZATION_ID,
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL.value,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE.value,
                status=status.value,
                data_version="matr-three-batch-v1",
                split_version="matr-three-batch-split-v1",
                feature_version="advanced-feature-v1",
                artifact_id=ARTIFACT_ID,
                artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
                normalization_statistics_sha256=NORMALIZATION_SHA256,
                decision_event_id=DECISION_EVENT_ID,
                ledger_sequence_number=3,
                ledger_head_sha256=LEDGER_HEAD_SHA256,
                source_registration_id=SOURCE_REGISTRATION_ID,
                source_identity_sha256=SOURCE_IDENTITY_SHA256,
                sample_manifest_sha256=sample_manifest_sha256,
                sample_count=sample_count,
                idempotency_key_sha256="c" * 64,
                request_sha256=_request_sha256(fixture.request),
                created_by_user_id=USER_ID,
                created_by_session_id=SESSION_ID,
                created_by_role=UserRole.ADMIN.value,
                created_at=NOW - timedelta(minutes=10),
                started_at=started_at,
                completed_at=completed_at,
                failure_code=failure_code,
                claim_token_sha256=claim_token_sha256,
                claim_attempt=claim_attempt,
                claim_lease_expires_at=lease_expires_at,
            )
        )


def _samples() -> tuple[MaterializedProjectResult, ...]:
    result = ToolResult(
        result_id="2fd1487d-4141-45ac-b115-d06139a55c6f",
        tool_name="test-only-calibration-producer",
        tool_version="test-only-v1",
        model_version="cyclepatch-direct-v1",
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        input_hash="e" * 64,
        values={
            "artifact_type": "test-only-calibration-sample",
            "artifact": {"cell_id": "test-cell-a"},
        },
        uncertainty=None,
        warnings=["TEST_ONLY_FIXTURE"],
        provenance=[
            ProvenanceRecord(
                source_id="test-only-calibration-source",
                source_kind=SourceKind.OBSERVED,
                uri="test-only://advanced-calibration/source",
                sha256="f" * 64,
                description="Explicit integration-test-only source",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )
    return (
        MaterializedProjectResult(
            ordinal=0,
            cell_id="test-cell-a",
            result=result,
        ),
    )


def _request_sha256(request: AdvancedCalibrationMaterializationRequest) -> str:
    return sha256_canonical(request.model_dump(mode="json"))


def _required_symbol(name: str) -> Any:
    value = getattr(jobs_module, name, None)
    if value is None:
        pytest.fail(f"missing public symbol: {name}")
    return value


def _materialization_error() -> type[BaseException]:
    return cast(
        type[BaseException],
        _required_symbol("AdvancedCalibrationMaterializationError"),
    )


def _record_payload(record: object) -> dict[str, object]:
    if hasattr(record, "model_dump"):
        return cast(dict[str, object], record.model_dump(mode="json"))
    if is_dataclass(record):
        return cast(dict[str, object], asdict(record))
    return dict(vars(record))


def _count(session: Session, model: type[object]) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)
