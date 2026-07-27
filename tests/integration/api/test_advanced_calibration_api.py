from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from quanxin_life.api.advanced_calibration import (
    create_advanced_calibration_http_adapter,
)

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationDispatchReceipt,
    AdvancedCalibrationMaterializationError,
    AdvancedCalibrationMaterializationRecord,
)
from quanxin_life.application.advanced_calibration_materialization import (
    AdvancedCalibrationMaterializationRequest,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProjectStatus,
    SessionStatus,
    UserRole,
    UserStatus,
)
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    Project,
    SessionRecord,
    User,
    UserProjectRole,
)

NOW = datetime(2026, 7, 27, 9, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
PASSWORD = "AdvancedCalibrationApi-2026!"
SOURCE_REGISTRATION_ID = "matr-three-batch-final-v1"
READY_MATERIALIZATION_ID = "83d689f2-3bee-458c-9857-e4c888b07bd5"
CREATED_MATERIALIZATION_ID = "7146be3e-1537-4f13-a83b-468f4b23bc5c"
ARTIFACT_ID = "154d7ac8-694d-4504-83f0-1d5ef08ac73e"
DECISION_EVENT_ID = "dc847030-21bd-4bcc-9b48-dbd9bfe556c8"
SAFE_SUMMARY_KEYS = {
    "materialization_id",
    "project_id",
    "task",
    "cutoff_cycle",
    "route_role",
    "status",
    "data_version",
    "split_version",
    "feature_version",
    "artifact_id",
    "artifact_manifest_sha256",
    "normalization_statistics_sha256",
    "decision_event_id",
    "ledger_sequence_number",
    "ledger_head_sha256",
    "source_registration_id",
    "source_identity_sha256",
    "sample_manifest_sha256",
    "sample_count",
    "created_at",
    "started_at",
    "completed_at",
    "failure_code",
}
FORBIDDEN_RESPONSE_KEYS = {
    "cell_id",
    "cell_ids",
    "observed",
    "observed_soh",
    "predicted",
    "predicted_soh",
    "prediction",
    "predictions",
    "label",
    "labels",
    "value",
    "values",
    "sample_ids",
    "result_ids",
    "path",
    "paths",
    "uri",
    "object_uri",
    "evidence_root",
    "claim_token",
}


class _MaterializationService:
    def __init__(self, *, project_id: str) -> None:
        self.create_calls: list[
            tuple[
                VerifiedProjectInvocationContext,
                AdvancedCalibrationMaterializationRequest,
                str,
            ]
        ] = []
        self.get_calls: list[tuple[str, str]] = []
        self.list_calls: list[str] = []
        self._records = {
            READY_MATERIALIZATION_ID: _record(
                materialization_id=READY_MATERIALIZATION_ID,
                project_id=project_id,
                status=AdvancedCalibrationMaterializationStatus.READY,
            )
        }
        self._idempotent: dict[str, AdvancedCalibrationMaterializationRecord] = {}

    def create(
        self,
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
        *,
        idempotency_key: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        self.create_calls.append((context, request, idempotency_key))
        if context.project_id != request.project_id:
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration materialization was not found"
            )
        existing = self._idempotent.get(idempotency_key)
        if existing is not None:
            return existing
        record = _record(
            materialization_id=CREATED_MATERIALIZATION_ID,
            project_id=request.project_id,
            status=AdvancedCalibrationMaterializationStatus.PENDING,
            task=request.task,
            cutoff_cycle=request.cutoff_cycle,
            route_role=request.route_role,
            source_registration_id=request.source_registration_id,
        )
        self._records[record.materialization_id] = record
        self._idempotent[idempotency_key] = record
        return record

    def get(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        self.get_calls.append((context.project_id, materialization_id))
        record = self._records.get(materialization_id)
        if record is None or record.project_id != context.project_id:
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration materialization was not found"
            )
        return record

    def list(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> tuple[AdvancedCalibrationMaterializationRecord, ...]:
        self.list_calls.append(context.project_id)
        return tuple(
            record
            for record in self._records.values()
            if record.project_id == context.project_id
        )


class _IdentityOnlyQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def enqueue(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationDispatchReceipt:
        payload = {"materialization_id": materialization_id}
        self.calls.append(payload)
        return AdvancedCalibrationDispatchReceipt(
            materialization_id=materialization_id,
            task_id=f"calibration-task-{len(self.calls):04d}",
        )


@dataclass(frozen=True, slots=True)
class _ApiContext:
    admin_client: TestClient
    member_client: TestClient
    outsider_client: TestClient
    session_factory: SessionFactory
    service: _MaterializationService
    queue: _IdentityOnlyQueue
    admin_user_id: str
    member_user_id: str
    active_project_id: str
    other_project_id: str
    archived_project_id: str


def _record(
    *,
    materialization_id: str,
    project_id: str,
    status: AdvancedCalibrationMaterializationStatus,
    task: AdvancedModelTask = AdvancedModelTask.RUL,
    cutoff_cycle: int = 100,
    route_role: AdvancedModelRouteRole = AdvancedModelRouteRole.COVERAGE,
    source_registration_id: str = SOURCE_REGISTRATION_ID,
) -> AdvancedCalibrationMaterializationRecord:
    ready = status is AdvancedCalibrationMaterializationStatus.READY
    return AdvancedCalibrationMaterializationRecord(
        materialization_id=materialization_id,
        project_id=project_id,
        task=task,
        cutoff_cycle=cutoff_cycle,
        route_role=route_role,
        status=status,
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-split-v1",
        feature_version="advanced-feature-v1",
        artifact_id=ARTIFACT_ID,
        artifact_manifest_sha256="1" * 64,
        normalization_statistics_sha256="2" * 64,
        decision_event_id=DECISION_EVENT_ID,
        ledger_sequence_number=4,
        ledger_head_sha256="3" * 64,
        source_registration_id=source_registration_id,
        source_identity_sha256="4" * 64,
        sample_manifest_sha256="5" * 64 if ready else None,
        sample_count=24 if ready else 0,
        created_at=NOW,
        started_at=NOW + timedelta(minutes=1) if ready else None,
        completed_at=NOW + timedelta(minutes=2) if ready else None,
        failure_code=None,
    )


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200


def _context(tmp_path: Path) -> _ApiContext:
    engine = create_engine_from_config(
        DatabaseConfig(
            url=f"sqlite+pysqlite:///{tmp_path / 'advanced-calibration-api.sqlite3'}"
        )
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    admin_user_id = str(uuid4())
    member_user_id = str(uuid4())
    outsider_user_id = str(uuid4())
    active_project_id = str(uuid4())
    other_project_id = str(uuid4())
    archived_project_id = str(uuid4())
    usernames = {
        admin_user_id: f"calibration-admin-{admin_user_id[:8]}@example.test",
        member_user_id: f"calibration-member-{member_user_id[:8]}@example.test",
        outsider_user_id: f"calibration-outsider-{outsider_user_id[:8]}@example.test",
    }
    with session_factory.begin() as session:
        for user_id, role in (
            (admin_user_id, UserRole.ADMIN),
            (member_user_id, UserRole.MEMBER),
            (outsider_user_id, UserRole.MEMBER),
        ):
            session.add(
                User(
                    id=user_id,
                    username=usernames[user_id],
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        for project_id, owner_id, status in (
            (active_project_id, member_user_id, ProjectStatus.ACTIVE),
            (other_project_id, outsider_user_id, ProjectStatus.ACTIVE),
            (archived_project_id, member_user_id, ProjectStatus.ARCHIVED),
        ):
            session.add(
                Project(
                    id=project_id,
                    owner_user_id=owner_id,
                    name=f"advanced calibration {project_id[:8]}",
                    status=status.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        for user_id, project_id in (
            (member_user_id, active_project_id),
            (member_user_id, archived_project_id),
            (outsider_user_id, other_project_id),
        ):
            session.add(
                UserProjectRole(
                    id=str(uuid4()),
                    user_id=user_id,
                    project_id=project_id,
                    role=UserRole.MEMBER.value,
                    created_at=NOW,
                )
            )

    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,)),
    )
    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    service = _MaterializationService(project_id=active_project_id)
    queue = _IdentityOnlyQueue()
    calibration_adapter = create_advanced_calibration_http_adapter(
        service,
        queue=queue,
        context_service=context_service,
        auth_adapter=auth_adapter,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        advanced_calibration_adapter=calibration_adapter,
    )
    admin_client = TestClient(app, base_url="https://api.example.test")
    member_client = TestClient(app, base_url="https://api.example.test")
    outsider_client = TestClient(app, base_url="https://api.example.test")
    _login(admin_client, usernames[admin_user_id])
    _login(member_client, usernames[member_user_id])
    _login(outsider_client, usernames[outsider_user_id])
    return _ApiContext(
        admin_client=admin_client,
        member_client=member_client,
        outsider_client=outsider_client,
        session_factory=session_factory,
        service=service,
        queue=queue,
        admin_user_id=admin_user_id,
        member_user_id=member_user_id,
        active_project_id=active_project_id,
        other_project_id=other_project_id,
        archived_project_id=archived_project_id,
    )


def _create_path(project_id: str) -> str:
    return f"/v1/projects/{project_id}/advanced-calibration/materializations"


def _create_payload() -> dict[str, object]:
    return {
        "task": AdvancedModelTask.RUL.value,
        "cutoff_cycle": 100,
        "route_role": AdvancedModelRouteRole.COVERAGE.value,
        "source_registration_id": SOURCE_REGISTRATION_ID,
    }


def _assert_safe_summary(payload: dict[str, Any]) -> None:
    assert set(payload) == SAFE_SUMMARY_KEYS
    assert payload["materialization_id"] in {
        READY_MATERIALIZATION_ID,
        CREATED_MATERIALIZATION_ID,
    }
    assert _recursive_keys(payload).isdisjoint(FORBIDDEN_RESPONSE_KEYS)
    rendered = str(payload).casefold()
    assert "sensitive-observed-value" not in rendered
    assert "sensitive-predicted-value" not in rendered
    assert "c:\\" not in rendered
    assert "/data/" not in rendered


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_recursive_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_recursive_keys(item))
        return keys
    return set()


def test_post_requires_admin_trusted_origin_and_idempotency_key(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    path = _create_path(context.active_project_id)
    payload = _create_payload()

    missing_origin = context.admin_client.post(
        path,
        headers={"Idempotency-Key": "calibration-create-0001"},
        json=payload,
    )
    missing_key = context.admin_client.post(
        path,
        headers={"Origin": ORIGIN},
        json=payload,
    )
    created = context.admin_client.post(
        path,
        headers={
            "Origin": ORIGIN,
            "Idempotency-Key": "calibration-create-0001",
        },
        json=payload,
    )

    assert missing_origin.status_code == 403
    assert missing_key.status_code == 422
    assert created.status_code == 202, created.text
    assert set(created.json()) == {"materialization", "dispatch"}
    _assert_safe_summary(created.json()["materialization"])
    assert created.json()["dispatch"] == {
        "materialization_id": CREATED_MATERIALIZATION_ID,
        "task_id": "calibration-task-0001",
    }
    assert context.queue.calls == [
        {"materialization_id": CREATED_MATERIALIZATION_ID}
    ]


def test_member_cannot_create_a_materialization(tmp_path: Path) -> None:
    context = _context(tmp_path)

    response = context.member_client.post(
        _create_path(context.active_project_id),
        headers={
            "Origin": ORIGIN,
            "Idempotency-Key": "calibration-member-0001",
        },
        json=_create_payload(),
    )

    assert response.status_code == 403
    assert context.queue.calls == []
    assert context.service.create_calls == []


def test_admin_and_member_can_get_only_safe_project_summaries(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    collection_path = _create_path(context.active_project_id)
    detail_path = f"{collection_path}/{READY_MATERIALIZATION_ID}"

    admin_list = context.admin_client.get(collection_path)
    member_list = context.member_client.get(collection_path)
    admin_detail = context.admin_client.get(detail_path)
    member_detail = context.member_client.get(detail_path)

    for response in (admin_list, member_list):
        assert response.status_code == 200, response.text
        assert len(response.json()) == 1
        _assert_safe_summary(response.json()[0])
    for response in (admin_detail, member_detail):
        assert response.status_code == 200, response.text
        _assert_safe_summary(response.json())
        assert response.json()["status"] == "READY"
        assert response.json()["sample_count"] == 24


def test_unknown_and_cross_project_materializations_are_hidden(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    unknown_id = str(uuid4())

    unknown = context.admin_client.get(
        f"{_create_path(context.active_project_id)}/{unknown_id}"
    )
    cross_project = context.admin_client.get(
        f"{_create_path(context.other_project_id)}/{READY_MATERIALIZATION_ID}"
    )
    invisible_project = context.member_client.get(
        _create_path(context.other_project_id)
    )

    assert unknown.status_code == 404
    assert cross_project.status_code == 404
    assert invisible_project.status_code == 404
    assert unknown.json() == {"detail": "advanced_calibration_not_found"}
    assert cross_project.json() == {"detail": "advanced_calibration_not_found"}


def test_revoked_session_and_inactive_project_fail_closed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with context.session_factory.begin() as session:
        session_record = (
            session.query(SessionRecord)
            .filter_by(
                user_id=context.member_user_id,
                status=SessionStatus.ACTIVE.value,
            )
            .one()
        )
        session_record.status = SessionStatus.REVOKED.value
        session_record.revoked_at = NOW

    revoked = context.member_client.get(
        _create_path(context.active_project_id)
    )
    inactive_get = context.admin_client.get(
        _create_path(context.archived_project_id)
    )
    inactive_post = context.admin_client.post(
        _create_path(context.archived_project_id),
        headers={
            "Origin": ORIGIN,
            "Idempotency-Key": "calibration-archived-0001",
        },
        json=_create_payload(),
    )

    assert revoked.status_code == 401
    assert inactive_get.status_code == 404
    assert inactive_post.status_code == 404
    assert context.queue.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed", [0.91, 0.90]),
        ("predicted", [0.90, 0.89]),
        ("path", r"C:\untrusted\calibration.parquet"),
        ("source_identity_sha256", "f" * 64),
        ("artifact_manifest_sha256", "e" * 64),
    ],
)
def test_post_rejects_caller_supplied_values_paths_and_hashes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    context = _context(tmp_path)

    response = context.admin_client.post(
        _create_path(context.active_project_id),
        headers={
            "Origin": ORIGIN,
            "Idempotency-Key": f"calibration-injected-{field}",
        },
        json={**_create_payload(), field: value},
    )

    assert response.status_code == 422
    assert context.queue.calls == []
    assert context.service.create_calls == []


def test_repeated_key_returns_existing_materialization_and_identity_only_receipt(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    headers = {
        "Origin": ORIGIN,
        "Idempotency-Key": "calibration-repeat-0001",
    }
    path = _create_path(context.active_project_id)

    first = context.admin_client.post(path, headers=headers, json=_create_payload())
    repeated = context.admin_client.post(
        path,
        headers=headers,
        json=_create_payload(),
    )

    assert first.status_code == 202, first.text
    assert repeated.status_code == 202, repeated.text
    assert first.json()["materialization"] == repeated.json()["materialization"]
    assert first.json()["materialization"]["materialization_id"] == (
        CREATED_MATERIALIZATION_ID
    )
    assert first.json()["dispatch"]["materialization_id"] == (
        CREATED_MATERIALIZATION_ID
    )
    assert repeated.json()["dispatch"]["materialization_id"] == (
        CREATED_MATERIALIZATION_ID
    )
    assert first.json()["dispatch"]["task_id"] != repeated.json()["dispatch"]["task_id"]
    assert context.queue.calls == [
        {"materialization_id": CREATED_MATERIALIZATION_ID},
        {"materialization_id": CREATED_MATERIALIZATION_ID},
    ]
    assert len(context.service.create_calls) == 2
    for verified_context, request, key in context.service.create_calls:
        assert verified_context.project_id == context.active_project_id
        assert verified_context.actor_role is UserRole.ADMIN
        assert request.model_dump(mode="json") == {
            "project_id": context.active_project_id,
            **_create_payload(),
        }
        assert key == "calibration-repeat-0001"
