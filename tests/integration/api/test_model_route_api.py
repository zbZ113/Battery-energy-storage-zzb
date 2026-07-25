from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.model_routes import create_model_route_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.model_artifact_catalog import (
    VerifiedModelArtifactRegistration,
)
from quanxin_life.application.model_route_activation import (
    GENESIS_EVENT_SHA256,
    ModelRouteActivationService,
)
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import ModelArtifact, ModelManifest, Project, User
from tests.integration.application.test_model_artifact_catalog_service import (
    _advanced_registrations,
)
from tests.integration.application.test_model_route_activation_service import (
    _second_candidate,
)

NOW = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
PASSWORD = "RouteLedgerPassphrase-2026!"


class _Source:
    def __init__(self, registrations: tuple[VerifiedModelArtifactRegistration, ...]) -> None:
        self._by_id = {item.artifact_id: item for item in registrations}

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration:
        return self._by_id[artifact_id]


@dataclass(frozen=True)
class _ApiContext:
    client: TestClient
    project_id: str
    first: VerifiedModelArtifactRegistration
    second: VerifiedModelArtifactRegistration
    admin_username: str
    member_username: str

    def login(self, username: str) -> None:
        response = self.client.post(
            "/v1/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": username, "password": PASSWORD},
        )
        assert response.status_code == 200


def _context(tmp_path: Path) -> _ApiContext:
    first = next(
        item
        for item in _advanced_registrations()
        if item.metadata.advanced_provenance is not None
        and item.metadata.advanced_provenance.routes[0].task == "RUL"
        and item.metadata.advanced_provenance.routes[0].role == "DEFAULT"
    )
    second = _second_candidate(first)
    registrations = (first, second)
    source = _Source(registrations)
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'model-routes.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    admin_id = str(uuid4())
    member_id = str(uuid4())
    project_id = str(uuid4())
    admin_username = f"route-admin-{admin_id[:8]}@example.test"
    member_username = f"route-member-{member_id[:8]}@example.test"
    with session_factory.begin() as session:
        for user_id, username, role in (
            (admin_id, admin_username, UserRole.ADMIN),
            (member_id, member_username, UserRole.MEMBER),
        ):
            session.add(
                User(
                    id=user_id,
                    username=username,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.add(
            Project(
                id=project_id,
                owner_user_id=member_id,
                name="model route API",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for registration in registrations:
            session.add(
                ModelArtifact(
                    id=registration.artifact_id,
                    project_id=project_id,
                    artifact_format=registration.artifact_format,
                    object_uri=registration.object_uri,
                    sha256=registration.artifact_sha256,
                    status="VERIFIED",
                    created_at=NOW,
                )
            )
            session.add(
                ModelManifest(
                    id=str(uuid4()),
                    artifact_id=registration.artifact_id,
                    model_version=registration.model_version,
                    manifest_uri=registration.manifest_uri,
                    manifest_sha256=registration.manifest_sha256,
                    metadata_json=registration.metadata.model_dump(mode="json"),
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
    route_adapter = create_model_route_http_adapter(
        ModelRouteActivationService(session_factory, source=source),
        auth_adapter=auth_adapter,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        model_route_adapter=route_adapter,
    )
    context = _ApiContext(
        client=TestClient(app, base_url="https://api.example.test"),
        project_id=project_id,
        first=first,
        second=second,
        admin_username=admin_username,
        member_username=member_username,
    )
    context.login(admin_username)
    return context


def _activation_payload(
    context: _ApiContext,
    artifact: VerifiedModelArtifactRegistration,
    *,
    expected_head: str,
    reason: str = "人工批准正式 RUL 默认路由",
) -> dict[str, object]:
    return {
        "project_id": context.project_id,
        "task": "RUL",
        "cutoff_cycle": 20,
        "role": "DEFAULT",
        "artifact_id": artifact.artifact_id,
        "reason": reason,
        "expected_previous_event_sha256": expected_head,
    }


def test_activation_requires_trusted_origin_and_is_idempotent(tmp_path: Path) -> None:
    context = _context(tmp_path)
    payload = _activation_payload(
        context,
        context.first,
        expected_head=GENESIS_EVENT_SHA256,
    )

    missing_origin = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Idempotency-Key": "route-activate-0001"},
        json=payload,
    )
    missing_key = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN},
        json=payload,
    )
    first = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-activate-0001"},
        json=payload,
    )
    repeated = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-activate-0001"},
        json=payload,
    )

    assert missing_origin.status_code == 403
    assert missing_key.status_code == 422
    assert first.status_code == 201
    assert repeated.status_code == 201
    assert repeated.json() == first.json()
    assert first.json()["sequence_number"] == 1
    assert first.json()["decision_type"] == "ACTIVATE"
    assert first.json()["artifact_id"] == context.first.artifact_id


def test_activation_is_admin_only_and_rejects_caller_provenance(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.login(context.member_username)
    payload = _activation_payload(
        context,
        context.first,
        expected_head=GENESIS_EVENT_SHA256,
    )
    headers = {"Origin": ORIGIN, "Idempotency-Key": "route-member-0001"}

    forbidden = context.client.post(
        "/v1/admin/model-routes/activations",
        headers=headers,
        json=payload,
    )
    context.login(context.admin_username)
    injected = context.client.post(
        "/v1/admin/model-routes/activations",
        headers=headers,
        json={
            **payload,
            "artifact_sha256": "f" * 64,
            "status": "ACTIVE",
            "object_uri": "file:///caller-selected/model.safetensors",
        },
    )

    assert forbidden.status_code == 403
    assert injected.status_code == 422


def test_switch_then_rollback_derives_artifact_from_historical_event(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    headers = {"Origin": ORIGIN, "Idempotency-Key": "route-first-0001"}
    first = context.client.post(
        "/v1/admin/model-routes/activations",
        headers=headers,
        json=_activation_payload(
            context,
            context.first,
            expected_head=GENESIS_EVENT_SHA256,
        ),
    )
    assert first.status_code == 201
    second = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-second-0001"},
        json=_activation_payload(
            context,
            context.second,
            expected_head=first.json()["event_sha256"],
            reason="切换到替代候选",
        ),
    )
    assert second.status_code == 201
    rollback_payload = {
        "project_id": context.project_id,
        "task": "RUL",
        "cutoff_cycle": 20,
        "role": "DEFAULT",
        "target_activation_event_id": first.json()["event_id"],
        "reason": "替代候选上线复核失败, 恢复先前版本",
        "expected_previous_event_sha256": second.json()["event_sha256"],
    }
    injected = context.client.post(
        "/v1/admin/model-routes/rollbacks",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-rollback-bad-0001"},
        json={**rollback_payload, "artifact_id": context.second.artifact_id},
    )
    rollback = context.client.post(
        "/v1/admin/model-routes/rollbacks",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-rollback-0001"},
        json=rollback_payload,
    )

    assert injected.status_code == 422
    assert rollback.status_code == 201
    assert rollback.json()["decision_type"] == "ROLLBACK"
    assert rollback.json()["artifact_id"] == context.first.artifact_id
    assert rollback.json()["rollback_target_event_id"] == first.json()["event_id"]


def test_history_is_verified_and_stale_head_returns_conflict(tmp_path: Path) -> None:
    context = _context(tmp_path)
    first = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-history-0001"},
        json=_activation_payload(
            context,
            context.first,
            expected_head=GENESIS_EVENT_SHA256,
        ),
    )
    assert first.status_code == 201
    stale = context.client.post(
        "/v1/admin/model-routes/activations",
        headers={"Origin": ORIGIN, "Idempotency-Key": "route-stale-0001"},
        json=_activation_payload(
            context,
            context.second,
            expected_head=GENESIS_EVENT_SHA256,
        ),
    )
    history = context.client.get(
        "/v1/model-routes/activation-events",
        params={
            "project_id": context.project_id,
            "task": "RUL",
            "cutoff_cycle": 20,
            "role": "DEFAULT",
        },
    )

    assert stale.status_code == 409
    assert stale.json()["detail"] == "model_route_state_conflict"
    assert history.status_code == 200
    assert history.json() == [first.json()]
