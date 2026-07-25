from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.application.model_artifact_catalog import (
    VerifiedModelArtifactRegistration,
)
from quanxin_life.application.model_route_activation import (
    GENESIS_EVENT_SHA256,
    ModelRouteActivationAccessError,
    ModelRouteActivationEventRecord,
    ModelRouteActivationService,
    ModelRouteActivationStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    Project,
    User,
)
from tests.integration.application.test_model_artifact_catalog_service import (
    _advanced_registrations,
)

NOW = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)


class _Source:
    def __init__(self, registrations: tuple[VerifiedModelArtifactRegistration, ...]) -> None:
        self._by_id = {item.artifact_id: item for item in registrations}
        self.resolve_calls = 0

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration:
        self.resolve_calls += 1
        try:
            return self._by_id[artifact_id]
        except KeyError as exc:
            raise KeyError(artifact_id) from exc


@dataclass(frozen=True)
class _Context:
    service: ModelRouteActivationService
    admin: AuthPrincipal
    alternate_admin: AuthPrincipal
    member: AuthPrincipal
    project_id: str
    first: VerifiedModelArtifactRegistration
    second: VerifiedModelArtifactRegistration
    source: _Source
    session_factory: SessionFactory


@pytest.fixture
def context(tmp_path: Path) -> _Context:
    first = next(
        item
        for item in _advanced_registrations()
        if item.metadata.advanced_provenance is not None
        and item.metadata.advanced_provenance.routes[0].role == "DEFAULT"
    )
    second = _second_candidate(first)
    source = _Source((first, second))
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'activation.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    admin = _principal(UserRole.ADMIN)
    alternate_admin = _principal(UserRole.ADMIN)
    member = _principal(UserRole.MEMBER)
    project_id = str(uuid4())
    with factory.begin() as session:
        for principal in (admin, alternate_admin, member):
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.add(
            Project(
                id=project_id,
                owner_user_id=member.user_id,
                name="activation ledger",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for registration in (first, second):
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
    return _Context(
        service=ModelRouteActivationService(factory, source=source),
        admin=admin,
        alternate_admin=alternate_admin,
        member=member,
        project_id=project_id,
        first=first,
        second=second,
        source=source,
        session_factory=factory,
    )


def test_activate_is_manual_hash_chained_and_idempotent(context: _Context) -> None:
    event = _activate(
        context,
        context.first,
        idempotency_key="activate-first",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    repeated = _activate(
        context,
        context.first,
        idempotency_key="activate-first",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )

    assert repeated == event
    assert event.sequence_number == 1
    assert event.previous_event_sha256 == GENESIS_EVENT_SHA256
    assert event.decision_type.value == "ACTIVATE"
    assert event.artifact_id == context.first.artifact_id
    assert event.actor_user_id == context.admin.user_id
    assert event.reason == "批准首个正式候选"
    assert len(event.event_sha256) == 64
    assert context.source.resolve_calls == 2
    assert context.service.list_events(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    ) == (event,)


def test_idempotency_key_cannot_replay_another_administrators_decision(
    context: _Context,
) -> None:
    event = _activate(
        context,
        context.first,
        idempotency_key="actor-bound-key",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )

    with pytest.raises(ModelRouteActivationStateError, match="idempotency"):
        context.service.activate(
            context.alternate_admin,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
            artifact_id=context.first.artifact_id,
            reason=event.reason,
            idempotency_key="actor-bound-key",
            expected_previous_event_sha256=GENESIS_EVENT_SHA256,
            occurred_at=NOW,
        )

    assert context.service.list_events(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    ) == (event,)


def test_activation_switch_and_rollback_only_append_history(context: _Context) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="activate-a",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    second = _activate(
        context,
        context.second,
        idempotency_key="activate-b",
        expected_previous=first.event_sha256,
        occurred_at=NOW.replace(minute=1),
    )
    rollback = context.service.rollback(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
        target_activation_event_id=first.event_id,
        reason="B 候选上线核验失败, 恢复 A",
        idempotency_key="rollback-to-a",
        expected_previous_event_sha256=second.event_sha256,
        occurred_at=NOW.replace(minute=2),
    )

    assert rollback.sequence_number == 3
    assert rollback.previous_event_sha256 == second.event_sha256
    assert rollback.rollback_target_event_id == first.event_id
    assert rollback.artifact_id == first.artifact_id
    assert rollback.decision_type.value == "ROLLBACK"
    history = context.service.list_events(
        context.member,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    )
    assert history == (first, second, rollback)


def test_stale_head_duplicate_key_and_route_mismatch_append_nothing(
    context: _Context,
) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="stable-key",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    with pytest.raises(ModelRouteActivationStateError, match="idempotency"):
        _activate(
            context,
            context.second,
            idempotency_key="stable-key",
            expected_previous=first.event_sha256,
            occurred_at=NOW.replace(minute=1),
        )
    with pytest.raises(ModelRouteActivationStateError, match=r"stale|head"):
        _activate(
            context,
            context.second,
            idempotency_key="stale-head",
            expected_previous=GENESIS_EVENT_SHA256,
            occurred_at=NOW.replace(minute=1),
        )
    with pytest.raises(ModelRouteActivationStateError, match="route"):
        context.service.activate(
            context.admin,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=50,
            role=AdvancedModelRouteRole.POINT_ACCURACY,
            artifact_id=context.second.artifact_id,
            reason="错误路由",
            idempotency_key="wrong-route",
            expected_previous_event_sha256=GENESIS_EVENT_SHA256,
            occurred_at=NOW.replace(minute=1),
        )
    with context.session_factory() as session:
        assert session.query(ModelRouteActivationEvent).count() == 1


def test_only_admin_can_decide_and_persisted_source_mismatch_fails_closed(
    context: _Context,
) -> None:
    with pytest.raises(ModelRouteActivationAccessError):
        context.service.activate(
            context.member,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
            artifact_id=context.first.artifact_id,
            reason="成员无权激活",
            idempotency_key="member-blocked",
            expected_previous_event_sha256=GENESIS_EVENT_SHA256,
            occurred_at=NOW,
        )
    with context.session_factory.begin() as session:
        manifest = session.query(ModelManifest).filter_by(
            artifact_id=context.first.artifact_id
        ).one()
        manifest.manifest_sha256 = "f" * 64
    with pytest.raises(ModelRouteActivationStateError, match=r"persisted|source"):
        _activate(
            context,
            context.first,
            idempotency_key="tampered-source",
            expected_previous=GENESIS_EVENT_SHA256,
            occurred_at=NOW,
        )
    with context.session_factory() as session:
        assert session.query(ModelRouteActivationEvent).count() == 0


def test_history_tampering_is_detected_before_another_append(context: _Context) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="activate-before-tamper",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    with context.session_factory.begin() as session:
        row = session.get(ModelRouteActivationEvent, first.event_id)
        assert row is not None
        row.reason = "tampered"

    with pytest.raises(ModelRouteActivationStateError, match=r"hash|ledger"):
        context.service.list_events(
            context.admin,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
        )
    with pytest.raises(ModelRouteActivationStateError, match=r"hash|ledger"):
        _activate(
            context,
            context.second,
            idempotency_key="blocked-after-tamper",
            expected_previous=first.event_sha256,
            occurred_at=NOW.replace(minute=1),
        )


def test_resigned_rollback_must_still_target_an_earlier_activation(
    context: _Context,
) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="semantic-first",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    second = _activate(
        context,
        context.second,
        idempotency_key="semantic-second",
        expected_previous=first.event_sha256,
        occurred_at=NOW.replace(minute=1),
    )
    rollback = context.service.rollback(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
        target_activation_event_id=first.event_id,
        reason="Restore the earlier verified candidate",
        idempotency_key="semantic-rollback",
        expected_previous_event_sha256=second.event_sha256,
        occurred_at=NOW.replace(minute=2),
    )
    resigned = rollback.model_dump(mode="json", exclude={"event_sha256"})
    resigned["rollback_target_event_id"] = rollback.event_id
    with context.session_factory.begin() as session:
        row = session.get(ModelRouteActivationEvent, rollback.event_id)
        assert row is not None
        row.rollback_target_event_id = rollback.event_id
        row.event_sha256 = sha256_canonical(resigned)

    with pytest.raises(ModelRouteActivationStateError, match=r"rollback|ledger"):
        context.service.list_events(
            context.admin,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
        )


def _activate(
    context: _Context,
    artifact: VerifiedModelArtifactRegistration,
    *,
    idempotency_key: str,
    expected_previous: str,
    occurred_at: datetime,
) -> ModelRouteActivationEventRecord:
    return context.service.activate(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
        artifact_id=artifact.artifact_id,
        reason="批准首个正式候选",
        idempotency_key=idempotency_key,
        expected_previous_event_sha256=expected_previous,
        occurred_at=occurred_at,
    )


def _second_candidate(
    first: VerifiedModelArtifactRegistration,
) -> VerifiedModelArtifactRegistration:
    artifact_id = str(uuid4())
    provenance = first.metadata.advanced_provenance
    assert provenance is not None
    route = provenance.routes[0]
    changed_route = route.model_copy(
        update={
            "run_id": f"{route.run_id}-replacement",
            "checkpoint_manifest_sha256": _digest("replacement-manifest"),
            "checkpoint_model_sha256": _digest("replacement-model"),
            "checkpoint_context_sha256": _digest("replacement-context"),
        }
    )
    return first.model_copy(
        update={
            "artifact_id": artifact_id,
            "object_uri": f"verified-model-artifact://{artifact_id}/bundle",
            "artifact_sha256": _digest("replacement-artifact"),
            "model_version": changed_route.run_id,
            "manifest_uri": f"verified-model-artifact://{artifact_id}/manifest",
            "manifest_sha256": _digest("replacement-catalog-manifest"),
            "metadata": first.metadata.model_copy(
                update={
                    "advanced_provenance": provenance.model_copy(
                        update={"routes": (changed_route,)}
                    )
                }
            ),
        }
    )


def _principal(role: UserRole) -> AuthPrincipal:
    user_id = str(uuid4())
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
