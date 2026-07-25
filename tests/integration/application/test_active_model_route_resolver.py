from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from quanxin_life.application.model_route_activation import (
    GENESIS_EVENT_SHA256,
    ModelRouteActivationNotFoundError,
    ModelRouteActivationStateError,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ModelRouteDecisionType,
    UserRole,
    sha256_canonical,
)
from quanxin_life.persistence.models import ModelRouteActivationEvent, Project
from tests.integration.application.test_model_route_activation_service import (
    NOW,
    _activate,
    _Context,
    _principal,
)
from tests.integration.application.test_model_route_activation_service import (
    context as activation_context,
)


@pytest.fixture
def context(tmp_path: Path) -> _Context:
    builder = activation_context.__wrapped__  # type: ignore[attr-defined]
    return builder(tmp_path)


def _resolve(context: _Context):  # type: ignore[no-untyped-def]
    return context.service.resolve_verified_active_model_route(
        context.member,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    )


def test_empty_route_fails_closed_without_catalog_fallback(context: _Context) -> None:
    with pytest.raises(ModelRouteActivationNotFoundError, match="route"):
        _resolve(context)

    assert context.source.resolve_calls == 0


def test_single_activation_resolves_exact_candidate_and_stream_head(
    context: _Context,
) -> None:
    event = _activate(
        context,
        context.first,
        idempotency_key="resolve-first",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )

    resolved = _resolve(context)

    provenance = context.first.metadata.advanced_provenance
    assert provenance is not None
    assert resolved.artifact == context.first
    assert resolved.route_provenance == provenance.routes[0]
    assert resolved.project_id == context.project_id
    assert resolved.task is AdvancedModelTask.RUL
    assert resolved.cutoff_cycle == 20
    assert resolved.role is AdvancedModelRouteRole.DEFAULT
    assert resolved.decision_event_id == event.event_id
    assert resolved.decision_type is ModelRouteDecisionType.ACTIVATE
    assert resolved.rollback_target_event_id is None
    assert resolved.ledger_sequence_number == 1
    assert resolved.ledger_head_sha256 == event.event_sha256
    assert resolved.route_provenance_sha256 == event.route_provenance_sha256
    assert provenance.lifecycle_status == "REGISTERED_CANDIDATE"
    assert provenance.activation_status == "NOT_ACTIVATED"

    with context.session_factory() as session:
        head = session.execute(
            text(
                "SELECT head_event_id, head_sequence, head_event_sha256 "
                "FROM model_route_activation_stream_heads "
                "WHERE project_id = :project_id AND task = 'RUL' "
                "AND cutoff_cycle = 20 AND route_role = 'DEFAULT'"
            ),
            {"project_id": context.project_id},
        ).one()
    assert tuple(head) == (event.event_id, 1, event.event_sha256)


def test_switch_and_rollback_resolve_from_latest_decision(context: _Context) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="resolve-a",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    second = _activate(
        context,
        context.second,
        idempotency_key="resolve-b",
        expected_previous=first.event_sha256,
        occurred_at=NOW.replace(minute=1),
    )
    switched = _resolve(context)
    rollback = context.service.rollback(
        context.admin,
        project_id=context.project_id,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
        target_activation_event_id=first.event_id,
        reason="替代候选不符合上线要求, 恢复 A",
        idempotency_key="resolve-rollback-a",
        expected_previous_event_sha256=second.event_sha256,
        occurred_at=NOW.replace(minute=2),
    )
    restored = _resolve(context)

    assert switched.artifact.artifact_id == context.second.artifact_id
    assert switched.decision_event_id == second.event_id
    assert restored.artifact.artifact_id == context.first.artifact_id
    assert restored.decision_event_id == rollback.event_id
    assert restored.decision_type is ModelRouteDecisionType.ROLLBACK
    assert restored.rollback_target_event_id == first.event_id
    assert restored.ledger_sequence_number == 3
    assert restored.ledger_head_sha256 == rollback.event_sha256


def test_resolver_rejects_route_change_during_trusted_source_verification(
    context: _Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _activate(
        context,
        context.first,
        idempotency_key="resolve-race-a",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    original_resolve = context.source.resolve
    switched = False

    def resolve_while_switching(artifact_id: str):  # type: ignore[no-untyped-def]
        nonlocal switched
        if not switched:
            switched = True
            _activate(
                context,
                context.second,
                idempotency_key="resolve-race-b",
                expected_previous=first.event_sha256,
                occurred_at=NOW.replace(minute=1),
            )
        return original_resolve(artifact_id)

    monkeypatch.setattr(context.source, "resolve", resolve_while_switching)

    with pytest.raises(ModelRouteActivationStateError, match=r"changed|head|concurrent"):
        _resolve(context)

    monkeypatch.setattr(context.source, "resolve", original_resolve)
    assert _resolve(context).artifact.artifact_id == context.second.artifact_id


def test_resolver_never_falls_back_across_route_coordinates(context: _Context) -> None:
    _activate(
        context,
        context.first,
        idempotency_key="resolve-route-isolation",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    calls_before = context.source.resolve_calls

    with pytest.raises(ModelRouteActivationNotFoundError, match="route"):
        context.service.resolve_verified_active_model_route(
            context.member,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=50,
            role=AdvancedModelRouteRole.DEFAULT,
        )

    assert context.source.resolve_calls == calls_before


def test_inactive_project_cannot_resolve_prior_activation(context: _Context) -> None:
    _activate(
        context,
        context.first,
        idempotency_key="resolve-before-archive",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    with context.session_factory.begin() as session:
        project = session.get(Project, context.project_id)
        assert project is not None
        project.status = "ARCHIVED"

    with pytest.raises(ModelRouteActivationNotFoundError, match="project"):
        _resolve(context)


def test_invisible_project_and_missing_trusted_source_fail_closed(
    context: _Context,
) -> None:
    _activate(
        context,
        context.first,
        idempotency_key="resolve-source-visibility",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    outsider = _principal(UserRole.MEMBER)
    with pytest.raises(ModelRouteActivationNotFoundError, match="project"):
        context.service.resolve_verified_active_model_route(
            outsider,
            project_id=context.project_id,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
        )

    context.source._by_id.pop(context.first.artifact_id)
    with pytest.raises(ModelRouteActivationStateError, match="source"):
        _resolve(context)


def test_resolver_rejects_self_consistent_event_snapshot_not_matching_source(
    context: _Context,
) -> None:
    event = _activate(
        context,
        context.first,
        idempotency_key="resolve-before-resigned-tamper",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    payload = event.model_dump(mode="json", exclude={"event_sha256"})
    payload["artifact_sha256"] = "f" * 64
    resigned_sha = sha256_canonical(payload)
    with context.session_factory.begin() as session:
        row = session.get(ModelRouteActivationEvent, event.event_id)
        assert row is not None
        row.artifact_sha256 = "f" * 64
        row.event_sha256 = resigned_sha
        session.execute(
            text(
                "UPDATE model_route_activation_stream_heads "
                "SET head_event_sha256 = :head_sha "
                "WHERE project_id = :project_id AND task = 'RUL' "
                "AND cutoff_cycle = 20 AND route_role = 'DEFAULT'"
            ),
            {"head_sha": resigned_sha, "project_id": context.project_id},
        )

    with pytest.raises(ModelRouteActivationStateError, match=r"snapshot|source"):
        _resolve(context)


def test_tail_event_delete_is_rejected_or_detected_by_stream_head(
    context: _Context,
) -> None:
    event = _activate(
        context,
        context.first,
        idempotency_key="resolve-tail-delete",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )

    try:
        with context.session_factory.begin() as session:
            row = session.get(ModelRouteActivationEvent, event.event_id)
            assert row is not None
            session.delete(row)
    except IntegrityError:
        assert _resolve(context).decision_event_id == event.event_id
    else:
        with pytest.raises(ModelRouteActivationStateError, match=r"head|history"):
            _resolve(context)


def test_corrupted_stream_head_blocks_resolution(context: _Context) -> None:
    _activate(
        context,
        context.first,
        idempotency_key="resolve-head-tamper",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    with context.session_factory.begin() as session:
        session.execute(
            text(
                "UPDATE model_route_activation_stream_heads "
                "SET head_sequence = 99 "
                "WHERE project_id = :project_id AND task = 'RUL' "
                "AND cutoff_cycle = 20 AND route_role = 'DEFAULT'"
            ),
            {"project_id": context.project_id},
        )

    with pytest.raises(ModelRouteActivationStateError, match=r"head|ledger"):
        _resolve(context)
