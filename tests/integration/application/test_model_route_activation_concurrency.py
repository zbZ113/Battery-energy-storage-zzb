from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import cast

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quanxin_life.application import model_route_activation as activation_module
from quanxin_life.application.model_route_activation import (
    GENESIS_EVENT_SHA256,
    ModelRouteActivationStateError,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from tests.integration.application.test_model_route_activation_service import (
    NOW,
    _activate,
    _Context,
)

SessionScope = Callable[[SessionFactory], AbstractContextManager[Session]]
pytest_plugins = (
    "tests.integration.application.test_model_route_activation_service",
)


def test_identical_concurrent_activation_recovers_the_verified_event(
    context: _Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _activate(
        context,
        context.first,
        idempotency_key="concurrent-activation",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    original_scope = cast(SessionScope, session_scope)
    calls = 0

    @contextmanager
    def conflict_once(factory: SessionFactory) -> Iterator[Session]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IntegrityError("INSERT", {}, RuntimeError("unique constraint"))
        with original_scope(factory) as session:
            yield session

    monkeypatch.setattr(activation_module, "session_scope", conflict_once)

    recovered = _activate(
        context,
        context.first,
        idempotency_key="concurrent-activation",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )

    assert recovered == expected
    assert calls == 2
    assert context.source.resolve_calls == 2


def test_concurrent_recovery_fails_closed_when_trusted_candidate_disappears(
    context: _Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate(
        context,
        context.first,
        idempotency_key="concurrent-source-recheck",
        expected_previous=GENESIS_EVENT_SHA256,
        occurred_at=NOW,
    )
    context.source._by_id.pop(context.first.artifact_id)
    original_scope = cast(SessionScope, session_scope)
    calls = 0

    @contextmanager
    def conflict_once(factory: SessionFactory) -> Iterator[Session]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IntegrityError("INSERT", {}, RuntimeError("unique constraint"))
        with original_scope(factory) as session:
            yield session

    monkeypatch.setattr(activation_module, "session_scope", conflict_once)

    with pytest.raises(
        ModelRouteActivationStateError,
        match="verified Advanced candidate source is unavailable",
    ):
        _activate(
            context,
            context.first,
            idempotency_key="concurrent-source-recheck",
            expected_previous=GENESIS_EVENT_SHA256,
            occurred_at=NOW,
        )

    assert calls == 2
