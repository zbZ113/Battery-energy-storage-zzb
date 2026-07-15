from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.exc import InvalidRequestError, StatementError
from sqlalchemy.orm import Session

from quanxin_life.persistence.database import (
    DatabaseConfig,
    create_engine_from_config,
    create_session_factory,
    session_scope,
)
from quanxin_life.persistence.models import Base, User


def test_engine_factory_uses_only_the_explicit_database_url() -> None:
    config = DatabaseConfig(url="sqlite+pysqlite:///:memory:")

    engine = create_engine_from_config(config)

    try:
        assert isinstance(engine, Engine)
        assert str(engine.url) == config.url
    finally:
        engine.dispose()


def test_database_config_rejects_a_missing_url() -> None:
    with pytest.raises(ValueError, match="database URL"):
        DatabaseConfig(url="  ")


@pytest.fixture
def database_engine() -> Iterator[Engine]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def test_session_scope_commits_and_closes(database_engine: Engine) -> None:
    factory = create_session_factory(database_engine)
    created_at = datetime.now(UTC)

    with session_scope(factory) as session:
        session.add(
            User(
                id="user-1",
                username="member@example.test",
                credential_hash="argon2id-test-hash",
                must_change_credential=True,
                role="MEMBER",
                status="ACTIVE",
                created_at=created_at,
                updated_at=created_at,
            )
        )

    with factory() as verification_session:
        assert verification_session.scalar(select(func.count()).select_from(User)) == 1

    with pytest.raises(InvalidRequestError, match="permanently closed"):
        session.execute(select(User))


def test_session_scope_rolls_back_and_closes_on_error(database_engine: Engine) -> None:
    factory = create_session_factory(database_engine)
    created_at = datetime.now(UTC)

    with pytest.raises(RuntimeError, match="force rollback"), session_scope(factory) as session:
        session.add(
            User(
                    id="user-rollback",
                    username="rollback@example.test",
                    credential_hash="argon2id-test-hash",
                    must_change_credential=True,
                    role="MEMBER",
                status="ACTIVE",
                created_at=created_at,
                updated_at=created_at,
            )
        )
        raise RuntimeError("force rollback")

    with factory() as verification_session:
        assert verification_session.scalar(select(func.count()).select_from(User)) == 0

    assert isinstance(session, Session)
    with pytest.raises(InvalidRequestError, match="permanently closed"):
        session.execute(select(User))


def test_timestamps_are_normalized_to_aware_utc(database_engine: Engine) -> None:
    factory = create_session_factory(database_engine)
    source_time = datetime(2026, 7, 15, 20, 0, tzinfo=timezone(timedelta(hours=8)))

    with session_scope(factory) as session:
        session.add(
            User(
                id="user-timezone",
                username="timezone@example.test",
                credential_hash="argon2id-test-hash",
                must_change_credential=True,
                role="MEMBER",
                status="ACTIVE",
                created_at=source_time,
                updated_at=source_time,
            )
        )

    with factory() as verification_session:
        stored = verification_session.get(User, "user-timezone")

    assert stored is not None
    assert stored.created_at == source_time.astimezone(UTC)
    assert stored.created_at.tzinfo is UTC


def test_naive_timestamps_are_rejected(database_engine: Engine) -> None:
    factory = create_session_factory(database_engine)
    naive_time = datetime(2026, 7, 15, 20, 0)

    with pytest.raises(StatementError, match="timezone"), session_scope(factory) as session:
        session.add(
            User(
                id="user-naive-time",
                username="naive@example.test",
                credential_hash="argon2id-test-hash",
                must_change_credential=True,
                role="MEMBER",
                status="ACTIVE",
                created_at=naive_time,
                updated_at=naive_time,
            )
        )
