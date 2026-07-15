from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from quanxin_life.auth import Argon2idPasswordHasher, PasswordPolicy
from quanxin_life.auth.service import (
    AccountUnavailableError,
    AuthenticationRequiredError,
    AuthService,
    InvalidCredentialsError,
)
from quanxin_life.auth.sqlalchemy_repository import (
    SqlAlchemyAuthTransaction,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import SessionStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import SessionRecord, User

NOW = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
USERNAME = "student@example.test"
TEMPORARY_PASSWORD = "temporary passphrase 2026"
NEW_PASSWORD = "new private passphrase 2026"


@pytest.fixture
def auth_context(tmp_path: Path) -> tuple[AuthService, object, str]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    user_id = str(uuid4())

    with session_factory.begin() as session:
        session.add(
            User(
                id=user_id,
                username=USERNAME,
                credential_hash=hasher.hash_password(TEMPORARY_PASSWORD),
                must_change_credential=True,
                role=UserRole.MEMBER.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )

    service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    return service, session_factory, user_id


def test_login_persists_only_token_hash_and_authenticates(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, session_factory, user_id = auth_context

    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)
    principal = service.authenticate(grant.raw_token, now=NOW + timedelta(minutes=1))

    assert principal.user_id == user_id
    assert principal.role is UserRole.MEMBER
    assert principal.must_change_password is True
    assert grant.raw_token.get_secret_value() not in repr(grant)

    with session_factory() as session:  # type: ignore[operator]
        record = session.scalar(select(SessionRecord))
        assert record is not None
        assert record.token_hash != grant.raw_token.get_secret_value()
        assert len(record.token_hash) == 64
        assert record.status == SessionStatus.ACTIVE.value


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("missing@example.test", TEMPORARY_PASSWORD),
        (USERNAME, "wrong passphrase 2026"),
    ],
)
def test_login_uses_one_generic_error_for_unknown_user_and_wrong_password(
    auth_context: tuple[AuthService, object, str],
    username: str,
    password: str,
) -> None:
    service, _, _ = auth_context

    with pytest.raises(InvalidCredentialsError, match="invalid credentials"):
        service.login(username=username, password=password, now=NOW)


def test_authenticate_rejects_expired_or_disabled_accounts(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, session_factory, user_id = auth_context
    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)

    with pytest.raises(AuthenticationRequiredError):
        service.authenticate(grant.raw_token, now=NOW + timedelta(hours=13))

    with session_factory.begin() as session:  # type: ignore[union-attr]
        user = session.get(User, user_id)
        assert user is not None
        user.status = UserStatus.DISABLED.value

    with pytest.raises(AccountUnavailableError):
        service.authenticate(grant.raw_token, now=NOW + timedelta(minutes=1))

    with pytest.raises(InvalidCredentialsError, match="invalid credentials"):
        service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)


def test_logout_is_idempotent_and_revokes_the_session(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, _, _ = auth_context
    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)

    service.logout(grant.raw_token, now=NOW + timedelta(minutes=1))
    service.logout(grant.raw_token, now=NOW + timedelta(minutes=2))
    service.logout(SecretStr("not-a-valid-token"), now=NOW + timedelta(minutes=3))

    with pytest.raises(AuthenticationRequiredError):
        service.authenticate(grant.raw_token, now=NOW + timedelta(minutes=4))


def test_change_password_revokes_all_old_sessions_and_issues_a_new_one(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, _, _ = auth_context
    first = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)
    second = service.login(
        username=USERNAME,
        password=TEMPORARY_PASSWORD,
        now=NOW + timedelta(minutes=1),
    )

    replacement = service.change_password(
        first.raw_token,
        current_password=TEMPORARY_PASSWORD,
        new_password=NEW_PASSWORD,
        now=NOW + timedelta(minutes=2),
    )

    assert replacement.principal.must_change_password is False
    for old_token in (first.raw_token, second.raw_token):
        with pytest.raises(AuthenticationRequiredError):
            service.authenticate(old_token, now=NOW + timedelta(minutes=3))
    assert service.authenticate(
        replacement.raw_token, now=NOW + timedelta(minutes=3)
    ).must_change_password is False
    with pytest.raises(InvalidCredentialsError):
        service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)
    service.login(username=USERNAME, password=NEW_PASSWORD, now=NOW + timedelta(minutes=4))


def test_change_password_uses_the_locked_session_and_user_lookup(
    auth_context: tuple[AuthService, object, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, _ = auth_context
    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)
    calls: list[str] = []
    original = SqlAlchemyAuthTransaction.lock_session_and_user_by_token_hash

    def track_locked_lookup(
        transaction: SqlAlchemyAuthTransaction, token_hash: str
    ) -> tuple[User | None, SessionRecord | None]:
        calls.append(token_hash)
        return original(transaction, token_hash)

    monkeypatch.setattr(
        SqlAlchemyAuthTransaction,
        "lock_session_and_user_by_token_hash",
        track_locked_lookup,
    )

    service.change_password(
        grant.raw_token,
        current_password=TEMPORARY_PASSWORD,
        new_password=NEW_PASSWORD,
        now=NOW + timedelta(minutes=1),
    )

    assert len(calls) == 1
    assert len(calls[0]) == 64


def test_failed_password_change_rolls_back_credentials_and_sessions(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, _, _ = auth_context
    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)

    with pytest.raises(InvalidCredentialsError):
        service.change_password(
            grant.raw_token,
            current_password="wrong current passphrase",
            new_password=NEW_PASSWORD,
            now=NOW + timedelta(minutes=1),
        )

    assert service.authenticate(
        grant.raw_token, now=NOW + timedelta(minutes=2)
    ).must_change_password is True
    service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)


def test_password_change_rolls_back_if_new_session_cannot_be_persisted(
    auth_context: tuple[AuthService, object, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, _ = auth_context
    grant = service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)

    def fail_after_flush(
        transaction: SqlAlchemyAuthTransaction, record: SessionRecord
    ) -> None:
        del transaction, record
        raise RuntimeError("simulated session persistence failure")

    monkeypatch.setattr(SqlAlchemyAuthTransaction, "add_session", fail_after_flush)
    with pytest.raises(RuntimeError, match="simulated session"):
        service.change_password(
            grant.raw_token,
            current_password=TEMPORARY_PASSWORD,
            new_password=NEW_PASSWORD,
            now=NOW + timedelta(minutes=1),
        )
    monkeypatch.undo()

    assert service.authenticate(
        grant.raw_token, now=NOW + timedelta(minutes=2)
    ).must_change_password is True
    service.login(username=USERNAME, password=TEMPORARY_PASSWORD, now=NOW)


def test_auth_service_rejects_naive_time(
    auth_context: tuple[AuthService, object, str],
) -> None:
    service, _, _ = auth_context

    with pytest.raises(ValueError, match="timezone"):
        service.login(
            username=USERNAME,
            password=TEMPORARY_PASSWORD,
            now=datetime(2026, 7, 15, 8, 0),
        )
