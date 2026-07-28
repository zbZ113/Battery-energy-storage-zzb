from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from deploy.bootstrap_admin import (
    BootstrapAdminConflictError,
    bootstrap_admin,
)
from quanxin_life.auth import Argon2idPasswordHasher
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import User

NOW = datetime(2026, 7, 28, 13, 0, tzinfo=UTC)
USERNAME = "owner@example.test"
PASSWORD = "temporary competition passphrase 2026"


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'bootstrap.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def test_bootstrap_creates_one_forced_rotation_admin_without_plaintext(
    session_factory,
) -> None:
    result = bootstrap_admin(
        session_factory,
        username=USERNAME,
        temporary_password=PASSWORD,
        now=NOW,
    )

    assert result.username == USERNAME
    assert result.created is True
    assert result.must_change_password is True
    assert PASSWORD not in repr(result)
    with session_factory() as session:
        user = session.scalar(select(User))
        assert user is not None
        assert user.role == UserRole.ADMIN.value
        assert user.status == UserStatus.ACTIVE.value
        assert user.must_change_credential is True
        assert PASSWORD not in user.credential_hash
        assert Argon2idPasswordHasher().verify_password(
            user.credential_hash,
            PASSWORD,
        )


def test_bootstrap_is_idempotent_only_for_the_same_unrotated_admin(
    session_factory,
) -> None:
    first = bootstrap_admin(
        session_factory,
        username=USERNAME,
        temporary_password=PASSWORD,
        now=NOW,
    )
    second = bootstrap_admin(
        session_factory,
        username=USERNAME,
        temporary_password=PASSWORD,
        now=NOW,
    )

    assert second.user_id == first.user_id
    assert second.created is False
    with session_factory() as session:
        assert len(tuple(session.scalars(select(User)))) == 1


def test_bootstrap_rejects_other_identity_or_existing_initialized_state(
    session_factory,
) -> None:
    bootstrap_admin(
        session_factory,
        username=USERNAME,
        temporary_password=PASSWORD,
        now=NOW,
    )

    with pytest.raises(BootstrapAdminConflictError):
        bootstrap_admin(
            session_factory,
            username="other@example.test",
            temporary_password=PASSWORD,
            now=NOW,
        )
    with session_factory.begin() as session:
        user = session.scalar(select(User))
        assert user is not None
        user.must_change_credential = False
    with pytest.raises(BootstrapAdminConflictError):
        bootstrap_admin(
            session_factory,
            username=USERNAME,
            temporary_password=PASSWORD,
            now=NOW,
        )
