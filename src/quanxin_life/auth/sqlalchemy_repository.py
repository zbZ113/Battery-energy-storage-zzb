"""SQLAlchemy transaction boundary for authentication state."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from quanxin_life.core import SessionStatus
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import SessionRecord, User


class AuthTransaction(Protocol):
    def find_user_by_username(self, username: str) -> User | None: ...

    def find_user_by_id(self, user_id: str) -> User | None: ...

    def find_session_by_token_hash(self, token_hash: str) -> SessionRecord | None: ...

    def lock_session_and_user_by_token_hash(
        self, token_hash: str
    ) -> tuple[User | None, SessionRecord | None]: ...

    def add_session(self, record: SessionRecord) -> None: ...

    def revoke_session(self, session_id: str, *, now: datetime) -> None: ...

    def revoke_all_user_sessions(self, user_id: str, *, now: datetime) -> None: ...

    def update_credential(
        self,
        user_id: str,
        credential_hash: str,
        *,
        must_change: bool,
        now: datetime,
    ) -> None: ...


class AuthTransactionFactory(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[AuthTransaction]: ...


class SqlAlchemyAuthTransaction:
    """Query/write adapter that intentionally never commits on its own."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_user_by_username(self, username: str) -> User | None:
        return self._session.scalar(select(User).where(User.username == username))

    def find_user_by_id(self, user_id: str) -> User | None:
        return self._session.get(User, user_id)

    def find_session_by_token_hash(self, token_hash: str) -> SessionRecord | None:
        return self._session.scalar(
            select(SessionRecord).where(SessionRecord.token_hash == token_hash)
        )

    def lock_session_and_user_by_token_hash(
        self, token_hash: str
    ) -> tuple[User | None, SessionRecord | None]:
        """Serialize password rotation per user and refresh the source session.

        The first lightweight lookup discovers the user.  The user row is then
        locked before the session is re-read with ``populate_existing``.  A
        concurrent password rotation therefore observes the first transaction's
        revoked session after it acquires the lock and cannot also succeed.
        """

        user_id = self._session.scalar(
            select(SessionRecord.user_id).where(SessionRecord.token_hash == token_hash)
        )
        if user_id is None:
            return None, None
        user = self._session.scalar(
            select(User).where(User.id == user_id).with_for_update()
        )
        if user is None:
            return None, None
        record = self._session.scalar(
            select(SessionRecord)
            .where(SessionRecord.token_hash == token_hash)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return user, record

    def add_session(self, record: SessionRecord) -> None:
        self._session.add(record)
        self._session.flush()

    def revoke_session(self, session_id: str, *, now: datetime) -> None:
        self._session.execute(
            update(SessionRecord)
            .where(
                SessionRecord.id == session_id,
                SessionRecord.status == SessionStatus.ACTIVE.value,
            )
            .values(status=SessionStatus.REVOKED.value, revoked_at=now)
        )

    def revoke_all_user_sessions(self, user_id: str, *, now: datetime) -> None:
        self._session.execute(
            update(SessionRecord)
            .where(
                SessionRecord.user_id == user_id,
                SessionRecord.status == SessionStatus.ACTIVE.value,
            )
            .values(status=SessionStatus.REVOKED.value, revoked_at=now)
        )

    def update_credential(
        self,
        user_id: str,
        credential_hash: str,
        *,
        must_change: bool,
        now: datetime,
    ) -> None:
        user = self._session.get(User, user_id)
        if user is None:
            raise LookupError("authenticated user no longer exists")
        user.credential_hash = credential_hash
        user.must_change_credential = must_change
        user.updated_at = now
        self._session.flush()


class SqlAlchemyAuthTransactionFactory:
    """Open one atomic SQLAlchemy transaction for each service operation."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    @contextmanager
    def transaction(self) -> Iterator[AuthTransaction]:
        with session_scope(self._session_factory) as session:
            yield SqlAlchemyAuthTransaction(session)


__all__ = [
    "AuthTransaction",
    "AuthTransactionFactory",
    "SqlAlchemyAuthTransaction",
    "SqlAlchemyAuthTransactionFactory",
]
