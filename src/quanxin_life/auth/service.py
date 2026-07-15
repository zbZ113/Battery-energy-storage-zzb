"""Transactional authentication use cases with opaque server-side sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import SecretStr

from quanxin_life.auth.contracts import AuthPrincipal, SessionGrant
from quanxin_life.auth.passwords import Argon2idPasswordHasher, PasswordPolicy
from quanxin_life.auth.sqlalchemy_repository import AuthTransaction, AuthTransactionFactory
from quanxin_life.auth.tokens import SessionTokenFactory, hash_session_token
from quanxin_life.core import SessionStatus, UserRole, UserStatus
from quanxin_life.persistence.models import SessionRecord, User


class AuthenticationError(RuntimeError):
    """Base class whose message never includes credentials or tokens."""


class InvalidCredentialsError(AuthenticationError):
    pass


class AuthenticationRequiredError(AuthenticationError):
    pass


class AccountUnavailableError(AuthenticationError):
    pass


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("authentication timestamps must include a timezone")
    return value.astimezone(UTC)


def _canonical_username(username: str) -> str:
    if not isinstance(username, str):
        raise InvalidCredentialsError("invalid credentials")
    value = username.strip().casefold()
    if not value or len(value) > 320:
        raise InvalidCredentialsError("invalid credentials")
    return value


class AuthService:
    """Own login, session authentication, logout, and atomic password rotation."""

    def __init__(
        self,
        *,
        transactions: AuthTransactionFactory,
        password_hasher: Argon2idPasswordHasher,
        password_policy: PasswordPolicy,
        session_ttl: timedelta = timedelta(hours=12),
        token_factory: SessionTokenFactory | None = None,
    ) -> None:
        if session_ttl <= timedelta(0) or session_ttl > timedelta(days=7):
            raise ValueError("session_ttl must be between zero and seven days")
        self._transactions = transactions
        self._password_hasher = password_hasher
        self._password_policy = password_policy
        self._session_ttl = session_ttl
        self._token_factory = token_factory or SessionTokenFactory()
        self._dummy_hash = password_hasher.hash_password(
            "fixed timing defense passphrase that is never a user credential"
        )

    def login(self, *, username: str, password: str, now: datetime) -> SessionGrant:
        current_time = _utc(now)
        canonical_username = _canonical_username(username)
        with self._transactions.transaction() as transaction:
            user = transaction.find_user_by_username(canonical_username)
            candidate_hash = user.credential_hash if user is not None else self._dummy_hash
            verified = self._password_hasher.verify_password(candidate_hash, password)
            if (
                user is None
                or not verified
                or user.status != UserStatus.ACTIVE.value
            ):
                raise InvalidCredentialsError("invalid credentials")
            return self._issue_session(transaction, user=user, now=current_time)

    def authenticate(self, token: SecretStr | str, *, now: datetime) -> AuthPrincipal:
        current_time = _utc(now)
        with self._transactions.transaction() as transaction:
            user, record = self._resolve_session(transaction, token=token, now=current_time)
            return self._principal(user, record)

    def logout(self, token: SecretStr | str | None, *, now: datetime) -> None:
        current_time = _utc(now)
        if token is None:
            return
        try:
            token_hash = hash_session_token(token)
        except ValueError:
            return
        with self._transactions.transaction() as transaction:
            record = transaction.find_session_by_token_hash(token_hash)
            if record is not None:
                transaction.revoke_session(record.id, now=current_time)

    def change_password(
        self,
        token: SecretStr | str,
        *,
        current_password: str,
        new_password: str,
        now: datetime,
    ) -> SessionGrant:
        current_time = _utc(now)
        self._password_policy.validate_password(new_password)
        with self._transactions.transaction() as transaction:
            user, _ = self._resolve_session(
                transaction,
                token=token,
                now=current_time,
                lock_for_password_change=True,
            )
            if not self._password_hasher.verify_password(
                user.credential_hash, current_password
            ):
                raise InvalidCredentialsError("invalid credentials")
            if self._password_hasher.verify_password(user.credential_hash, new_password):
                raise ValueError("new password must differ from the current password")

            new_hash = self._password_hasher.hash_password(new_password)
            transaction.update_credential(
                user.id,
                new_hash,
                must_change=False,
                now=current_time,
            )
            transaction.revoke_all_user_sessions(user.id, now=current_time)
            user.credential_hash = new_hash
            user.must_change_credential = False
            return self._issue_session(transaction, user=user, now=current_time)

    def _resolve_session(
        self,
        transaction: AuthTransaction,
        *,
        token: SecretStr | str,
        now: datetime,
        lock_for_password_change: bool = False,
    ) -> tuple[User, SessionRecord]:
        try:
            token_hash = hash_session_token(token)
        except ValueError as exc:
            raise AuthenticationRequiredError("authentication required") from exc
        if lock_for_password_change:
            user, record = transaction.lock_session_and_user_by_token_hash(token_hash)
        else:
            record = transaction.find_session_by_token_hash(token_hash)
            user = (
                transaction.find_user_by_id(record.user_id) if record is not None else None
            )
        if (
            record is None
            or record.status != SessionStatus.ACTIVE.value
            or record.expires_at <= now
        ):
            raise AuthenticationRequiredError("authentication required")
        if user is None:
            raise AuthenticationRequiredError("authentication required")
        self._ensure_user_available(user)
        return user, record

    def _issue_session(
        self,
        transaction: AuthTransaction,
        *,
        user: User,
        now: datetime,
    ) -> SessionGrant:
        raw_token = self._token_factory.issue()
        expires_at = now + self._session_ttl
        record = SessionRecord(
            id=str(uuid4()),
            user_id=user.id,
            token_hash=hash_session_token(raw_token),
            status=SessionStatus.ACTIVE.value,
            created_at=now,
            expires_at=expires_at,
            revoked_at=None,
        )
        transaction.add_session(record)
        return SessionGrant(
            raw_token=raw_token,
            principal=self._principal(user, record),
            expires_at=expires_at,
        )

    @staticmethod
    def _ensure_user_available(user: User) -> None:
        if user.status != UserStatus.ACTIVE.value:
            raise AccountUnavailableError("account is unavailable")

    @staticmethod
    def _principal(user: User, record: SessionRecord) -> AuthPrincipal:
        try:
            role = UserRole(user.role)
        except ValueError as exc:
            raise AccountUnavailableError("account is unavailable") from exc
        return AuthPrincipal(
            user_id=user.id,
            session_id=record.id,
            username=user.username,
            role=role,
            must_change_password=user.must_change_credential,
        )


__all__ = [
    "AccountUnavailableError",
    "AuthService",
    "AuthenticationError",
    "AuthenticationRequiredError",
    "InvalidCredentialsError",
]
