"""Authentication primitives and application services."""

from quanxin_life.auth.contracts import AuthPrincipal, SessionGrant
from quanxin_life.auth.passwords import (
    Argon2idConfig,
    Argon2idPasswordHasher,
    PasswordPolicy,
)
from quanxin_life.auth.service import (
    AccountUnavailableError,
    AuthenticationError,
    AuthenticationRequiredError,
    AuthService,
    InvalidCredentialsError,
)
from quanxin_life.auth.sqlalchemy_repository import SqlAlchemyAuthTransactionFactory
from quanxin_life.auth.tokens import SessionTokenFactory, hash_session_token

__all__ = [
    "AccountUnavailableError",
    "Argon2idConfig",
    "Argon2idPasswordHasher",
    "AuthPrincipal",
    "AuthService",
    "AuthenticationError",
    "AuthenticationRequiredError",
    "InvalidCredentialsError",
    "PasswordPolicy",
    "SessionGrant",
    "SessionTokenFactory",
    "SqlAlchemyAuthTransactionFactory",
    "hash_session_token",
]
