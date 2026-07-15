"""Secret-safe runtime contracts for server-side authentication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import SecretStr

from quanxin_life.core import UserRole


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    """Authenticated identity; it never contains a password or session token."""

    user_id: str
    session_id: str
    username: str
    role: UserRole
    must_change_password: bool


@dataclass(frozen=True, slots=True, repr=False)
class SessionGrant:
    """One raw opaque token handed only to the HTTP cookie adapter."""

    raw_token: SecretStr
    principal: AuthPrincipal
    expires_at: datetime

    def __repr__(self) -> str:
        return (
            "SessionGrant(raw_token=SecretStr('**********'), "
            f"principal={self.principal!r}, expires_at={self.expires_at!r})"
        )


__all__ = ["AuthPrincipal", "SessionGrant"]
