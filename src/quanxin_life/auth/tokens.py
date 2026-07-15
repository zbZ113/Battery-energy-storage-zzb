"""Opaque server-side session-token generation and one-way hashing."""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Callable

from pydantic import SecretStr

_SESSION_TOKEN = re.compile(r"[A-Za-z0-9_-]{43,256}\Z")


class SessionTokenFactory:
    """Issue at least 256 bits of URL-safe randomness for browser sessions."""

    def __init__(self, *, generator: Callable[[int], str] | None = None) -> None:
        self._generator = generator or secrets.token_urlsafe

    def issue(self) -> SecretStr:
        raw_token = self._generator(32)
        if not isinstance(raw_token, str) or not _SESSION_TOKEN.fullmatch(raw_token):
            raise RuntimeError("session token generator returned an unsafe value")
        return SecretStr(raw_token)


def hash_session_token(token: SecretStr | str) -> str:
    """Hash an opaque cookie token before any database lookup or storage."""

    raw_token = token.get_secret_value() if isinstance(token, SecretStr) else token
    if not isinstance(raw_token, str) or not _SESSION_TOKEN.fullmatch(raw_token):
        raise ValueError("session token is invalid")
    return hashlib.sha256(raw_token.encode("ascii")).hexdigest()


__all__ = ["SessionTokenFactory", "hash_session_token"]
