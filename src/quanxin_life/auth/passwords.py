"""Argon2id password hashing and beginner-friendly passphrase policy."""

from __future__ import annotations

import unicodedata

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type
from pydantic import BaseModel, ConfigDict, Field


class Argon2idConfig(BaseModel):
    """Versioned Argon2id work factors; defaults target an interactive login."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_cost: int = Field(default=2, ge=1, le=10)
    memory_cost_kib: int = Field(default=19 * 1024, ge=8 * 1024, le=1024 * 1024)
    parallelism: int = Field(default=1, ge=1, le=16)
    hash_len: int = Field(default=32, ge=16, le=64)
    salt_len: int = Field(default=16, ge=16, le=64)


class PasswordPolicy(BaseModel):
    """Passphrase policy without brittle character-class composition rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_length: int = Field(default=12, ge=12, le=128)
    max_length: int = Field(default=128, ge=64, le=1024)

    def validate_password(self, password: str) -> str:
        if not isinstance(password, str):
            raise TypeError("password must be text")
        if len(password) < self.min_length:
            raise ValueError(f"password must contain at least {self.min_length} characters")
        if len(password) > self.max_length:
            raise ValueError(f"password must contain at most {self.max_length} characters")
        if password != password.strip():
            raise ValueError("password must not contain leading or trailing whitespace")
        if any(unicodedata.category(character).startswith("C") for character in password):
            raise ValueError("password must not contain control characters")
        return password


class Argon2idPasswordHasher:
    """Small fail-closed wrapper around argon2-cffi's Argon2id implementation."""

    def __init__(self, *, config: Argon2idConfig | None = None) -> None:
        self._config = config or Argon2idConfig()
        self._hasher = PasswordHasher(
            time_cost=self._config.time_cost,
            memory_cost=self._config.memory_cost_kib,
            parallelism=self._config.parallelism,
            hash_len=self._config.hash_len,
            salt_len=self._config.salt_len,
            type=Type.ID,
        )

    def hash_password(self, password: str) -> str:
        """Create a salted Argon2id hash; never persist the input password."""

        if not isinstance(password, str) or not password:
            raise ValueError("password must not be blank")
        return self._hasher.hash(password)

    def verify_password(self, credential_hash: str, password: str) -> bool:
        """Verify an Argon2id hash and return False for malformed credentials."""

        if not isinstance(credential_hash, str) or not credential_hash.startswith("$argon2id$"):
            return False
        if not isinstance(password, str):
            return False
        try:
            return bool(self._hasher.verify(credential_hash, password))
        except (InvalidHashError, VerificationError):
            return False

    def needs_rehash(self, credential_hash: str) -> bool:
        """Return True when a valid stored hash uses obsolete work factors."""

        if not isinstance(credential_hash, str) or not credential_hash.startswith("$argon2id$"):
            return True
        try:
            return self._hasher.check_needs_rehash(credential_hash)
        except (InvalidHashError, VerificationError):
            return True


__all__ = ["Argon2idConfig", "Argon2idPasswordHasher", "PasswordPolicy"]
