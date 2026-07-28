"""Create the one forced-rotation ADMIN for an empty competition database."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ConfigDict, Field
from sqlalchemy import select

from quanxin_life.auth import Argon2idPasswordHasher, PasswordPolicy
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence import (
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import User

_MAX_SECRET_BYTES = 8_192


class BootstrapAdminConflictError(RuntimeError):
    """Raised when bootstrap would mutate an initialized identity store."""


class BootstrapAdminResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1, max_length=64)
    username: str = Field(min_length=1, max_length=320)
    created: bool
    must_change_password: bool


def bootstrap_admin(
    session_factory: SessionFactory,
    *,
    username: str,
    temporary_password: str,
    now: datetime,
) -> BootstrapAdminResult:
    """Create one ADMIN or recognize the same untouched bootstrap identity."""

    normalized_username = _username(username)
    timestamp = _utc(now)
    validated_password = PasswordPolicy().validate_password(temporary_password)
    hasher = Argon2idPasswordHasher()

    with session_factory.begin() as session:
        users = tuple(session.scalars(select(User).with_for_update()))
        if users:
            if len(users) != 1:
                raise BootstrapAdminConflictError(
                    "identity store is already initialized"
                )
            existing = users[0]
            if (
                existing.username != normalized_username
                or existing.role != UserRole.ADMIN.value
                or existing.status != UserStatus.ACTIVE.value
                or not existing.must_change_credential
                or not hasher.verify_password(
                    existing.credential_hash,
                    validated_password,
                )
            ):
                raise BootstrapAdminConflictError(
                    "identity store is already initialized"
                )
            return BootstrapAdminResult(
                user_id=existing.id,
                username=existing.username,
                created=False,
                must_change_password=True,
            )

        user_id = str(uuid4())
        session.add(
            User(
                id=user_id,
                username=normalized_username,
                credential_hash=hasher.hash_password(validated_password),
                must_change_credential=True,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
    return BootstrapAdminResult(
        user_id=user_id,
        username=normalized_username,
        created=True,
        must_change_password=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        database_url = _read_secret_line(
            _required_environment("QUANXIN_DATABASE_URL_FILE"),
            label="database URL",
        )
        temporary_password = _read_secret_line(
            _required_environment("QUANXIN_BOOTSTRAP_ADMIN_PASSWORD_FILE"),
            label="bootstrap password",
        )
        result = bootstrap_admin(
            create_session_factory(
                create_engine_from_config(DatabaseConfig(url=database_url))
            ),
            username=_required_environment("QUANXIN_BOOTSTRAP_ADMIN_USERNAME"),
            temporary_password=temporary_password,
            now=datetime.now(UTC),
        )
    except (BootstrapAdminConflictError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"required bootstrap setting is missing: {name}")
    return normalized


def _read_secret_line(raw_path: str, *, label: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} secret must be an absolute regular file")
    if path.stat().st_size > _MAX_SECRET_BYTES:
        raise ValueError(f"{label} secret file is too large")
    try:
        value = path.read_text(encoding="utf-8").rstrip("\r\n")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} secret file cannot be read") from exc
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} secret file must contain one nonblank line")
    return value


def _username(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 320
        or normalized != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("bootstrap username is invalid")
    return normalized


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("bootstrap timestamp must include a timezone")
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapAdminConflictError",
    "BootstrapAdminResult",
    "bootstrap_admin",
]
