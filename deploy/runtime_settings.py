"""Fail-closed settings for the single-node competition deployment."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SECRET_BYTES = 8_192


@dataclass(frozen=True, slots=True)
class CompetitionRuntimeSettings:
    """Operator-owned runtime identity without implicit environment fallbacks."""

    database_url: SecretStr
    redis_url: SecretStr
    trusted_origin: str
    data_root: Path
    artifact_root: Path
    policy_root: Path
    deployment_registry_root: Path
    deployment_registry_id: str
    calibration_registrations_file: Path

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> CompetitionRuntimeSettings:
        database_url = _secret_url(
            environment,
            "QUANXIN_DATABASE_URL_FILE",
            schemes={"postgresql", "postgresql+psycopg"},
            label="database URL",
        )
        redis_url = _secret_url(
            environment,
            "QUANXIN_REDIS_URL_FILE",
            schemes={"redis", "rediss"},
            label="Redis URL",
        )
        trusted_origin = _trusted_origin(
            _required(environment, "QUANXIN_TRUSTED_ORIGIN")
        )
        registry_id = _required(
            environment,
            "QUANXIN_DEPLOYMENT_REGISTRY_ID",
        ).lower()
        if _SHA256.fullmatch(registry_id) is None:
            raise ValueError("deployment registry ID must be a SHA-256 digest")
        return cls(
            database_url=database_url,
            redis_url=redis_url,
            trusted_origin=trusted_origin,
            data_root=_directory(
                environment,
                "QUANXIN_DATA_ROOT",
            ),
            artifact_root=_directory(
                environment,
                "QUANXIN_ARTIFACT_ROOT",
            ),
            policy_root=_directory(
                environment,
                "QUANXIN_POLICY_ROOT",
            ),
            deployment_registry_root=_directory(
                environment,
                "QUANXIN_DEPLOYMENT_REGISTRY_ROOT",
            ),
            deployment_registry_id=registry_id,
            calibration_registrations_file=_regular_file(
                environment,
                "QUANXIN_CALIBRATION_REGISTRATIONS_FILE",
            ),
        )


def _required(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key, "")
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"required runtime setting is missing: {key}")
    return normalized


def _operator_path(environment: Mapping[str, str], key: str) -> Path:
    path = Path(_required(environment, key))
    if not path.is_absolute():
        raise ValueError(f"operator path must be absolute: {key}")
    if path.is_symlink():
        raise ValueError(f"operator path must not be a symbolic link: {key}")
    return path


def _directory(environment: Mapping[str, str], key: str) -> Path:
    path = _operator_path(environment, key)
    if not path.is_dir():
        raise ValueError(f"operator path must be an existing directory: {key}")
    return path.resolve(strict=True)


def _regular_file(environment: Mapping[str, str], key: str) -> Path:
    path = _operator_path(environment, key)
    if not path.is_file():
        raise ValueError(f"operator path must be an existing regular file: {key}")
    return path.resolve(strict=True)


def _secret_url(
    environment: Mapping[str, str],
    key: str,
    *,
    schemes: set[str],
    label: str,
) -> SecretStr:
    path = _regular_file(environment, key)
    if path.stat().st_size > _MAX_SECRET_BYTES:
        raise ValueError(f"{label} secret file is too large")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} secret file cannot be read") from exc
    value = raw.strip()
    if not value or "\x00" in value or len(value.splitlines()) != 1:
        raise ValueError(f"{label} secret file must contain one nonblank line")
    parsed = urlsplit(value)
    if parsed.scheme not in schemes or not parsed.hostname:
        raise ValueError(f"{label} uses an unsupported or incomplete URL")
    return SecretStr(value)


def _trusted_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError("production trusted Origin must use HTTPS")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("production trusted origin must contain only scheme and authority")
    return value.removesuffix("/")


__all__ = ["CompetitionRuntimeSettings"]
