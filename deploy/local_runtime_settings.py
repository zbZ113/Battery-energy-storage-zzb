"""Loopback-only settings for the local competition runtime."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

from deploy.runtime_settings import CompetitionRuntimeSettings

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LocalCompetitionRuntimeSettings(CompetitionRuntimeSettings):
    """Production-equivalent runtime paths with an HTTP loopback origin."""

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> LocalCompetitionRuntimeSettings:
        trusted_origin = _loopback_http_origin(
            environment.get("QUANXIN_TRUSTED_ORIGIN", "")
        )
        production_environment = dict(environment)
        production_environment["QUANXIN_TRUSTED_ORIGIN"] = "https://localhost"
        settings = CompetitionRuntimeSettings.from_environment(
            production_environment
        )
        return cls(
            database_url=settings.database_url,
            redis_url=settings.redis_url,
            trusted_origin=trusted_origin,
            data_root=settings.data_root,
            artifact_root=settings.artifact_root,
            policy_root=settings.policy_root,
            deployment_registry_root=settings.deployment_registry_root,
            deployment_registry_id=settings.deployment_registry_id,
            calibration_registrations_file=(
                settings.calibration_registrations_file
            ),
            calibration_evidence_root=settings.calibration_evidence_root,
            agent_policy_file=settings.agent_policy_file,
        )


def _loopback_http_origin(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("local trusted origin must be a loopback HTTP origin") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("local trusted origin must be a loopback HTTP origin")
    return normalized.removesuffix("/")


__all__ = ["LocalCompetitionRuntimeSettings"]
