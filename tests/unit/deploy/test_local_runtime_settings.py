from __future__ import annotations

from pathlib import Path

import pytest


def _settings_type():
    try:
        from deploy.local_runtime_settings import LocalCompetitionRuntimeSettings
    except ModuleNotFoundError:
        pytest.fail("LocalCompetitionRuntimeSettings is not implemented")
    return LocalCompetitionRuntimeSettings


def _environment(tmp_path: Path, *, origin: str) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database_secret = tmp_path / "database_url"
    database_secret.write_text(
        "postgresql+psycopg://quanxin:database-secret@postgres:5432/quanxin\n",
        encoding="utf-8",
    )
    redis_secret = tmp_path / "redis_url"
    redis_secret.write_text(
        "redis://:redis-secret@redis:6379/0\n",
        encoding="utf-8",
    )
    roots: dict[str, Path] = {}
    for name in (
        "data",
        "artifacts",
        "policies",
        "deployment-registry",
        "calibration-evidence",
    ):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    registrations = tmp_path / "calibration-sources.json"
    registrations.write_text("[]\n", encoding="utf-8")
    agent_policy = roots["policies"] / "advanced-agent.json"
    agent_policy.write_text(
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":0.1}\n',
        encoding="utf-8",
    )
    return {
        "QUANXIN_DATABASE_URL_FILE": str(database_secret),
        "QUANXIN_REDIS_URL_FILE": str(redis_secret),
        "QUANXIN_TRUSTED_ORIGIN": origin,
        "QUANXIN_DATA_ROOT": str(roots["data"]),
        "QUANXIN_ARTIFACT_ROOT": str(roots["artifacts"]),
        "QUANXIN_POLICY_ROOT": str(roots["policies"]),
        "QUANXIN_DEPLOYMENT_REGISTRY_ROOT": str(
            roots["deployment-registry"]
        ),
        "QUANXIN_DEPLOYMENT_REGISTRY_ID": "a" * 64,
        "QUANXIN_CALIBRATION_REGISTRATIONS_FILE": str(registrations),
        "QUANXIN_CALIBRATION_EVIDENCE_ROOT": str(
            roots["calibration-evidence"]
        ),
        "QUANXIN_AGENT_POLICY_FILE": str(agent_policy),
    }


@pytest.mark.parametrize(
    "origin",
    (
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
    ),
)
def test_local_settings_accept_only_loopback_http(
    tmp_path: Path,
    origin: str,
) -> None:
    settings = _settings_type().from_environment(
        _environment(tmp_path, origin=origin)
    )

    assert settings.trusted_origin == origin


@pytest.mark.parametrize(
    "origin",
    (
        "http://192.168.1.2:8080",
        "http://api.example.test",
        "https://localhost:8080",
        "http://localhost:8080/path",
        "http://user@localhost:8080",
        "http://localhost:8080?query=1",
    ),
)
def test_local_settings_reject_non_loopback_or_non_http_origin(
    tmp_path: Path,
    origin: str,
) -> None:
    with pytest.raises(ValueError, match="loopback HTTP origin"):
        _settings_type().from_environment(
            _environment(tmp_path, origin=origin)
        )


def test_local_settings_repr_does_not_expose_service_credentials(
    tmp_path: Path,
) -> None:
    settings = _settings_type().from_environment(
        _environment(tmp_path, origin="http://127.0.0.1:8080")
    )

    rendered = repr(settings) + str(settings)

    assert "database-secret" not in rendered
    assert "redis-secret" not in rendered
    assert "**********" in rendered
