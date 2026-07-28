from __future__ import annotations

from pathlib import Path

import pytest

from deploy.runtime_settings import CompetitionRuntimeSettings


def _environment(tmp_path: Path) -> dict[str, str]:
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
    roots = {}
    for name in ("data", "artifacts", "policies", "deployment-registry"):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    registrations = tmp_path / "calibration-registrations.json"
    registrations.write_text("[]\n", encoding="utf-8")
    return {
        "QUANXIN_DATABASE_URL_FILE": str(database_secret),
        "QUANXIN_REDIS_URL_FILE": str(redis_secret),
        "QUANXIN_TRUSTED_ORIGIN": "https://demo.example.test",
        "QUANXIN_DATA_ROOT": str(roots["data"]),
        "QUANXIN_ARTIFACT_ROOT": str(roots["artifacts"]),
        "QUANXIN_POLICY_ROOT": str(roots["policies"]),
        "QUANXIN_DEPLOYMENT_REGISTRY_ROOT": str(
            roots["deployment-registry"]
        ),
        "QUANXIN_DEPLOYMENT_REGISTRY_ID": "a" * 64,
        "QUANXIN_CALIBRATION_REGISTRATIONS_FILE": str(registrations),
    }


def test_runtime_settings_load_only_explicit_files_and_paths(tmp_path: Path) -> None:
    settings = CompetitionRuntimeSettings.from_environment(
        _environment(tmp_path)
    )

    assert settings.trusted_origin == "https://demo.example.test"
    assert settings.database_url.get_secret_value().startswith(
        "postgresql+psycopg://"
    )
    assert settings.redis_url.get_secret_value().startswith("redis://")
    assert settings.deployment_registry_id == "a" * 64
    assert settings.data_root.is_absolute()
    assert settings.calibration_registrations_file.is_file()


def test_runtime_settings_repr_never_exposes_credentials(tmp_path: Path) -> None:
    settings = CompetitionRuntimeSettings.from_environment(
        _environment(tmp_path)
    )

    rendered = repr(settings) + str(settings)

    assert "database-secret" not in rendered
    assert "redis-secret" not in rendered
    assert "**********" in rendered


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("QUANXIN_TRUSTED_ORIGIN", "http://demo.example.test", "HTTPS"),
        ("QUANXIN_TRUSTED_ORIGIN", "https://demo.example.test/path", "origin"),
        ("QUANXIN_DEPLOYMENT_REGISTRY_ID", "not-a-sha", "SHA-256"),
    ),
)
def test_runtime_settings_reject_unsafe_public_identity(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    environment = _environment(tmp_path)
    environment[key] = value

    with pytest.raises(ValueError, match=message):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_relative_or_missing_operator_paths(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["QUANXIN_ARTIFACT_ROOT"] = "relative/artifacts"

    with pytest.raises(ValueError, match="absolute"):
        CompetitionRuntimeSettings.from_environment(environment)

    environment = _environment(tmp_path / "second")
    missing = tmp_path / "missing"
    environment["QUANXIN_POLICY_ROOT"] = str(missing)
    with pytest.raises(ValueError, match="directory"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_secret_symlinks(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    link = tmp_path / "database-link"
    try:
        link.symlink_to(Path(environment["QUANXIN_DATABASE_URL_FILE"]))
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    environment["QUANXIN_DATABASE_URL_FILE"] = str(link)

    with pytest.raises(ValueError, match="symbolic link"):
        CompetitionRuntimeSettings.from_environment(environment)
