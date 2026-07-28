from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import SecretStr

from deploy.migrate import run_database_migrations
from deploy.runtime_settings import CompetitionRuntimeSettings


class _Config:
    def __init__(self, path: str) -> None:
        self.path = path
        self.values: dict[str, str] = {}

    def set_main_option(self, key: str, value: str) -> None:
        self.values[key] = value


def _settings(tmp_path: Path) -> CompetitionRuntimeSettings:
    roots = []
    for name in ("data", "artifacts", "policies", "registry"):
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
    registrations = tmp_path / "registrations.json"
    registrations.write_text("[]\n", encoding="utf-8")
    return CompetitionRuntimeSettings(
        database_url=SecretStr(
            "postgresql+psycopg://quanxin:p%40ss@postgres:5432/quanxin"
        ),
        redis_url=SecretStr("redis://:secret@redis:6379/0"),
        trusted_origin="https://demo.example.test",
        data_root=roots[0],
        artifact_root=roots[1],
        policy_root=roots[2],
        deployment_registry_root=roots[3],
        deployment_registry_id="a" * 64,
        calibration_registrations_file=registrations,
    )


def test_migration_injects_secret_url_and_upgrades_only_to_head(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def upgrade(config: _Config, revision: str) -> None:
        calls["config"] = config
        calls["revision"] = revision

    config = run_database_migrations(
        _settings(tmp_path),
        config_path=tmp_path / "alembic.ini",
        config_factory=_Config,
        upgrade=upgrade,
    )

    assert config.path == str(tmp_path / "alembic.ini")
    assert config.values == {
        "sqlalchemy.url": (
            "postgresql+psycopg://quanxin:p%%40ss@postgres:5432/quanxin"
        )
    }
    assert calls == {"config": config, "revision": "head"}
