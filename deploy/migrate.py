"""Run the reviewed Alembic chain using a secret-backed PostgreSQL URL."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from alembic import command
from alembic.config import Config

from deploy.runtime_settings import CompetitionRuntimeSettings


class MigrationConfig(Protocol):
    def set_main_option(self, key: str, value: str) -> None: ...


ConfigT = TypeVar("ConfigT", bound=MigrationConfig)


def _alembic_config(path: str) -> Config:
    return Config(path)


def _upgrade_head(config: Config, revision: str) -> None:
    command.upgrade(config, revision)


def run_database_migrations(
    settings: CompetitionRuntimeSettings,
    *,
    config_path: Path,
    config_factory: Callable[[str], ConfigT],
    upgrade: Callable[[ConfigT, str], None],
) -> ConfigT:
    """Upgrade one explicit database without persisting its credentials."""

    config = config_factory(str(config_path))
    escaped_url = settings.database_url.get_secret_value().replace("%", "%%")
    config.set_main_option("sqlalchemy.url", escaped_url)
    upgrade(config, "head")
    return config


def main() -> int:
    settings = CompetitionRuntimeSettings.from_environment(os.environ)
    root = Path(__file__).resolve().parents[1]
    run_database_migrations(
        settings,
        config_path=root / "alembic.ini",
        config_factory=_alembic_config,
        upgrade=_upgrade_head,
    )
    print("DATABASE_MIGRATIONS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
