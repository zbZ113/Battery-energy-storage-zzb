"""Run the reviewed migration chain with loopback-only local settings."""

from __future__ import annotations

import os
from pathlib import Path

from deploy.local_runtime_settings import LocalCompetitionRuntimeSettings
from deploy.migrate import (
    _alembic_config,
    _upgrade_head,
    run_database_migrations,
)


def main() -> int:
    settings = LocalCompetitionRuntimeSettings.from_environment(os.environ)
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
