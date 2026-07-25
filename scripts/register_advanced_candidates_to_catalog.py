"""Register the managed 15-artifact Advanced bundle in the product SQL catalog."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.application.advanced_deployment_registry import (  # noqa: E402
    AdvancedDeepModelArtifactCatalogSource,
    AdvancedDeploymentBundleRegistry,
)
from quanxin_life.application.model_artifact_catalog import (  # noqa: E402
    ModelArtifactCatalogService,
)
from quanxin_life.auth import AuthPrincipal  # noqa: E402
from quanxin_life.core import UserRole, UserStatus  # noqa: E402
from quanxin_life.persistence import (  # noqa: E402
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig  # noqa: E402
from quanxin_life.persistence.models import User  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("registry_id")
    parser.add_argument("project_id")
    parser.add_argument("--admin-user-id", required=True)
    parser.add_argument("--database-url-env", default="QUANXIN_DATABASE_URL")
    args = parser.parse_args()

    database_url = os.environ.get(args.database_url_env, "").strip()
    if not database_url:
        parser.error(f"database URL environment variable is missing: {args.database_url_env}")
    engine = create_engine_from_config(DatabaseConfig(url=database_url))
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        user = session.scalar(select(User).where(User.id == args.admin_user_id))
    if (
        user is None
        or user.role != UserRole.ADMIN.value
        or user.status != UserStatus.ACTIVE.value
        or user.must_change_credential
    ):
        parser.error("admin user must be active and fully initialized")
    principal = AuthPrincipal(
        user_id=user.id,
        session_id="advanced-catalog-batch-cli",
        username=user.username,
        role=UserRole.ADMIN,
        must_change_password=False,
    )
    source = AdvancedDeepModelArtifactCatalogSource(
        AdvancedDeploymentBundleRegistry(args.registry_root),
        args.registry_id,
    )
    batch = ModelArtifactCatalogService(
        session_factory,
        source=source,
    ).register_advanced_candidates(
        principal,
        project_id=args.project_id,
        registered_at=datetime.now(UTC),
    )
    print(
        json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
