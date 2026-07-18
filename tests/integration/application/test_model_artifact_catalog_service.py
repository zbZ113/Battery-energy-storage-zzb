from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from quanxin_life.application.model_artifact_catalog import (
    ClassicModelArtifactCatalogSource,
    ModelArtifactCatalogAccessError,
    ModelArtifactCatalogNotFoundError,
    ModelArtifactCatalogService,
    ModelArtifactCatalogStateError,
)
from quanxin_life.application.model_artifacts import (
    ArtifactFormat,
    ArtifactKind,
    ModelArtifactManifest,
    ModelArtifactRegistry,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    ModelArtifact,
    ModelManifest,
    Project,
    User,
    UserProjectRole,
)

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=UTC)


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


@pytest.fixture
def catalog_context(
    tmp_path: Path,
) -> tuple[
    ModelArtifactCatalogService,
    dict[str, AuthPrincipal],
    str,
    str,
    Path,
    SessionFactory,
]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "model.ubj"
    artifact_path.write_bytes(b"verified-xgboost-native-bytes")
    artifact_id = str(uuid4())
    manifest = ModelArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_UBJ,
        relative_path=artifact_path.name,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        size_bytes=artifact_path.stat().st_size,
        model_version="xgboost-v1",
        data_version="matr-three-batch-v1",
        feature_version="matr-early-features-v1",
        split_version="matr-three-batch-split-v1",
        schema_version="model-artifact-v1",
        dataset_id="MATR",
        cutoff_cycle=20,
        feature_names=("capacity_delta", "resistance_growth"),
        created_at=NOW,
    )
    registry = ModelArtifactRegistry(artifact_root)
    registry.register(manifest)

    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals = {
        "admin": _principal(str(uuid4()), UserRole.ADMIN),
        "member": _principal(str(uuid4()), UserRole.MEMBER),
        "judge": _principal(str(uuid4()), UserRole.JUDGE),
        "outsider": _principal(str(uuid4()), UserRole.MEMBER),
    }
    project_id = str(uuid4())
    with session_factory.begin() as session:
        for principal in principals.values():
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.add(
            Project(
                id=project_id,
                owner_user_id=principals["member"].user_id,
                name="verified artifacts",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=principals["judge"].user_id,
                project_id=project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    service = ModelArtifactCatalogService(
        session_factory,
        source=ClassicModelArtifactCatalogSource(registry),
    )
    return (
        service,
        principals,
        project_id,
        artifact_id,
        artifact_path,
        session_factory,
    )


def test_admin_registration_is_verified_idempotent_and_queryable(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, _, session_factory = catalog_context

    first = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )
    second = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW.replace(hour=19),
    )

    assert second == first
    assert first.project_id == project_id
    assert first.artifact_kind == "xgboost"
    assert first.artifact_format == "xgboost-ubj"
    assert first.status == "VERIFIED"
    assert first.dataset_id == "MATR"
    assert first.cutoff_cycle == 20
    assert first.object_uri.startswith(f"verified-model-artifact://{artifact_id}/")
    assert first.manifest_uri == f"verified-model-artifact://{artifact_id}/manifest"
    assert service.get_artifact(principals["judge"], artifact_id) == first
    assert service.list_artifacts(
        principals["member"],
        project_id=project_id,
        artifact_kind="xgboost",
        cutoff_cycle=20,
    ) == (first,)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 1
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 1


def test_registration_requires_admin_and_queries_hide_invisible_projects(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, _, _ = catalog_context

    with pytest.raises(ModelArtifactCatalogAccessError):
        service.register_artifact(
            principals["member"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )
    registered = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )

    assert service.list_artifacts(principals["outsider"]) == ()
    with pytest.raises(ModelArtifactCatalogNotFoundError):
        service.get_artifact(principals["outsider"], registered.artifact_id)


def test_registration_reverifies_bytes_and_persisted_context(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, artifact_path, session_factory = (
        catalog_context
    )
    service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )
    artifact_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        service.register_artifact(
            principals["admin"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )

    artifact_path.write_bytes(b"verified-xgboost-native-bytes")
    with session_factory.begin() as session:
        row = session.get(ModelManifest, session.scalar(select(ModelManifest.id)))
        assert row is not None
        row.model_version = "tampered-version"

    with pytest.raises(ModelArtifactCatalogStateError, match="persisted"):
        service.register_artifact(
            principals["admin"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )
