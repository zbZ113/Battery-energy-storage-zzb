from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.model_artifacts import create_model_artifact_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.model_artifact_catalog import (
    ClassicModelArtifactCatalogSource,
    ModelArtifactCatalogService,
)
from quanxin_life.application.model_artifacts import (
    ArtifactFormat,
    ArtifactKind,
    ModelArtifactManifest,
    ModelArtifactRegistry,
)
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import Project, User, UserProjectRole

NOW = datetime(2026, 7, 18, 19, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
USERNAME = "artifact-admin@example.test"
PASSWORD = "temporary artifact password 2026"


def _client(tmp_path: Path) -> tuple[TestClient, str, str]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "model.ubj"
    artifact_path.write_bytes(b"verified-native-model")
    artifact_id = str(uuid4())
    registry = ModelArtifactRegistry(artifact_root)
    registry.register(
        ModelArtifactManifest(
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
            cutoff_cycle=50,
            feature_names=("capacity_delta",),
            created_at=NOW,
        )
    )
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    user_id = str(uuid4())
    project_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            User(
                id=user_id,
                username=USERNAME,
                credential_hash=hasher.hash_password(PASSWORD),
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Project(
                id=project_id,
                owner_user_id=user_id,
                name="artifact catalog",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=user_id,
                project_id=project_id,
                role=UserRole.ADMIN.value,
                created_at=NOW,
            )
        )
    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,)),
    )
    model_artifact_adapter = create_model_artifact_http_adapter(
        ModelArtifactCatalogService(
            session_factory,
            source=ClassicModelArtifactCatalogSource(registry),
        ),
        auth_adapter=auth_adapter,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        model_artifact_adapter=model_artifact_adapter,
    )
    client = TestClient(app, base_url="https://api.example.test")
    login = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": PASSWORD},
    )
    assert login.status_code == 200
    return client, project_id, artifact_id


def test_admin_registers_and_queries_only_verified_artifact_metadata(
    tmp_path: Path,
) -> None:
    client, project_id, artifact_id = _client(tmp_path)

    blocked = client.post(
        "/v1/admin/model-artifacts",
        json={"project_id": project_id, "artifact_id": artifact_id},
    )
    created = client.post(
        "/v1/admin/model-artifacts",
        headers={"Origin": ORIGIN},
        json={"project_id": project_id, "artifact_id": artifact_id},
    )

    assert blocked.status_code == 403
    assert created.status_code == 201
    assert created.json()["status"] == "VERIFIED"
    assert created.json()["artifact_kind"] == "xgboost"
    assert created.json()["object_uri"].startswith("verified-model-artifact://")
    assert str(tmp_path) not in created.text
    assert "mae" not in created.text.casefold()
    assert "metrics" not in created.text.casefold()

    listed = client.get(
        "/v1/model-artifacts",
        params={
            "project_id": project_id,
            "artifact_kind": "xgboost",
            "cutoff_cycle": 50,
        },
    )
    detail = client.get(f"/v1/model-artifacts/{artifact_id}")
    assert listed.status_code == 200
    assert listed.json() == [created.json()]
    assert detail.status_code == 200
    assert detail.json() == created.json()


def test_unknown_verified_source_is_reported_without_catalog_mutation(
    tmp_path: Path,
) -> None:
    client, project_id, _ = _client(tmp_path)

    response = client.post(
        "/v1/admin/model-artifacts",
        headers={"Origin": ORIGIN},
        json={"project_id": project_id, "artifact_id": str(uuid4())},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "model_artifact_source_unavailable"
    assert client.get("/v1/model-artifacts").json() == []
