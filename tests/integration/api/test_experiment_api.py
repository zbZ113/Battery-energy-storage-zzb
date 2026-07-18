from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.experiments import create_experiment_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.a100_suite_import import (
    ImportedA100SuiteRecord,
    ImportedA100TaskRecord,
    RegisteredA100Suite,
)
from quanxin_life.application.experiments import ExperimentRegistryService
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

NOW = datetime(2026, 7, 18, 17, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
USERNAME = "experimenter@example.test"
PASSWORD = "temporary experiment password 2026"
IMPORT_ID = "a" * 64


class _SuiteSource:
    def __init__(self, root: Path) -> None:
        self._suite = ImportedA100SuiteRecord(
            import_id=IMPORT_ID,
            mode="smoke",
            source_commit="b" * 40,
            config_sha256="c" * 64,
            input_bundle_sha256="d" * 64,
            data_version="matr-three-batch-v1",
            split_version="matr-three-batch-split-v1",
            feature_version="matr-features-v1",
            output_sha256=IMPORT_ID,
            transfer_sha256="e" * 64,
            task_count=1,
            file_count=10,
            formal_performance_claim=False,
            registered_relative_root=f"runs/{IMPORT_ID}",
            imported_at=NOW,
        )
        self._task = ImportedA100TaskRecord(
            run_id="run-cpmlp-20-20260712",
            model_name="cpmlp",
            cutoff_cycle=20,
            seed=20260712,
            config_sha256="c" * 64,
            input_bundle_sha256="d" * 64,
            source_commit="b" * 40,
            data_version="matr-three-batch-v1",
            split_version="matr-three-batch-split-v1",
            feature_version="matr-features-v1",
            context_sha256="f" * 64,
            task_relative_root="cutoff-20/cpmlp/seed-20260712",
            completed_at=NOW,
        )
        self._root = root

    def resolve(self, import_id: str) -> RegisteredA100Suite:
        if import_id != IMPORT_ID:
            raise KeyError(import_id)
        return RegisteredA100Suite(record=self._suite, output_root=self._root)

    def list_tasks(self, import_id: str) -> tuple[ImportedA100TaskRecord, ...]:
        if import_id != IMPORT_ID:
            raise KeyError(import_id)
        return (self._task,)


def _client(tmp_path: Path) -> tuple[TestClient, str]:
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
                role=UserRole.MEMBER.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Project(
                id=project_id,
                owner_user_id=user_id,
                name="experiment registry",
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
                role=UserRole.MEMBER.value,
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
    experiment_adapter = create_experiment_http_adapter(
        ExperimentRegistryService(
            session_factory,
            source=_SuiteSource(tmp_path / "managed"),
        ),
        auth_adapter=auth_adapter,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth_adapter,
        experiment_adapter=experiment_adapter,
    )
    client = TestClient(app, base_url="https://api.example.test")
    login = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": PASSWORD},
    )
    assert login.status_code == 200
    return client, project_id


def test_authenticated_operator_registers_and_queries_verified_experiment(
    tmp_path: Path,
) -> None:
    client, project_id = _client(tmp_path)

    blocked = client.post(
        "/v1/experiments",
        json={"project_id": project_id, "import_id": IMPORT_ID},
    )
    created = client.post(
        "/v1/experiments",
        headers={"Origin": ORIGIN},
        json={"project_id": project_id, "import_id": IMPORT_ID},
    )

    assert blocked.status_code == 403
    assert created.status_code == 201
    experiment_id = created.json()["experiment_id"]
    listed = client.get("/v1/experiments", params={"project_id": project_id})
    runs = client.get(
        "/v1/experiment-runs",
        params={
            "project_id": project_id,
            "model_name": "cpmlp",
            "cutoff_cycle": 20,
            "seed": 20260712,
        },
    )
    detail = client.get(f"/v1/experiments/{experiment_id}")

    assert listed.status_code == 200
    assert listed.json() == [created.json()]
    assert detail.json() == created.json()
    assert runs.status_code == 200
    assert len(runs.json()) == 1
    assert runs.json()[0]["target"] == "matr_official_cycle_life"
    assert "metrics" not in runs.text.casefold()
    assert "mae" not in runs.text.casefold()


def test_unknown_import_is_reported_without_creating_an_experiment(
    tmp_path: Path,
) -> None:
    client, project_id = _client(tmp_path)

    response = client.post(
        "/v1/experiments",
        headers={"Origin": ORIGIN},
        json={"project_id": project_id, "import_id": "9" * 64},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "experiment_source_unavailable"
    assert client.get("/v1/experiments").json() == []
