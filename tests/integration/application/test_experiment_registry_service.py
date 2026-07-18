from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from quanxin_life.application.a100_suite_import import (
    ImportedA100SuiteRecord,
    ImportedA100TaskRecord,
    RegisteredA100Suite,
)
from quanxin_life.application.experiments import (
    ExperimentAccessError,
    ExperimentNotFoundError,
    ExperimentRegistryService,
    ExperimentSourceError,
    ExperimentStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    ExperimentRun,
    ExperimentSuite,
    Project,
    User,
    UserProjectRole,
)

NOW = datetime(2026, 7, 18, 16, 0, tzinfo=UTC)
IMPORT_ID = "a" * 64


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


class _SuiteSource:
    def __init__(
        self,
        suite: ImportedA100SuiteRecord,
        tasks: tuple[ImportedA100TaskRecord, ...],
        root: Path,
    ) -> None:
        self.suite = suite
        self.tasks = tasks
        self.root = root

    def resolve(self, import_id: str) -> RegisteredA100Suite:
        if import_id != self.suite.import_id:
            raise KeyError(import_id)
        return RegisteredA100Suite(record=self.suite, output_root=self.root)

    def list_tasks(self, import_id: str) -> tuple[ImportedA100TaskRecord, ...]:
        if import_id != self.suite.import_id:
            raise KeyError(import_id)
        return self.tasks


def _suite_record(*, task_count: int = 2) -> ImportedA100SuiteRecord:
    return ImportedA100SuiteRecord(
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
        task_count=task_count,
        file_count=20,
        formal_performance_claim=False,
        registered_relative_root=f"runs/{IMPORT_ID}",
        imported_at=NOW,
    )


def _task(model: str, cutoff: int, seed: int) -> ImportedA100TaskRecord:
    return ImportedA100TaskRecord(
        run_id=f"run-{model}-{cutoff}-{seed}",
        model_name=model,
        cutoff_cycle=cutoff,
        seed=seed,
        config_sha256="c" * 64,
        input_bundle_sha256="d" * 64,
        source_commit="b" * 40,
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-split-v1",
        feature_version="matr-features-v1",
        context_sha256=("f" if model == "cpmlp" else "1") * 64,
        task_relative_root=f"cutoff-{cutoff}/{model}/seed-{seed}",
        completed_at=NOW,
    )


@pytest.fixture
def registry_context(
    tmp_path: Path,
) -> tuple[
    ExperimentRegistryService,
    dict[str, AuthPrincipal],
    str,
    SessionFactory,
]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'experiments.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals = {
        "member": _principal(str(uuid4()), UserRole.MEMBER),
        "outsider": _principal(str(uuid4()), UserRole.MEMBER),
        "judge": _principal(str(uuid4()), UserRole.JUDGE),
        "admin": _principal(str(uuid4()), UserRole.ADMIN),
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
                name="A100 experiments",
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
    source = _SuiteSource(
        _suite_record(),
        (_task("cpmlp", 20, 20260712), _task("hybrid", 50, 20260712)),
        tmp_path / "managed",
    )
    return (
        ExperimentRegistryService(session_factory, source=source),
        principals,
        project_id,
        session_factory,
    )


def test_register_is_idempotent_and_tasks_are_filterable(
    registry_context: tuple[
        ExperimentRegistryService,
        dict[str, AuthPrincipal],
        str,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, session_factory = registry_context

    first = service.register_suite(
        principals["member"],
        project_id=project_id,
        import_id=IMPORT_ID,
        registered_at=NOW,
    )
    second = service.register_suite(
        principals["member"],
        project_id=project_id,
        import_id=IMPORT_ID,
        registered_at=NOW.replace(hour=17),
    )

    assert second == first
    assert first.import_id == IMPORT_ID
    assert first.dataset_id == "MATR"
    assert first.target == "matr_official_cycle_life"
    assert first.formal_performance_claim is False
    assert first.evidence_uri == f"a100-suite://{IMPORT_ID}"
    assert len(service.list_suites(principals["member"], project_id=project_id)) == 1
    filtered = service.list_runs(
        principals["member"],
        project_id=project_id,
        model_name="hybrid",
        cutoff_cycle=50,
        seed=20260712,
    )
    assert len(filtered) == 1
    assert filtered[0].experiment_id == first.experiment_id
    assert filtered[0].task_relative_root == "cutoff-50/hybrid/seed-20260712"
    assert filtered[0].evidence_uri == (
        f"a100-suite://{IMPORT_ID}/cutoff-50/hybrid/seed-20260712"
    )
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ExperimentSuite)) == 1
        assert session.scalar(select(func.count()).select_from(ExperimentRun)) == 2


def test_project_visibility_and_mutation_roles_are_enforced(
    registry_context: tuple[
        ExperimentRegistryService,
        dict[str, AuthPrincipal],
        str,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, _ = registry_context
    registered = service.register_suite(
        principals["member"],
        project_id=project_id,
        import_id=IMPORT_ID,
        registered_at=NOW,
    )

    assert service.get_suite(principals["judge"], registered.experiment_id) == registered
    assert service.list_suites(principals["outsider"], project_id=project_id) == ()
    assert service.list_runs(principals["outsider"], project_id=project_id) == ()
    with pytest.raises(ExperimentNotFoundError):
        service.get_suite(principals["outsider"], registered.experiment_id)
    with pytest.raises(ExperimentNotFoundError):
        service.register_suite(
            principals["outsider"],
            project_id=project_id,
            import_id=IMPORT_ID,
            registered_at=NOW,
        )
    with pytest.raises(ExperimentAccessError):
        service.register_suite(
            principals["judge"],
            project_id=project_id,
            import_id=IMPORT_ID,
            registered_at=NOW,
        )


def test_incomplete_task_catalog_rolls_back_registration(
    registry_context: tuple[
        ExperimentRegistryService,
        dict[str, AuthPrincipal],
        str,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, session_factory = registry_context
    service = ExperimentRegistryService(
        session_factory,
        source=_SuiteSource(
            _suite_record(),
            (_task("cpmlp", 20, 20260712),),
            Path("managed"),
        ),
    )

    with pytest.raises(ExperimentSourceError, match="task catalog"):
        service.register_suite(
            principals["member"],
            project_id=project_id,
            import_id=IMPORT_ID,
            registered_at=NOW,
        )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ExperimentSuite)) == 0
        assert session.scalar(select(func.count()).select_from(ExperimentRun)) == 0


def test_reregistration_rejects_a_tampered_persisted_task_identity(
    registry_context: tuple[
        ExperimentRegistryService,
        dict[str, AuthPrincipal],
        str,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, session_factory = registry_context
    service.register_suite(
        principals["member"],
        project_id=project_id,
        import_id=IMPORT_ID,
        registered_at=NOW,
    )
    with session_factory.begin() as session:
        row = session.scalar(
            select(ExperimentRun).where(ExperimentRun.model_name == "hybrid")
        )
        assert row is not None
        row.model_name = "tampered"

    with pytest.raises(ExperimentStateError, match="task matrix"):
        service.register_suite(
            principals["member"],
            project_id=project_id,
            import_id=IMPORT_ID,
            registered_at=NOW,
        )
