from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import Field
from sqlalchemy import Engine, func, select, text
from sqlalchemy.engine import make_url

from quanxin_life.agents.supervisor import (
    SupervisorPlanner,
    SupervisorPlanningRequest,
)
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.agent_run_execution import AgentRunExecutionWorker
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentRunStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence import create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    AgentStep,
    ProvenanceRecordRow,
    RecordBatchBinding,
    SessionRecord,
    ToolResultRecord,
    User,
)
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry

NOW = datetime(2026, 8, 2, 6, 0, tzinfo=UTC)
POSTGRES_URL_ENV = "QUANXIN_TEST_POSTGRES_URL"


class _RecordBatchInput(ContractModel):
    record_batch_id: str = Field(min_length=1)


class _PredictionInput(ContractModel):
    upstream_result_id: str = Field(min_length=1)
    route_role: str = Field(min_length=1)


class _ConformalInput(ContractModel):
    operation: str = Field(min_length=1)
    task: str = Field(min_length=1)
    route_role: str = Field(min_length=1)
    alpha: float | None = None
    calibration_sample_result_ids: tuple[str, ...] = ()
    prediction_result_id: str | None = None
    calibration_result_id: str | None = None


class _ReportInput(ContractModel):
    rul_result_id: str = Field(min_length=1)
    soh_result_id: str = Field(min_length=1)
    rul_conformal_result_id: str = Field(min_length=1)
    soh_conformal_result_id: str = Field(min_length=1)


class _RuntimeV7Context:
    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        del run_id, project_id
        return dataset_id

    def resolve_context(
        self,
        *,
        run_id: str,
        project_id: str,
        reference: str,
    ) -> object:
        del run_id, project_id
        values: dict[str, object] = {
            "context.rul_point_route_role": "POINT_ACCURACY",
            "context.rul_coverage_route_role": "COVERAGE",
            "context.soh_route_role": "MEAN_ACCURACY",
            "context.conformal_calibrate_operation": "CALIBRATE",
            "context.conformal_issue_operation": "ISSUE",
            "context.rul_task": "RUL",
            "context.soh_task": "SOH",
            "context.conformal_alpha": 0.1,
            "context.rul_calibration_sample_result_ids": ("rul-sample",),
            "context.soh_calibration_sample_result_ids": ("soh-sample",),
        }
        return values[reference]


def _postgres_url() -> str:
    value = os.environ.get(POSTGRES_URL_ENV)
    if value is None:
        pytest.skip(f"{POSTGRES_URL_ENV} is required for PostgreSQL verification")
    parsed = make_url(value)
    if (
        parsed.get_backend_name() != "postgresql"
        or parsed.host not in {"127.0.0.1", "localhost", "::1"}
        or parsed.password is not None
    ):
        raise RuntimeError(
            "PostgreSQL verification requires a passwordless loopback test database"
        )
    return value


def _reset_public_schema(engine: Engine) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))


@pytest.fixture(scope="module")
def postgres_runtime() -> Iterator[tuple[Engine, SessionFactory]]:
    database_url = _postgres_url()
    engine = create_engine_from_config(DatabaseConfig(url=database_url))
    _reset_public_schema(engine)
    config = Config(str(Path("alembic.ini").resolve()))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")
    command.check(config)
    try:
        yield engine, create_session_factory(engine)
    finally:
        command.downgrade(config, "base")
        _reset_public_schema(engine)
        engine.dispose()


def _principal(session_factory: SessionFactory) -> AuthPrincipal:
    user_id = str(uuid4())
    session_id = str(uuid4())
    principal = AuthPrincipal(
        user_id=user_id,
        session_id=session_id,
        username=f"postgres-runtime-{user_id[:8]}@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )
    with session_factory.begin() as session:
        session.add(
            User(
                id=user_id,
                username=principal.username,
                credential_hash="test-only-credential-hash",
                must_change_credential=False,
                role=principal.role.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=session_id,
                user_id=user_id,
                token_hash=uuid4().hex + uuid4().hex,
                status="ACTIVE",
                created_at=NOW,
                expires_at=NOW + timedelta(hours=1),
            )
        )
    return principal


def _tool_result(tool_name: StandardToolName, value: ContractModel) -> ToolResult:
    input_value = value.model_dump(mode="json")
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version="runtime-v7-postgres-test-v1",
        model_version=None,
        data_version="runtime-v7-postgres-test-data-v1",
        feature_version="runtime-v7-postgres-test-features-v1",
        input_hash=sha256_canonical(input_value),
        values={"artifact_identity": tool_name.value},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id=f"source-{uuid4()}",
                source_kind=SourceKind.OBSERVED,
                uri="test://runtime-v7-postgres",
                sha256=sha256_canonical(input_value),
                description="PostgreSQL Runtime V7 verification input",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _tool_service() -> tuple[ToolInvocationService, list[StandardToolName]]:
    registry = ToolRegistry()
    calls: list[StandardToolName] = []

    def register(tool_name: StandardToolName, input_model: type[ContractModel]) -> None:
        def execute(value: ContractModel) -> ToolResult:
            calls.append(tool_name)
            return _tool_result(tool_name, value)

        registry.register(
            ToolDefinition(
                tool_name=tool_name,
                tool_version="runtime-v7-postgres-test-v1",
                input_model=input_model,
                executor=execute,
            )
        )

    register(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES, _RecordBatchInput)
    register(StandardToolName.PREDICT_CYCLE_LIFE, _PredictionInput)
    register(StandardToolName.PREDICT_SOH_TRAJECTORY, _PredictionInput)
    register(StandardToolName.CALIBRATE_PREDICTION_INTERVAL, _ConformalInput)
    register(StandardToolName.GENERATE_AUDITED_REPORT, _ReportInput)
    return ToolInvocationService(registry=registry), calls


def test_postgresql_migrations_reach_runtime_v7_head(
    postgres_runtime: tuple[Engine, SessionFactory],
) -> None:
    engine, _ = postgres_runtime
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0015"


def test_fixed_nine_step_agent_persists_exact_results_on_postgresql(
    postgres_runtime: tuple[Engine, SessionFactory],
) -> None:
    _, session_factory = postgres_runtime
    principal = _principal(session_factory)
    project = ProjectService(session_factory).create_project(
        principal,
        name="Runtime V7 PostgreSQL",
        now=NOW,
    )
    dataset_service = DatasetService(session_factory)
    dataset = dataset_service.create_dataset(
        principal,
        project_id=project.project_id,
        name="Runtime V7 PostgreSQL dataset",
        data_version="runtime-v7-postgres-test-data-v1",
        schema_version="canonical-v1",
        now=NOW,
    )
    frozen = dataset_service.freeze_dataset(principal, dataset.dataset_id, now=NOW)
    record_batch_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            RecordBatchBinding(
                id=record_batch_id,
                binding_schema_version="record-batch-binding-v1",
                content_batch_id="sha256:" + "1" * 64,
                project_id=project.project_id,
                dataset_id=frozen.dataset_id,
                source_manifest_sha256="2" * 64,
                registration_sha256="3" * 64,
                content_dataset_id="POSTGRES_RUNTIME_V7",
                dataset_schema_version="canonical-v1",
                cell_id="postgres-runtime-v7-cell",
                cutoff_cycle=100,
                data_version="runtime-v7-postgres-test-data-v1",
                split_version="cell-split-v1",
                feature_version="runtime-v7-postgres-test-features-v1",
                created_by_user_id=principal.user_id,
                created_at=NOW,
            )
        )

    tool_service, calls = _tool_service()
    available_tools = (
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
        StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        StandardToolName.GENERATE_AUDITED_REPORT,
    )
    run_service = AgentRunService(
        session_factory,
        planner=SupervisorPlanner(gateway=None, clock=lambda: NOW),
    )
    run = run_service.create_run(
        principal,
        request=SupervisorPlanningRequest(
            project_id=project.project_id,
            user_goal="verify the fixed Runtime V7 Agent",
            dataset_ids=(record_batch_id,),
            requested_outputs=("advanced_single_cell_analysis",),
        ),
        idempotency_key=f"postgres-runtime-v7-{uuid4()}",
        available_tools=available_tools,
        now=NOW,
    )
    worker = AgentRunExecutionWorker(
        session_factory,
        run_service=run_service,
        tool_service=tool_service,
        context_resolver=_RuntimeV7Context(),
        clock=lambda: NOW,
    )

    completed = worker.execute(run_id=run.run_id, plan_hash=run.plan.plan_hash)
    repeated = worker.execute(run_id=run.run_id, plan_hash=run.plan.plan_hash)

    assert completed.status is AgentRunStatus.COMPLETED
    assert repeated == completed
    assert len(completed.completed_step_ids) == 9
    assert len(calls) == 9
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(AgentStep).where(AgentStep.run_id == run.run_id)
        ) == 9
        assert session.scalar(
            select(func.count())
            .select_from(ToolResultRecord)
            .where(ToolResultRecord.run_id == run.run_id)
        ) == 9
        assert session.scalar(
            select(func.count())
            .select_from(ProvenanceRecordRow)
            .join(ToolResultRecord)
            .where(ToolResultRecord.run_id == run.run_id)
        ) == 9
