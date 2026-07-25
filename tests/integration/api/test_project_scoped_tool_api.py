from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import Field

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.audit import ProjectAuditLedger
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthPrincipal,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import (
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import Project, User
from quanxin_life.tools import (
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
PASSWORD = "ProjectScopedTools-2026!"


class _ProjectToolInput(ContractModel):
    batch_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _ApiContext:
    owner_client: TestClient
    outsider_client: TestClient
    invocation_service: ToolInvocationService
    project_audit_ledger: ProjectAuditLedger
    executor_calls: list[tuple[_ProjectToolInput, VerifiedProjectInvocationContext]]
    owner_user_id: str
    active_project_id: str
    archived_project_id: str


def _principal(user_id: str, username: str) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=username,
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200


def _api_context(tmp_path: Path) -> _ApiContext:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'project-tools.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    owner_user_id = str(uuid4())
    outsider_user_id = str(uuid4())
    owner_username = f"project-owner-{owner_user_id[:8]}@example.test"
    outsider_username = f"project-outsider-{outsider_user_id[:8]}@example.test"
    with session_factory.begin() as session:
        for user_id, username in (
            (owner_user_id, owner_username),
            (outsider_user_id, outsider_username),
        ):
            session.add(
                User(
                    id=user_id,
                    username=username,
                    credential_hash=hasher.hash_password(PASSWORD),
                    must_change_credential=False,
                    role=UserRole.MEMBER.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )

    project_service = ProjectService(session_factory)
    owner_principal = _principal(owner_user_id, owner_username)
    active_project = project_service.create_project(
        owner_principal,
        name="Project-scoped tool active fixture",
        now=NOW,
    )
    archived_project = project_service.create_project(
        owner_principal,
        name="Project-scoped tool archived fixture",
        now=NOW,
    )
    with session_factory.begin() as session:
        persisted = session.get(Project, archived_project.project_id)
        assert persisted is not None
        persisted.status = ProjectStatus.ARCHIVED.value
        persisted.updated_at = NOW

    project_context_service = ProjectInvocationContextService(session_factory)
    executor_calls: list[tuple[_ProjectToolInput, VerifiedProjectInvocationContext]] = []
    registry = ToolRegistry(project_context_validator=project_context_service)

    def executor(
        value: _ProjectToolInput,
        invocation_context: VerifiedProjectInvocationContext,
    ) -> ToolResult:
        executor_calls.append((value, invocation_context))
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
            tool_version="project-scoped-http-test-v1",
            model_version="project-scoped-http-model-v1",
            data_version="project-scoped-http-data-v1",
            feature_version="project-scoped-http-feature-v1",
            input_hash=sha256_canonical(value.model_dump(mode="json")),
            values={"validated_batch_id": value.batch_id},
            provenance=[
                ProvenanceRecord(
                    source_id="project-scoped-http-fixture",
                    source_kind=SourceKind.OBSERVED,
                    uri="test://project-scoped-http/fixture",
                    sha256=sha256_canonical({"fixture": "project-scoped-http"}),
                    description="Non-numeric project-scoped HTTP test fixture",
                    created_at=NOW,
                )
            ],
            created_at=NOW,
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="project-scoped-http-test-v1",
            input_model=_ProjectToolInput,
            execution_scope=ToolExecutionScope.PROJECT,
            executor=None,
            project_executor=executor,
        )
    )
    project_audit_ledger = ProjectAuditLedger(
        context_validator=project_context_service
    )
    invocation_service = ToolInvocationService(
        registry=registry,
        project_audit_ledger=project_audit_ledger,
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
    app = create_fastapi_app(
        invocation_service,
        auth_adapter=auth_adapter,
        project_invocation_context_service=project_context_service,
    )
    owner_client = TestClient(app, base_url="https://api.example.test")
    outsider_client = TestClient(app, base_url="https://api.example.test")
    _login(owner_client, owner_username)
    _login(outsider_client, outsider_username)
    return _ApiContext(
        owner_client=owner_client,
        outsider_client=outsider_client,
        invocation_service=invocation_service,
        project_audit_ledger=project_audit_ledger,
        executor_calls=executor_calls,
        owner_user_id=owner_user_id,
        active_project_id=active_project.project_id,
        archived_project_id=archived_project.project_id,
    )


def test_authenticated_project_tool_uses_server_resolved_invocation_context(
    tmp_path: Path,
) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/projects/{context.active_project_id}/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={"batch_id": "batch-owned-by-project"},
    )

    assert response.status_code == 200
    assert response.json()["values"] == {
        "validated_batch_id": "batch-owned-by-project"
    }
    assert len(context.executor_calls) == 1
    validated_input, verified_context = context.executor_calls[0]
    assert validated_input == _ProjectToolInput(batch_id="batch-owned-by-project")
    assert verified_context.project_id == context.active_project_id
    assert verified_context.actor_user_id == context.owner_user_id
    assert verified_context.actor_session_id
    result = context.project_audit_ledger.resolve_registered_result(
        verified_context,
        response.json()["result_id"],
    )
    assert result.values == {"validated_batch_id": "batch-owned-by-project"}


def test_invisible_and_inactive_projects_are_hidden_without_tool_execution(
    tmp_path: Path,
) -> None:
    context = _api_context(tmp_path)

    invisible = context.outsider_client.post(
        f"/v1/projects/{context.active_project_id}/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={"batch_id": "must-not-execute-invisible"},
    )
    inactive = context.owner_client.post(
        f"/v1/projects/{context.archived_project_id}/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={"batch_id": "must-not-execute-inactive"},
    )

    assert invisible.status_code == 404
    assert inactive.status_code == 404
    assert context.executor_calls == []


def test_project_tool_requires_trusted_origin_before_execution(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/projects/{context.active_project_id}/tools/validate_battery_data",
        json={"batch_id": "must-not-execute-cross-site"},
    )

    assert response.status_code == 403
    assert context.executor_calls == []


def test_payload_cannot_inject_a_project_identifier(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        f"/v1/projects/{context.active_project_id}/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={
            "batch_id": "must-not-execute-injected-project",
            "project_id": str(uuid4()),
        },
    )

    assert response.status_code == 422
    assert context.executor_calls == []


def test_generic_http_route_cannot_execute_project_tool(tmp_path: Path) -> None:
    context = _api_context(tmp_path)

    response = context.owner_client.post(
        "/v1/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={"batch_id": "must-not-execute-without-project"},
    )

    assert response.status_code == 403
    assert context.executor_calls == []
