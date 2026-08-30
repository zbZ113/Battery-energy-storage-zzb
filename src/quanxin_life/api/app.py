"""Optional FastAPI transport adapter for the shared tool invocation service.

Importing this module never imports FastAPI.  The factory fails explicitly when
the optional API dependency group is absent, while the domain service remains
usable by MCP, command-line and test callers.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    ProjectInvocationNotFoundError,
)
from quanxin_life.application.lifetime_workflow import (
    LifetimeDecisionWorkflowRequest,
    LifetimeDecisionWorkflowResult,
)
from quanxin_life.audit import AuditLedgerError
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.tools import (
    StandardToolName,
    ToolAuthorizationError,
    ToolContractError,
    ToolExecutionError,
    ToolInputValidationError,
    UnknownToolError,
)


class FastApiDependencyUnavailable(RuntimeError):
    """Raised when an API host is requested without the optional dependency group."""


LifetimeWorkflowRunner = Callable[
    [ToolInvocationService, LifetimeDecisionWorkflowRequest],
    LifetimeDecisionWorkflowResult,
]
DOMAIN_ROUTE_TOOL_MAP: dict[str, StandardToolName] = {
    "/v1/analyses/quality": StandardToolName.VALIDATE_BATTERY_DATA,
    "/v1/predictions/lifetime": StandardToolName.PREDICT_CYCLE_LIFE,
    "/v1/predictions/trajectory": StandardToolName.PREDICT_SOH_TRAJECTORY,
    "/v1/predictions/update": StandardToolName.UPDATE_CELL_PARAMETERS,
    "/v1/physics/check": StandardToolName.CHECK_OPERATING_CONDITION,
    "/v1/experiments/recommend": StandardToolName.RECOMMEND_NEXT_EXPERIMENT,
    "/v1/decisions/batch": StandardToolName.MAKE_BATCH_DECISION,
}


def create_fastapi_app(
    service: ToolInvocationService,
    *,
    lifetime_workflow_runner: LifetimeWorkflowRunner | None = None,
    auth_adapter: Any | None = None,
    project_adapter: Any | None = None,
    dataset_adapter: Any | None = None,
    record_batch_adapter: Any | None = None,
    agent_run_adapter: Any | None = None,
    analysis_catalog_adapter: Any | None = None,
    knowledge_adapter: Any | None = None,
    feishu_adapter: Any | None = None,
    aily_adapter: Any | None = None,
    aily_mcp_adapter: Any | None = None,
    industrial_adapter: Any | None = None,
    experiment_adapter: Any | None = None,
    model_artifact_adapter: Any | None = None,
    model_route_adapter: Any | None = None,
    advanced_calibration_adapter: Any | None = None,
    project_report_adapter: Any | None = None,
    report_exporter: Any | None = None,
    project_invocation_context_service: ProjectInvocationContextService | None = None,
) -> Any:
    """Create the HTTP adapter without duplicating domain-tool execution logic."""
    try:
        fastapi_module = importlib.import_module("fastapi")
    except ModuleNotFoundError as exc:  # pragma: no cover
        message = "FastAPI support requires installing the 'quanxin-life[api]' extra"
        raise FastApiDependencyUnavailable(message) from exc

    app_options: dict[str, object] = {
        "title": "Hiro Tool API",
        "version": "v1",
    }
    if aily_mcp_adapter is not None:
        lifespan = getattr(aily_mcp_adapter, "lifespan", None)
        if not callable(lifespan):
            raise ValueError("aily_mcp_adapter must provide a parent lifespan")
        app_options["lifespan"] = lifespan
    app: Any = fastapi_module.FastAPI(**app_options)
    ready_user_dependencies: list[Any] = []
    operator_dependencies: list[Any] = []
    admin_dependencies: list[Any] = []
    operator_principal_dependency: Any | None = None
    if auth_adapter is not None:
        cors_module = importlib.import_module("fastapi.middleware.cors")
        app.add_middleware(
            cors_module.CORSMiddleware,
            allow_origins=list(auth_adapter.allowed_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type", "Idempotency-Key", "Origin"],
        )
        app.include_router(auth_adapter.router)
        ready_user_dependencies = [
            fastapi_module.Depends(auth_adapter.require_ready_user)
        ]
        operator_dependency = auth_adapter.require_roles(
            {UserRole.ADMIN, UserRole.MEMBER}
        )
        operator_principal_dependency = fastapi_module.Depends(operator_dependency)
        operator_dependencies = [
            operator_principal_dependency,
            fastapi_module.Depends(auth_adapter.require_trusted_origin),
        ]
        admin_dependencies = [
            fastapi_module.Depends(auth_adapter.require_roles({UserRole.ADMIN}))
        ]
    if project_adapter is not None:
        if auth_adapter is None:
            raise ValueError("project_adapter requires auth_adapter")
        app.include_router(project_adapter.router)
    if dataset_adapter is not None:
        if auth_adapter is None:
            raise ValueError("dataset_adapter requires auth_adapter")
        app.include_router(dataset_adapter.router)
    if record_batch_adapter is not None:
        if auth_adapter is None:
            raise ValueError("record_batch_adapter requires auth_adapter")
        app.include_router(record_batch_adapter.router)
    if agent_run_adapter is not None:
        if auth_adapter is None:
            raise ValueError("agent_run_adapter requires auth_adapter")
        app.include_router(agent_run_adapter.router)
    if analysis_catalog_adapter is not None:
        if auth_adapter is None:
            raise ValueError("analysis_catalog_adapter requires auth_adapter")
        app.include_router(analysis_catalog_adapter.router)
    if knowledge_adapter is not None:
        if auth_adapter is None:
            raise ValueError("knowledge_adapter requires auth_adapter")
        app.include_router(knowledge_adapter.router)
    if feishu_adapter is not None:
        # Feishu callbacks authenticate their exact body bytes and deliberately
        # do not pass through browser session or CSRF dependencies.
        app.include_router(feishu_adapter.router)
    if aily_adapter is not None:
        # Aily uses its own connector Bearer credential and never inherits the
        # browser session or the generic unauthenticated MCP transport.
        app.include_router(aily_adapter.router)
    if aily_mcp_adapter is not None:
        mount_path = getattr(aily_mcp_adapter, "mount_path", None)
        asgi_app = getattr(aily_mcp_adapter, "asgi_app", None)
        if (
            not isinstance(mount_path, str)
            or not mount_path.startswith("/v1/aily/mcp/")
            or not callable(asgi_app)
        ):
            raise ValueError("aily_mcp_adapter mount contract is invalid")
        app.mount(mount_path, asgi_app)
    if industrial_adapter is not None:
        if auth_adapter is None:
            raise ValueError("industrial_adapter requires auth_adapter")
        app.include_router(industrial_adapter.router)
    if experiment_adapter is not None:
        if auth_adapter is None:
            raise ValueError("experiment_adapter requires auth_adapter")
        app.include_router(experiment_adapter.router)
    if model_artifact_adapter is not None:
        if auth_adapter is None:
            raise ValueError("model_artifact_adapter requires auth_adapter")
        app.include_router(model_artifact_adapter.router)
    if model_route_adapter is not None:
        if auth_adapter is None:
            raise ValueError("model_route_adapter requires auth_adapter")
        app.include_router(model_route_adapter.router)
    if advanced_calibration_adapter is not None:
        if auth_adapter is None:
            raise ValueError("advanced_calibration_adapter requires auth_adapter")
        app.include_router(advanced_calibration_adapter.router)
    if project_report_adapter is not None:
        if auth_adapter is None:
            raise ValueError("project_report_adapter requires auth_adapter")
        app.include_router(project_report_adapter.router)
    if project_invocation_context_service is not None and auth_adapter is None:
        raise ValueError("project_invocation_context_service requires auth_adapter")
    if project_invocation_context_service is not None:
        if service.registry.project_context_validator is not project_invocation_context_service:
            raise ValueError(
                "project tool registry must use the HTTP project context service"
            )
        if (
            service.project_audit_ledger is None
            or service.project_audit_ledger.context_validator
            is not project_invocation_context_service
        ):
            raise ValueError(
                "project audit ledger must use the HTTP project context service"
            )
    ready_route_options = (
        {"dependencies": ready_user_dependencies} if ready_user_dependencies else {}
    )
    operator_route_options = (
        {"dependencies": operator_dependencies} if operator_dependencies else {}
    )
    admin_route_options = (
        {"dependencies": admin_dependencies} if admin_dependencies else {}
    )

    @app.get("/health")  # type: ignore[untyped-decorator]
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "quanxin-life-tool-api"}

    @app.get("/v1/tools", **ready_route_options)  # type: ignore[untyped-decorator]
    async def list_tools() -> list[dict[str, Any]]:
        return [schema.model_dump(mode="json") for schema in service.registry.list_schemas()]

    @app.get(  # type: ignore[untyped-decorator]
        "/v1/results/{result_id}", **admin_route_options
    )
    async def get_audit_result(result_id: str) -> dict[str, Any]:
        if service.audit_ledger is None:
            raise fastapi_module.HTTPException(
                status_code=503,
                detail="audit result storage is not available in this application context",
            )
        try:
            result = service.audit_ledger.resolve_registered_result(result_id)
        except ValueError as exc:
            raise fastapi_module.HTTPException(
                status_code=404,
                detail="audit result was not found",
            ) from exc
        return result.model_dump(mode="json")

    @app.get(  # type: ignore[untyped-decorator]
        "/v1/reports/{result_id}", **admin_route_options
    )
    async def get_audited_report(result_id: str) -> dict[str, str]:
        if service.audit_ledger is None:
            raise fastapi_module.HTTPException(
                status_code=503,
                detail="audit result storage is not available in this application context",
            )
        try:
            result = service.audit_ledger.resolve_registered_result(result_id)
        except ValueError as exc:
            raise fastapi_module.HTTPException(
                status_code=404,
                detail="audited report was not found",
            ) from exc
        markdown = result.values.get("markdown")
        if (
            result.tool_name != StandardToolName.GENERATE_AUDITED_REPORT.value
            or not isinstance(markdown, str)
            or not markdown
        ):
            raise fastapi_module.HTTPException(
                status_code=404,
                detail="audited report was not found",
            )
        return {"result_id": result.result_id, "markdown": markdown}

    if report_exporter is not None:

        @app.get(  # type: ignore[untyped-decorator]
            "/v1/reports/{result_id}/artifacts/{artifact_format}",
            **admin_route_options,
        )
        async def get_audited_report_artifact(
            result_id: str,
            artifact_format: str,
        ) -> Any:
            from quanxin_life.reporting import ReportArtifactFormat

            try:
                selected_format = ReportArtifactFormat(artifact_format)
            except ValueError as exc:
                raise fastapi_module.HTTPException(
                    status_code=422,
                    detail="unsupported audited report artifact format",
                ) from exc
            try:
                artifact = report_exporter.export(result_id, selected_format)
            except RuntimeError as exc:
                raise fastapi_module.HTTPException(
                    status_code=503,
                    detail="requested report renderer is not installed",
                ) from exc
            except ValueError as exc:
                status_code = 404 if "not registered" in str(exc) else 409
                raise fastapi_module.HTTPException(
                    status_code=status_code,
                    detail="audited report artifact is unavailable",
                ) from exc
            return fastapi_module.Response(
                content=artifact.payload,
                media_type=artifact.media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{artifact.filename}"',
                    "ETag": f'"sha256:{artifact.sha256}"',
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "private, no-store",
                },
            )

    async def execute_domain_tool(
        tool_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            invocation = ToolInvocation(
                tool_name=StandardToolName(tool_name), input_value=payload
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = service.invoke(invocation)
        except ToolAuthorizationError as exc:
            raise fastapi_module.HTTPException(
                status_code=403,
                detail="tool access denied",
            ) from exc
        except ToolInputValidationError as exc:
            raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
        except UnknownToolError as exc:
            raise fastapi_module.HTTPException(
                status_code=503,
                detail="requested tool is not available in this application context",
            ) from exc
        except (AuditLedgerError, ToolContractError, ToolExecutionError) as exc:
            raise fastapi_module.HTTPException(
                status_code=500,
                detail="domain tool execution failed",
            ) from exc
        return result.model_dump(mode="json")

    @app.post(  # type: ignore[untyped-decorator]
        "/v1/tools/{tool_name}", **operator_route_options
    )
    async def invoke_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await execute_domain_tool(tool_name, payload)

    if project_invocation_context_service is not None:
        assert auth_adapter is not None
        assert operator_principal_dependency is not None
        assert service.project_audit_ledger is not None
        project_operator_principal = operator_principal_dependency
        project_ready_principal = fastapi_module.Depends(
            auth_adapter.require_ready_user
        )
        project_audit_ledger = service.project_audit_ledger

        @app.get(  # type: ignore[untyped-decorator]
            "/v1/projects/{project_id}/results/{result_id}",
        )
        async def get_project_result(
            project_id: str,
            result_id: str,
            principal: AuthPrincipal = project_ready_principal,
        ) -> dict[str, Any]:
            try:
                context = project_invocation_context_service.resolve_http(
                    principal,
                    project_id,
                )
                result = project_audit_ledger.resolve_registered_result(
                    context,
                    result_id,
                )
            except ProjectInvocationAccessError as exc:
                raise fastapi_module.HTTPException(
                    status_code=403,
                    detail="project_result_access_denied",
                ) from exc
            except ProjectInvocationNotFoundError as exc:
                raise fastapi_module.HTTPException(
                    status_code=404,
                    detail="project_scope_not_found",
                ) from exc
            except AuditLedgerError as exc:
                raise fastapi_module.HTTPException(
                    status_code=503,
                    detail="project_result_storage_unavailable",
                ) from exc
            except ValueError as exc:
                raise fastapi_module.HTTPException(
                    status_code=404,
                    detail="project_result_not_found",
                ) from exc
            return result.model_dump(mode="json")

        @app.post(  # type: ignore[untyped-decorator]
            "/v1/projects/{project_id}/tools/{tool_name}",
            dependencies=[fastapi_module.Depends(auth_adapter.require_trusted_origin)],
        )
        async def invoke_project_tool(
            project_id: str,
            tool_name: str,
            payload: dict[str, Any],
            principal: AuthPrincipal = project_operator_principal,
        ) -> dict[str, Any]:
            try:
                invocation = ToolInvocation(
                    tool_name=StandardToolName(tool_name),
                    input_value=payload,
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                context = project_invocation_context_service.resolve_http(
                    principal,
                    project_id,
                )
                result = service.invoke_in_project(invocation, context=context)
            except ProjectInvocationAccessError as exc:
                raise fastapi_module.HTTPException(
                    status_code=403,
                    detail="project_tool_access_denied",
                ) from exc
            except ProjectInvocationNotFoundError as exc:
                raise fastapi_module.HTTPException(
                    status_code=404,
                    detail="project_scope_not_found",
                ) from exc
            except ToolInputValidationError as exc:
                raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
            except UnknownToolError as exc:
                raise fastapi_module.HTTPException(
                    status_code=503,
                    detail="requested tool is not available in this application context",
                ) from exc
            except AuditLedgerError as exc:
                raise fastapi_module.HTTPException(
                    status_code=503,
                    detail="project audit storage is not available",
                ) from exc
            except ToolAuthorizationError as exc:
                raise fastapi_module.HTTPException(
                    status_code=403,
                    detail="tool access denied",
                ) from exc
            except (ToolContractError, ToolExecutionError) as exc:
                raise fastapi_module.HTTPException(
                    status_code=500,
                    detail="domain tool execution failed",
                ) from exc
            return result.model_dump(mode="json")

    def domain_tool_endpoint(
        tool_name: StandardToolName,
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def endpoint(payload: dict[str, Any]) -> dict[str, Any]:
            return await execute_domain_tool(tool_name.value, payload)

        endpoint.__name__ = f"invoke_{tool_name.value}"
        return endpoint

    for route_path, route_tool_name in DOMAIN_ROUTE_TOOL_MAP.items():
        app.post(route_path, **operator_route_options)(
            domain_tool_endpoint(route_tool_name)
        )

    if lifetime_workflow_runner is not None:

        @app.post(  # type: ignore[untyped-decorator]
            "/v1/workflows/lifetime-decision", **operator_route_options
        )
        async def run_lifetime_decision(payload: dict[str, Any]) -> dict[str, Any]:
            try:
                request = LifetimeDecisionWorkflowRequest.model_validate(payload)
            except (TypeError, ValueError, ValidationError) as exc:
                raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                result = lifetime_workflow_runner(service, request)
            except (AuditLedgerError, ToolContractError, ToolExecutionError, ValueError) as exc:
                raise fastapi_module.HTTPException(
                    status_code=500,
                    detail="lifetime decision workflow failed",
                ) from exc
            return result.model_dump(mode="json")

    return app
