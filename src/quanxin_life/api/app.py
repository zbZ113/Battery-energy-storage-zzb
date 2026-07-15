"""Optional FastAPI transport adapter for the shared tool invocation service.

Importing this module never imports FastAPI.  The factory fails explicitly when
the optional API dependency group is absent, while the domain service remains
usable by MCP, command-line and test callers.
"""

from __future__ import annotations

import importlib
from base64 import b64decode
from binascii import Error as Base64DecodeError
from collections.abc import Callable
from typing import Any

from pydantic import Field, ValidationError

from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.application.ingestion import (
    MAX_CANONICAL_CSV_BYTES,
    CanonicalCsvBatchRegistration,
)
from quanxin_life.application.lifetime_workflow import (
    LifetimeDecisionWorkflowRequest,
    LifetimeDecisionWorkflowResult,
)
from quanxin_life.audit import AuditLedgerError
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel
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
CanonicalCsvRegistrar = Callable[[bytes, CanonicalCsvBatchRegistration], str]


class CanonicalCsvUploadRequest(ContractModel):
    """JSON-safe transport envelope for a canonical CSV byte payload."""

    payload_base64: str = Field(min_length=1)
    registration: CanonicalCsvBatchRegistration

    def decoded_payload(self) -> bytes:
        try:
            payload = b64decode(self.payload_base64, validate=True)
        except (Base64DecodeError, ValueError) as exc:
            raise ValueError("payload_base64 must be valid base64") from exc
        if not payload:
            raise ValueError("decoded canonical CSV payload must not be empty")
        if len(payload) > MAX_CANONICAL_CSV_BYTES:
            raise ValueError("decoded canonical CSV payload exceeds the configured size limit")
        return payload


def create_fastapi_app(
    service: ToolInvocationService,
    *,
    lifetime_workflow_runner: LifetimeWorkflowRunner | None = None,
    canonical_csv_registrar: CanonicalCsvRegistrar | None = None,
    auth_adapter: Any | None = None,
    project_adapter: Any | None = None,
) -> Any:
    """Create the HTTP adapter without duplicating domain-tool execution logic."""
    try:
        fastapi_module = importlib.import_module("fastapi")
    except ModuleNotFoundError as exc:  # pragma: no cover
        message = "FastAPI support requires installing the 'quanxin-life[api]' extra"
        raise FastApiDependencyUnavailable(message) from exc

    app: Any = fastapi_module.FastAPI(title="泉芯智寿 Tool API", version="v1")
    ready_user_dependencies: list[Any] = []
    operator_dependencies: list[Any] = []
    admin_dependencies: list[Any] = []
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
        operator_dependencies = [
            fastapi_module.Depends(
                auth_adapter.require_roles({UserRole.ADMIN, UserRole.MEMBER})
            ),
            fastapi_module.Depends(auth_adapter.require_trusted_origin),
        ]
        admin_dependencies = [
            fastapi_module.Depends(auth_adapter.require_roles({UserRole.ADMIN}))
        ]
    if project_adapter is not None:
        if auth_adapter is None:
            raise ValueError("project_adapter requires auth_adapter")
        app.include_router(project_adapter.router)
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

    @app.post(  # type: ignore[untyped-decorator]
        "/v1/tools/{tool_name}", **operator_route_options
    )
    async def invoke_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
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

    if canonical_csv_registrar is not None:

        @app.post(  # type: ignore[untyped-decorator]
            "/v1/batches/canonical-csv", **operator_route_options
        )
        async def register_canonical_csv(payload: dict[str, Any]) -> dict[str, str]:
            try:
                request = CanonicalCsvUploadRequest.model_validate(payload)
                raw_payload = request.decoded_payload()
            except (TypeError, ValueError, ValidationError) as exc:
                raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                record_batch_id = canonical_csv_registrar(
                    raw_payload,
                    request.registration,
                )
            except ValueError as exc:
                raise fastapi_module.HTTPException(status_code=422, detail=str(exc)) from exc
            if not isinstance(record_batch_id, str) or not record_batch_id.strip():
                raise fastapi_module.HTTPException(
                    status_code=500,
                    detail="canonical CSV registrar returned an invalid record batch identifier",
                )
            return {"record_batch_id": record_batch_id}

    return app
