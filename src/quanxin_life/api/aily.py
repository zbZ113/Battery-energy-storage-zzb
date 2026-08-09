"""Bearer-protected Aily facade over existing Agent and audit contracts."""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import AgentRunState, ToolResult
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorizer
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.reporting import AuditedReportArtifact, ReportArtifactFormat
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z")
_AILY_BEARER = HTTPBearer(auto_error=False)
_AILY_SECURITY_DEPENDENCY = Security(_AILY_BEARER)


class AilyConnectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr

    @field_validator("api_key")
    @classmethod
    def api_key_is_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Aily connector API key must not be blank")
        return value


class AilyCreateAnalysisTaskRequest(BaseModel):
    """Reference-only request; battery values remain in registered data batches."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: FeishuAnalysisTask
    data_batch_id: str = Field(min_length=1, max_length=200)

    @field_validator("data_batch_id")
    @classmethod
    def data_batch_id_is_safe(cls, value: str) -> str:
        return _reference(value, field_name="data_batch_id")


class AilyAnalysisTaskGateway(Protocol):
    def create_analysis_task(
        self, request: AilyCreateAnalysisTaskRequest
    ) -> AgentRunState: ...

    def get_analysis_task(self, run_id: str) -> AgentRunState: ...


class AilyReportExporter(Protocol):
    def export(
        self, result_id: str, format: ReportArtifactFormat
    ) -> AuditedReportArtifact: ...


@dataclass(frozen=True, slots=True)
class AilyHttpAdapter:
    router: APIRouter


def create_aily_http_adapter(
    config: AilyConnectorConfig,
    *,
    gateway: AilyAnalysisTaskGateway,
    audit_ledger: AuditLedger,
    report_exporter: AilyReportExporter,
    result_authorizer: AuditedResultAuthorizer,
) -> AilyHttpAdapter:
    """Expose four stable Aily operations without exposing model identities."""

    def require_connector(
        credentials: HTTPAuthorizationCredentials | None = _AILY_SECURITY_DEPENDENCY,
    ) -> None:
        expected = config.api_key.get_secret_value()
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(expected, credentials.credentials)
        ):
            raise HTTPException(
                status_code=401,
                detail="aily_connector_authentication_failed",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authorized = [Depends(require_connector)]
    router = APIRouter(
        prefix="/v1/aily",
        tags=["aily-connector"],
        dependencies=authorized,
    )

    @router.post(
        "/analysis-tasks",
        response_model=AgentRunState,
        status_code=202,
    )
    def create_analysis_task(
        request: AilyCreateAnalysisTaskRequest,
    ) -> AgentRunState:
        try:
            state = gateway.create_analysis_task(request)
            return AgentRunState.model_validate(state.model_dump(mode="json"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="aily_analysis_task_rejected"
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail="aily_analysis_task_unavailable"
            ) from exc

    @router.get(
        "/analysis-tasks/{run_id}",
        response_model=AgentRunState,
    )
    def get_analysis_task(run_id: str) -> AgentRunState:
        return _resolve_run(gateway, run_id)

    @router.get(
        "/analysis-tasks/{run_id}/results/{result_id}",
        response_model=ToolResult,
    )
    def get_audited_result(run_id: str, result_id: str) -> ToolResult:
        state = _resolve_run(gateway, run_id)
        checked_result_id = _bound_result_id(state, result_id)
        try:
            result = audit_ledger.resolve_registered_result(checked_result_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404, detail="aily_audited_result_not_found"
            ) from exc
        _authorize_result(result_authorizer, result)
        return result

    @router.get("/analysis-tasks/{run_id}/reports/{report_result_id}")
    def get_audited_report(run_id: str, report_result_id: str) -> Response:
        state = _resolve_run(gateway, run_id)
        checked_result_id = _bound_result_id(state, report_result_id)
        try:
            report_result = audit_ledger.resolve_registered_result(checked_result_id)
            _require_audited_report_result(report_result)
            _authorize_result(result_authorizer, report_result)
            artifact = report_exporter.export(
                checked_result_id, ReportArtifactFormat.MARKDOWN
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=404, detail="aily_audited_report_not_found"
            ) from exc
        if (
            artifact.source_result_id != checked_result_id
            or artifact.format is not ReportArtifactFormat.MARKDOWN
            or sha256(artifact.payload).hexdigest() != artifact.sha256
            or _SAFE_FILENAME.fullmatch(artifact.filename) is None
        ):
            raise HTTPException(
                status_code=409, detail="aily_audited_report_invalid"
            )
        return Response(
            content=artifact.payload,
            media_type=artifact.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{artifact.filename}"',
                "ETag": f'"sha256:{artifact.sha256}"',
                "X-Tool-Result-Id": checked_result_id,
            },
        )

    return AilyHttpAdapter(router=router)


def _resolve_run(gateway: AilyAnalysisTaskGateway, run_id: str) -> AgentRunState:
    try:
        checked_run_id = _reference(run_id, field_name="run_id")
        state = gateway.get_analysis_task(checked_run_id)
        return AgentRunState.model_validate(state.model_dump(mode="json"))
    except (AttributeError, LookupError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=404, detail="aily_analysis_task_not_found"
        ) from exc


def _bound_result_id(state: AgentRunState, result_id: str) -> str:
    try:
        checked_result_id = _reference(result_id, field_name="result_id")
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail="aily_audited_result_not_found"
        ) from exc
    if checked_result_id not in state.result_ids:
        raise HTTPException(
            status_code=404, detail="aily_audited_result_not_found"
        )
    return checked_result_id


def _authorize_result(
    authorizer: AuditedResultAuthorizer,
    result: ToolResult,
) -> None:
    try:
        authorization = authorizer.authorize(result)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "aily_audited_result_rejected",
                "reason": "RESULT_AUTHORIZATION_INVALID",
            },
        ) from exc
    if authorization.allowed:
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": "aily_audited_result_rejected",
            "reason": authorization.rejection_reason
            or "RESULT_DISPLAY_NOT_AUTHORIZED",
        },
    )


def _require_audited_report_result(result: ToolResult) -> None:
    """Reject run-bound results that are not produced by the audited report tool."""
    if (
        result.tool_name != AUDITED_REPORT_TOOL_NAME
        or result.tool_version != AUDITED_REPORT_TOOL_VERSION
        or result.model_version != REPORTING_VERSION
    ):
        raise ValueError("result is not an audited report ToolResult")


def _reference(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe machine reference")
    return normalized


__all__ = [
    "AilyAnalysisTaskGateway",
    "AilyConnectorConfig",
    "AilyCreateAnalysisTaskRequest",
    "AilyHttpAdapter",
    "AilyReportExporter",
    "create_aily_http_adapter",
]
