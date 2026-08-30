"""Protected Streamable HTTP MCP surface for the audited Aily workflow."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from ipaddress import ip_address
from numbers import Real
from typing import Any, Literal, Protocol

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from quanxin_life.api.aily import (
    AilyAnalysisSourceReference,
    AilyAnalysisTaskGateway,
    AilyCreateAnalysisTaskRequest,
    AilyCreateRecheckActionRequest,
    AilyCreateScenarioContextRequest,
    AilyRecheckActionGateway,
    AilyRecheckActionState,
    AilyReportExporter,
    AilyResultResolver,
    AilyScenarioContextGateway,
    AilyScenarioContextState,
)
from quanxin_life.application.aily_mcp_authorization import (
    ResolvedAilyAnalysisSource,
)
from quanxin_life.core import AgentRunState, ToolResult
from quanxin_life.integrations.feishu.analysis_bitable import (
    AnalysisBitableProjectionError,
    build_audited_analysis_bitable_fields,
)
from quanxin_life.integrations.feishu.bitable import (
    CHINESE_ANALYSIS_BITABLE_PROFILE,
    BitableConflictError,
    BitableValidationError,
)
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorizer
from quanxin_life.integrations.feishu.recheck_actions import (
    RecheckActionAuthorizationError,
    RecheckActionInProgressError,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.reporting import ReportArtifactFormat
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
    RECOMMENDATION_REPORT_RENDERER_VERSION,
)

_MOUNT_PREFIX = "/v1/aily/mcp"
_SOURCE_IP_HEADER = b"x-quanxin-aily-source-ip"
_AILY_USER_HEADER = b"x-aily-user"
_AILY_EMAIL_HEADER = b"x-aily-email"
_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}\Z")
_SAFE_ENDPOINT_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,128}\Z")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z")
_SAFE_EMAIL = re.compile(r"[^@\s\x00-\x1f]{1,128}@[^@\s\x00-\x1f]{1,190}\Z")
_EMAIL_LESS_DISCOVERY_METHODS = frozenset(
    {"initialize", "notifications/initialized", "ping", "tools/list"}
)
_LOGGER = logging.getLogger(__name__)
_SUPPORTED_ANALYSIS_TASKS = Literal[
    FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
    FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
    FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
]


class AilyMcpConfig(BaseModel):
    """Fail-closed inputs for one dedicated Aily Streamable HTTP endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_token: SecretStr
    allowed_source_ips: tuple[str, ...] = Field(min_length=1, max_length=256)
    allowed_hosts: tuple[str, ...] = Field(min_length=1, max_length=32)
    trust_gateway_source_ip: bool = False
    max_request_bytes: int = Field(default=256 * 1024, ge=1024, le=1024 * 1024)
    max_response_chars: int = Field(default=20_000, ge=1_000, le=20_000)

    @field_validator("endpoint_token")
    @classmethod
    def endpoint_token_is_an_opaque_path_segment(cls, value: SecretStr) -> SecretStr:
        if _SAFE_ENDPOINT_TOKEN.fullmatch(value.get_secret_value().strip()) is None:
            raise ValueError("Aily MCP endpoint token must be a high-entropy path segment")
        return value

    @field_validator("allowed_source_ips")
    @classmethod
    def source_ips_are_exact_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            if "/" in value:
                raise ValueError("Aily MCP source IPs must be exact addresses")
            try:
                checked = str(ip_address(value.strip()))
            except ValueError as exc:
                raise ValueError("Aily MCP source IP is invalid") from exc
            normalized.append(checked)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Aily MCP source IPs must be unique")
        return tuple(normalized)

    @field_validator("allowed_hosts")
    @classmethod
    def hosts_are_exact_authorities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip().lower() for value in values)
        if any(
            not value
            or len(value) > 255
            or "://" in value
            or "/" in value
            or any(character.isspace() for character in value)
            for value in normalized
        ):
            raise ValueError("Aily MCP allowed hosts must be exact authorities")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Aily MCP allowed hosts must be unique")
        return normalized

    @property
    def mount_path(self) -> str:
        return f"{_MOUNT_PREFIX}/{self.endpoint_token.get_secret_value().strip()}"


class AilyMcpCallerAuthorizer(Protocol):
    def authorize_identity(self, *, aily_user_id: str) -> None: ...

    def authorize_run_reference(
        self,
        *,
        aily_user_id: str,
        run_id: str,
    ) -> None: ...

    def resolve_analysis_source(
        self,
        *,
        aily_user_id: str,
        task_label: str,
    ) -> ResolvedAilyAnalysisSource: ...


@dataclass(frozen=True, slots=True)
class AilyMcpAdapter:
    server: FastMCP[Any]
    asgi_app: ASGIApp
    mount_path: str = field(repr=False)
    lifespan: Callable[[Any], Any]


def create_aily_mcp_adapter(
    config: AilyMcpConfig,
    *,
    gateway: AilyAnalysisTaskGateway,
    scenario_context_gateway: AilyScenarioContextGateway,
    audit_ledger: AilyResultResolver,
    report_exporter: AilyReportExporter,
    result_authorizer: AuditedResultAuthorizer,
    caller_authorizer: AilyMcpCallerAuthorizer,
    recheck_action_gateway: AilyRecheckActionGateway | None = None,
) -> AilyMcpAdapter:
    """Create one Aily-only MCP server without exposing the generic registry."""

    server: FastMCP[Any] = FastMCP(
        "Hiro 电芯寿命研发助手",
        instructions=(
            "只编排受审计工具并解释工具结果。不得生成、补写或修改 SOH、RUL、"
            "寿命、置信区间或经营指标。CyclePatch 个体预测与 BLAST 参考工况推演"
            "是独立工具, 不得互相换算。"
        ),
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(config.allowed_hosts),
            allowed_origins=[],
        ),
    )

    @server.tool(
        name="quanxin_resolve_analysis_source",
        description=(
            "按 `电芯ID | cutoff-N` 任务标签解析当前调用者拥有的成功飞书上传。"
            "返回后续工具所需的 source_run_id 与 data_batch_id; "
            "cutoff 仅从绑定的受审循环寿命 ToolResult 复核, 不从文件名猜测。"
        ),
        structured_output=True,
    )
    async def resolve_analysis_source(
        task_label: str,
        ctx: Context[Any, Any, Any],
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        try:
            resolved = caller_authorizer.resolve_analysis_source(
                aily_user_id=user_id,
                task_label=task_label,
            )
            response = AilyAnalysisSourceReference(
                task_label=resolved.task_label,
                source_run_id=resolved.source_run_id,
                data_batch_id=resolved.data_batch_id,
                cell_id=resolved.cell_id,
                cutoff_cycle=resolved.cutoff_cycle,
            ).model_dump(mode="json")
            return _bounded_mcp_result(response, config.max_response_chars)
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("任务标签未获授权或没有匹配的受审分析") from exc

    @server.tool(
        name="quanxin_create_scenario_context",
        description=(
            "创建受约束的参考工况上下文。温度、倍率、SOC、DoD 等仅作为工程师"
            "声明的工况假设; 不得提交任何预测输出。"
        ),
        structured_output=True,
    )
    async def create_scenario_context(
        request: AilyCreateScenarioContextRequest,
        ctx: Context[Any, Any, Any],
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        _authorize_caller(caller_authorizer, user_id, request.source_run_id)
        try:
            state = scenario_context_gateway.create_scenario_context(request)
            response = AilyScenarioContextState.model_validate(
                state.model_dump(mode="json")
            ).model_dump(mode="json")
            return _bounded_mcp_result(response, config.max_response_chars)
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("工况上下文未获授权或无法创建") from exc

    @server.tool(
        name="quanxin_create_analysis_task",
        description=(
            "根据已登记的数据批次或工况上下文创建异步分析任务。返回后应轮询任务"
            "状态; 不得在本次调用内同步运行模型。"
        ),
        structured_output=True,
    )
    async def create_analysis_task(
        task_type: _SUPPORTED_ANALYSIS_TASKS,
        source_run_id: str,
        ctx: Context[Any, Any, Any],
        data_batch_id: str | None = None,
        scenario_context_id: str | None = None,
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        _authorize_caller(caller_authorizer, user_id, source_run_id)
        try:
            request = AilyCreateAnalysisTaskRequest(
                task_type=task_type,
                source_run_id=source_run_id,
                data_batch_id=data_batch_id,
                scenario_context_id=scenario_context_id,
            )
            state = gateway.create_analysis_task(request)
            response = AgentRunState.model_validate(
                state.model_dump(mode="json")
            ).model_dump(mode="json")
            return _bounded_mcp_result(response, config.max_response_chars)
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("分析任务未获授权或无法创建") from exc

    @server.tool(
        name="quanxin_get_analysis_task",
        description="查询已授权异步分析任务状态。任务完成前继续轮询, 不得补写结果。",
        structured_output=True,
    )
    async def get_analysis_task(
        run_id: str,
        ctx: Context[Any, Any, Any],
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        _authorize_caller(caller_authorizer, user_id, run_id)
        response = _resolve_run(gateway, run_id).model_dump(mode="json")
        return _bounded_mcp_result(response, config.max_response_chars)

    @server.tool(
        name="quanxin_get_audited_result",
        description=(
            "读取与任务绑定且通过展示授权的受审结果。大型曲线数组不会直接返回; "
            "请结合受审报告或飞书曲线附件解释。"
        ),
        structured_output=True,
    )
    async def get_audited_result(
        run_id: str,
        result_id: str,
        ctx: Context[Any, Any, Any],
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        _authorize_caller(caller_authorizer, user_id, run_id)
        state = _resolve_run(gateway, run_id)
        result = _resolve_bound_result(
            state=state,
            result_id=result_id,
            audit_ledger=audit_ledger,
        )
        authorization = _authorize_result(result_authorizer, result)
        response = _audited_result_projection(
            run_id=state.run_id,
            result=result,
            route_id=authorization.route_id,
            activation_status=authorization.activation_status,
            evidence_level=authorization.evidence_level.value,
            supported_domain=authorization.supported_domain,
            max_response_chars=config.max_response_chars,
        )
        return _bounded_mcp_result(response, config.max_response_chars)

    @server.tool(
        name="quanxin_get_audited_report",
        description="读取与任务绑定、完整性校验通过的中文受审 Markdown 报告。",
        structured_output=True,
    )
    async def get_audited_report(
        run_id: str,
        report_result_id: str,
        ctx: Context[Any, Any, Any],
    ) -> CallToolResult:
        user_id = _verified_aily_user(ctx)
        _authorize_caller(caller_authorizer, user_id, run_id)
        state = _resolve_run(gateway, run_id)
        report_result = _resolve_bound_result(
            state=state,
            result_id=report_result_id,
            audit_ledger=audit_ledger,
        )
        _require_audited_report_result(report_result)
        _authorize_result(result_authorizer, report_result)
        try:
            artifact = report_exporter.export(
                report_result.result_id,
                ReportArtifactFormat.MARKDOWN,
            )
            content = artifact.payload.decode("utf-8")
        except (LookupError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            raise ValueError("受审报告不可用") from exc
        if (
            artifact.source_result_id != report_result.result_id
            or artifact.format is not ReportArtifactFormat.MARKDOWN
            or sha256(artifact.payload).hexdigest() != artifact.sha256
            or _SAFE_FILENAME.fullmatch(artifact.filename) is None
        ):
            raise ValueError("受审报告完整性校验失败")
        response: dict[str, object] = {
            "任务ID": state.run_id,
            "报告结果ID": report_result.result_id,
            "文件名": artifact.filename,
            "文件SHA256": artifact.sha256,
            "内容": content,
        }
        return _bounded_mcp_result(response, config.max_response_chars)

    if recheck_action_gateway is not None:

        @server.tool(
            name="quanxin_create_recheck_action",
            description=(
                "根据受审建议结果创建一次幂等复检动作。只接受任务、结果和责任人引用。"
            ),
            structured_output=True,
        )
        async def create_recheck_action(
            request: AilyCreateRecheckActionRequest,
            ctx: Context[Any, Any, Any],
        ) -> CallToolResult:
            user_id = _verified_aily_user(ctx)
            _authorize_caller(caller_authorizer, user_id, request.source_run_id)
            try:
                receipt = recheck_action_gateway.create_recheck_action(
                    source_run_id=request.source_run_id,
                    source_result_id=request.source_result_id,
                    responsibility_reference=request.responsibility_reference,
                )
                response = AilyRecheckActionState.model_validate(
                    {
                        "action_key": receipt.action_key,
                        "record_id": receipt.record_id,
                        "source_run_id": receipt.source_run_id,
                        "source_result_id": receipt.source_result_id,
                        "action_type": receipt.action_type,
                        "responsibility_reference": receipt.responsibility_reference,
                        "permission_reference": receipt.permission_reference,
                        "status": receipt.status,
                        "created_at": receipt.created_at,
                        "updated_at": receipt.updated_at,
                    }
                ).model_dump(mode="json")
                return _bounded_mcp_result(response, config.max_response_chars)
            except (
                BitableConflictError,
                BitableValidationError,
                RecheckActionAuthorizationError,
                RecheckActionInProgressError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError("复检动作未获授权或无法创建") from exc

    inner_app = server.streamable_http_app()
    secured_app: ASGIApp = _AilyMcpSecurityMiddleware(
        inner_app,
        allowed_source_ips=frozenset(config.allowed_source_ips),
        trust_gateway_source_ip=config.trust_gateway_source_ip,
        max_request_bytes=config.max_request_bytes,
        caller_authorizer=caller_authorizer,
    )

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        async with server.session_manager.run():
            yield

    return AilyMcpAdapter(
        server=server,
        asgi_app=secured_app,
        mount_path=config.mount_path,
        lifespan=lifespan,
    )


class _AilyMcpSecurityMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_source_ips: frozenset[str],
        trust_gateway_source_ip: bool,
        max_request_bytes: int,
        caller_authorizer: AilyMcpCallerAuthorizer,
    ) -> None:
        self._app = app
        self._allowed_source_ips = allowed_source_ips
        self._trust_gateway_source_ip = trust_gateway_source_ip
        self._max_request_bytes = max_request_bytes
        self._caller_authorizer = caller_authorizer

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        source_ip = _single_header(scope, _SOURCE_IP_HEADER)
        user_id = _single_header(scope, _AILY_USER_HEADER)
        email = _single_header(scope, _AILY_EMAIL_HEADER)
        source_ip_allowed = _source_ip_is_allowed(
            source_ip,
            allowed_source_ips=self._allowed_source_ips,
            trust_gateway_source_ip=self._trust_gateway_source_ip,
        )
        if (
            not source_ip_allowed
            or _SAFE_REFERENCE.fullmatch(user_id or "") is None
            or (email is not None and _SAFE_EMAIL.fullmatch(email) is None)
        ):
            _log_security_rejection(
                scope=scope,
                reason_code="REQUEST_METADATA_REJECTED",
                source_ip_allowed=source_ip_allowed,
                user_id=user_id,
            )
            await _send_json_error(send, 403, "mcp_request_rejected")
            return
        messages, size_ok = await _buffer_request(receive, self._max_request_bytes)
        if not size_ok:
            await _send_json_error(send, 413, "mcp_request_too_large")
            return
        assert user_id is not None
        try:
            self._caller_authorizer.authorize_identity(aily_user_id=user_id)
        except (LookupError, RuntimeError, TypeError, ValueError):
            # Aily probes discovery with an unbound identity and no email.
            if (
                email is not None
                or _mcp_request_method(messages) not in _EMAIL_LESS_DISCOVERY_METHODS
            ):
                _log_security_rejection(
                    scope=scope,
                    reason_code="IDENTITY_NOT_BOUND",
                    source_ip_allowed=source_ip_allowed,
                    user_id=user_id,
                )
                await _send_json_error(send, 403, "mcp_request_rejected")
                return
        state = scope.setdefault("state", {})
        state["aily_mcp_user_id"] = user_id
        replay = deque(messages)

        async def replay_receive() -> Message:
            if replay:
                return replay.popleft()
            return {"type": "http.disconnect"}

        await self._app(scope, replay_receive, send)


def _single_header(scope: Scope, name: bytes) -> str | None:
    values: list[str] = []
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            values.append(bytes(value).decode("latin-1").strip())
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


def _log_security_rejection(
    *,
    scope: Scope,
    reason_code: str,
    source_ip_allowed: bool,
    user_id: str | None,
) -> None:
    user_header_count = _header_count(scope, _AILY_USER_HEADER)
    email_header_count = _header_count(scope, _AILY_EMAIL_HEADER)
    user_id_sha256 = (
        sha256(user_id.encode("utf-8")).hexdigest()
        if user_id is not None
        else None
    )
    _LOGGER.warning(
        "Aily MCP request rejected reason=%s source_ip_allowed=%s "
        "user_headers=%s email_headers=%s user_sha256=%s",
        reason_code,
        source_ip_allowed,
        user_header_count,
        email_header_count,
        user_id_sha256,
        extra={
            "reason_code": reason_code,
            "source_ip_allowed": source_ip_allowed,
            "user_header_count": user_header_count,
            "email_header_count": email_header_count,
            "user_id_sha256": user_id_sha256,
        },
    )


def _header_count(scope: Scope, name: bytes) -> int:
    return sum(1 for key, _value in scope.get("headers", []) if key.lower() == name)


def _source_ip_is_allowed(
    source_ip: str | None,
    *,
    allowed_source_ips: frozenset[str],
    trust_gateway_source_ip: bool,
) -> bool:
    if source_ip is None:
        return False
    try:
        normalized = str(ip_address(source_ip))
    except ValueError:
        return False
    return trust_gateway_source_ip or normalized in allowed_source_ips


async def _buffer_request(
    receive: Receive,
    max_request_bytes: int,
) -> tuple[list[Message], bool]:
    messages: list[Message] = []
    total = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] == "http.request":
            total += len(message.get("body", b""))
            if total > max_request_bytes:
                return messages, False
            if not message.get("more_body", False):
                return messages, True
        elif message["type"] == "http.disconnect":
            return messages, True


def _mcp_request_method(messages: list[Message]) -> str | None:
    payload = b"".join(
        bytes(message.get("body", b""))
        for message in messages
        if message["type"] == "http.request"
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    method = decoded.get("method")
    return method if isinstance(method, str) else None


async def _send_json_error(send: Send, status: int, code: str) -> None:
    body = json.dumps({"error": code}, separators=(",", ":")).encode("ascii")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _verified_aily_user(ctx: Context[Any, Any, Any]) -> str:
    try:
        request = ctx.request_context.request
        if request is None:
            raise ValueError("missing HTTP request")
        user_id = request.state.aily_mcp_user_id
    except (AttributeError, RuntimeError) as exc:
        raise ValueError("Aily MCP caller is not authorized") from exc
    if not isinstance(user_id, str) or _SAFE_REFERENCE.fullmatch(user_id) is None:
        raise ValueError("Aily MCP caller is not authorized")
    return user_id


def _authorize_caller(
    authorizer: AilyMcpCallerAuthorizer,
    aily_user_id: str,
    run_id: str,
) -> None:
    try:
        authorizer.authorize_run_reference(
            aily_user_id=aily_user_id,
            run_id=_reference(run_id),
        )
    except (LookupError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Aily MCP caller is not authorized") from exc


def _resolve_run(gateway: AilyAnalysisTaskGateway, run_id: str) -> AgentRunState:
    checked_run_id = _reference(run_id)
    try:
        state = gateway.get_analysis_task(checked_run_id)
        checked_state = AgentRunState.model_validate(state.model_dump(mode="json"))
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("分析任务不可用") from exc
    if checked_state.run_id != checked_run_id:
        raise ValueError("analysis task identity mismatch")
    return checked_state


def _resolve_bound_result(
    *,
    state: AgentRunState,
    result_id: str,
    audit_ledger: AilyResultResolver,
) -> ToolResult:
    checked_result_id = _reference(result_id)
    if checked_result_id not in state.result_ids:
        raise ValueError("受审结果不可用")
    try:
        result = audit_ledger.resolve_registered_result(checked_result_id)
        checked_result = ToolResult.model_validate(result.model_dump(mode="json"))
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("受审结果不可用") from exc
    if checked_result.result_id != checked_result_id:
        raise ValueError("audited result identity mismatch")
    return checked_result


def _authorize_result(
    authorizer: AuditedResultAuthorizer,
    result: ToolResult,
) -> Any:
    try:
        authorization = authorizer.authorize(result)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("受审结果展示授权失败") from exc
    if not authorization.allowed:
        raise ValueError("受审结果未获展示授权")
    return authorization


def _audited_result_projection(
    *,
    run_id: str,
    result: ToolResult,
    route_id: str,
    activation_status: str,
    evidence_level: str,
    supported_domain: str,
    max_response_chars: int,
) -> dict[str, object]:
    response: dict[str, object] = {
        "任务ID": run_id,
        "结果ID": result.result_id,
        "工具名称": result.tool_name,
        "工具版本": result.tool_version,
        "模型路线": route_id,
        "模型状态": activation_status,
        "模型版本": result.model_version or "不适用",
        "数据版本": result.data_version,
        "特征版本": result.feature_version,
        "证据类型": evidence_level,
        "适用范围": supported_domain,
        "技术警告": list(result.warnings),
    }
    try:
        scalar_fields = build_audited_analysis_bitable_fields(result)
    except AnalysisBitableProjectionError:
        scalar_fields = {}
    if scalar_fields:
        response["工程师可读结果"] = CHINESE_ANALYSIS_BITABLE_PROFILE.remote_fields(
            scalar_fields
        )
    else:
        serialized_values = json.dumps(
            result.values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized_values) <= max_response_chars // 2:
            response["受审原始结果"] = result.values
        else:
            response["大型结果说明"] = (
                "曲线数组未直接返回, 请读取受审报告或飞书曲线附件。"
            )
    if result.tool_name in {
        "compare_operation_scenarios",
        "project_storage_lifetime",
    }:
        response["参考工况寿命"] = _scenario_lifetime_projection(result)
    return response


def _scenario_lifetime_projection(result: ToolResult) -> dict[str, object]:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("status") != "COMPLETED":
        raise ValueError("参考工况寿命结果不可用")
    if result.tool_name == "compare_operation_scenarios":
        baseline = artifact.get("baseline")
        comparisons = artifact.get("comparisons")
        if not isinstance(baseline, Mapping) or not isinstance(comparisons, list):
            raise ValueError("参考工况寿命结果不可用")
        projections: tuple[object, ...] = (baseline, *comparisons)
    elif result.tool_name == "project_storage_lifetime":
        projections = (artifact.get("projection"),)
    else:
        raise ValueError("参考工况寿命结果不可用")

    rows: list[dict[str, object]] = []
    for projection in projections:
        if not isinstance(projection, Mapping):
            raise ValueError("参考工况寿命结果不可用")
        scenario_id = _reference(projection.get("scenario_id"))
        support = projection.get("support")
        eol = projection.get("eol")
        if not isinstance(support, Mapping) or not isinstance(eol, Mapping):
            raise ValueError("参考工况寿命结果不可用")
        support_status = support.get("status")
        eol_status = eol.get("status")
        if support_status not in {"SUPPORTED", "NEAR_BOUNDARY"} or eol_status not in {
            "REACHED",
            "NOT_REACHED",
        }:
            raise ValueError("参考工况寿命结果不可用")
        threshold = _finite_scenario_number(
            projection.get("eol_threshold"),
            field_name="EOL threshold",
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("参考工况寿命结果不可用")
        row: dict[str, object] = {
            "工况ID": scenario_id,
            "支持状态": support_status,
            "EOL阈值": threshold,
        }
        if eol_status == "REACHED":
            lifetime_years = _finite_scenario_number(
                eol.get("natural_year"),
                field_name="EOL natural year",
            )
            if lifetime_years < 0.0:
                raise ValueError("参考工况寿命结果不可用")
            row.update(
                {
                    "寿命结论": "EOL_REACHED",
                    "从BOL起算参考寿命年": lifetime_years,
                }
            )
        else:
            lower_bound = _finite_scenario_number(
                projection.get("final_natural_year"),
                field_name="scenario horizon",
            )
            if lower_bound < 0.0:
                raise ValueError("参考工况寿命结果不可用")
            row.update(
                {
                    "寿命结论": "LOWER_BOUND_ONLY",
                    "从BOL起算参考寿命下限年": lower_bound,
                }
            )
        rows.append(row)
    if not rows:
        raise ValueError("参考工况寿命结果不可用")
    return {"起算状态": "BOL", "工况": rows}


def _finite_scenario_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} is not finite")
    return number


def _require_audited_report_result(result: ToolResult) -> None:
    if (
        result.tool_name != AUDITED_REPORT_TOOL_NAME
        or result.tool_version != AUDITED_REPORT_TOOL_VERSION
        or result.model_version
        not in {REPORTING_VERSION, RECOMMENDATION_REPORT_RENDERER_VERSION}
        or result.values.get("rendering_version") != result.model_version
    ):
        raise ValueError("结果不是受审报告")


def _bounded_mcp_result(
    response: dict[str, object],
    max_chars: int,
) -> CallToolResult:
    text_fallback = json.dumps(
        response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    result = CallToolResult(
        content=[TextContent(type="text", text=text_fallback)],
        structuredContent=response,
        isError=False,
    )
    rendered = json.dumps(
        result.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    if len(rendered) > max_chars:
        raise ValueError("受审响应超过 Aily 单次上下文上限")
    return result


def _reference(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError("引用格式无效")
    return normalized


__all__ = [
    "AilyMcpAdapter",
    "AilyMcpCallerAuthorizer",
    "AilyMcpConfig",
    "create_aily_mcp_adapter",
]
