"""Dependency-light client for the public FastAPI workbench endpoints.

The client intentionally knows only transport contracts.  It never imports a
domain model, loads a model artifact, or calculates an engineering value.
"""

from __future__ import annotations

import os
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

AgentRunStatusValue: TypeAlias = Literal[
    "PLANNING",
    "RUNNING",
    "AWAITING_APPROVAL",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "FALLBACK",
]
AgentPlanningModeValue: TypeAlias = Literal["LLM", "FIXED_FALLBACK"]
AgentDispatchStatusValue: TypeAlias = Literal["PENDING", "DISPATCHED"]
_AGENT_RUN_STATUSES = frozenset(
    {
        "PLANNING",
        "RUNNING",
        "AWAITING_APPROVAL",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "FALLBACK",
    }
)
_AGENT_PLANNING_MODES = frozenset({"LLM", "FIXED_FALLBACK"})
_AGENT_DISPATCH_STATUSES = frozenset({"PENDING", "DISPATCHED"})


class WorkbenchError(RuntimeError):
    """Base class for failures that can be shown safely by the workbench."""


class ApiConnectionError(WorkbenchError):
    """Raised when the FastAPI service cannot be reached."""


class ApiResponseError(WorkbenchError):
    """Raised when a successful HTTP response violates the public contract."""


class ApiHttpError(WorkbenchError):
    """Raised when the FastAPI service returns a non-success status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API request failed ({status_code}): {detail}")


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Transport-neutral decoded HTTP response."""

    status_code: int
    body: object


class HttpTransport(Protocol):
    """Small injectable boundary used by unit tests and the httpx adapter."""

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse: ...


class HttpxTransport:
    """Persistent httpx transport that retains the opaque session cookie."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        try:
            response = self._client.request(
                method,
                url,
                json=dict(json_body) if json_body is not None else None,
                headers={"Accept": "application/json", **dict(headers or {})},
            )
        except httpx.HTTPError as exc:
            raise ApiConnectionError("FastAPI service is unreachable") from exc
        if not response.content:
            body: object = None
        else:
            try:
                body = cast(object, response.json())
            except ValueError:
                body = response.text
        return HttpResponse(response.status_code, body)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


@dataclass(frozen=True, slots=True)
class WorkbenchConfig:
    """Fail-closed runtime configuration for the Streamlit HTTP client."""

    environment: Literal["development", "production"]
    api_base_url: str
    trusted_origin: str

    @classmethod
    def from_environment(cls) -> WorkbenchConfig:
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> WorkbenchConfig:
        environment_value = _required_setting(values, "QUANXIN_ENVIRONMENT")
        if environment_value not in {"development", "production"}:
            raise ValueError("QUANXIN_ENVIRONMENT must be development or production")
        environment = cast(Literal["development", "production"], environment_value)
        api_base_url = _normalize_base_url(
            _required_setting(values, "QUANXIN_API_BASE_URL")
        )
        trusted_origin = _normalize_origin(
            _required_setting(values, "QUANXIN_TRUSTED_ORIGIN")
        )
        if environment == "production":
            if urlsplit(api_base_url).scheme != "https":
                raise ValueError("production API base URL must use HTTPS")
            if urlsplit(trusted_origin).scheme != "https":
                raise ValueError("production trusted origin must use HTTPS")
        else:
            for description, value in (
                ("development API base URL", api_base_url),
                ("development trusted origin", trusted_origin),
            ):
                parsed = urlsplit(value)
                if parsed.scheme != "http" or not _is_loopback_hostname(parsed.hostname):
                    raise ValueError(f"{description} must use HTTP on localhost")
        return cls(
            environment=environment,
            api_base_url=api_base_url,
            trusted_origin=trusted_origin,
        )


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    user_id: str
    username: str
    role: str
    must_change_password: bool


@dataclass(frozen=True, slots=True)
class AuthSession(AuthPrincipal):
    expires_at: str


@dataclass(frozen=True, slots=True)
class HealthStatus:
    status: str
    service: str

    def display_payload(self) -> dict[str, str]:
        return {"status": self.status, "service": self.service}


@dataclass(frozen=True, slots=True)
class AgentRunSnapshot:
    """Transport-only view of one server-owned Agent run.

    Nested intent and plan objects are preserved exactly as issued by FastAPI.
    The workbench does not derive step state, results, or engineering values.
    """

    run_id: str
    project_id: str
    created_by_user_id: str
    status: AgentRunStatusValue
    planning_mode: AgentPlanningModeValue
    intent: dict[str, object]
    plan: dict[str, object]
    dispatch_status: AgentDispatchStatusValue
    dispatch_task_id: str | None
    created_at: str
    updated_at: str
    completed_at: str | None

    def display_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "status": self.status,
            "planning_mode": self.planning_mode,
            "dispatch_status": self.dispatch_status,
            "dispatch_task_id": self.dispatch_task_id,
            "intent": dict(self.intent),
            "plan": dict(self.plan),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True, slots=True)
class LifetimeWorkflowResult:
    """Identifier-only terminal result returned by the workflow endpoint."""

    status: str
    quality_result_id: str
    feature_result_id: str | None
    prediction_result_id: str | None
    calibration_result_id: str | None
    interval_result_id: str | None
    decision_result_id: str | None
    report_result_id: str | None
    warnings: tuple[str, ...]

    def display_payload(self) -> dict[str, object]:
        """Return identifiers and warnings only; never synthesize numeric results."""
        return {
            "status": self.status,
            "quality_result_id": self.quality_result_id,
            "feature_result_id": self.feature_result_id,
            "prediction_result_id": self.prediction_result_id,
            "calibration_result_id": self.calibration_result_id,
            "interval_result_id": self.interval_result_id,
            "decision_result_id": self.decision_result_id,
            "report_result_id": self.report_result_id,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class AuditedMarkdown:
    result_id: str
    markdown: str

    def download_bytes(self) -> bytes:
        """Encode the unchanged server-issued report for Streamlit download."""
        return self.markdown.encode("utf-8")


class ApiClient:
    """HTTP-only facade over the thin FastAPI application API."""

    def __init__(
        self,
        base_url: str,
        *,
        trusted_origin: str,
        transport: HttpTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if transport is not None and http_client is not None:
            raise ValueError("transport and http_client are mutually exclusive")
        self._base_url = _normalize_base_url(base_url)
        self._trusted_origin = _normalize_origin(trusted_origin)
        self._transport = transport or HttpxTransport(client=http_client)

    @classmethod
    def from_config(cls, config: WorkbenchConfig) -> ApiClient:
        return cls(
            config.api_base_url,
            trusted_origin=config.trusted_origin,
        )

    def login(self, *, username: str, password: str) -> AuthSession:
        body = self._request(
            "POST",
            "/v1/auth/login",
            {
                "username": _normalize_identifier(username, "username"),
                "password": _require_nonblank_secret(password, "password"),
            },
        )
        return _parse_auth_session(body, "login response")

    def me(self) -> AuthPrincipal:
        return _parse_auth_principal(self._request("GET", "/v1/auth/me"), "me response")

    def change_password(
        self,
        *,
        current_password: str,
        new_password: str,
    ) -> AuthSession:
        body = self._request(
            "POST",
            "/v1/auth/change-password",
            {
                "current_password": _require_nonblank_secret(
                    current_password, "current_password"
                ),
                "new_password": _require_nonblank_secret(new_password, "new_password"),
            },
        )
        return _parse_auth_session(body, "change-password response")

    def logout(self) -> None:
        self._request("POST", "/v1/auth/logout")

    def close(self) -> None:
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def health(self) -> HealthStatus:
        body = self._request("GET", "/health")
        payload = _require_mapping(body, "health response")
        return HealthStatus(
            status=_require_nonblank_string(payload, "status", "health response"),
            service=_require_nonblank_string(payload, "service", "health response"),
        )

    def list_tools(self) -> tuple[dict[str, object], ...]:
        body = self._request("GET", "/v1/tools")
        if not isinstance(body, list):
            raise ApiResponseError("tool discovery response must be a JSON list")
        tools: list[dict[str, object]] = []
        for index, item in enumerate(body):
            mapping = _require_mapping(item, f"tool discovery item {index}")
            tools.append(dict(mapping))
        return tuple(tools)

    def create_agent_run(
        self,
        *,
        project_id: str,
        user_goal: str,
        dataset_ids: tuple[str, ...],
        requested_outputs: tuple[str, ...],
        idempotency_key: str,
    ) -> AgentRunSnapshot:
        normalized_outputs = _normalize_identifier_sequence(
            requested_outputs,
            "requested_outputs",
            require_nonempty=True,
        )
        payload: dict[str, object] = {
            "project_id": _normalize_identifier(project_id, "project_id"),
            "user_goal": _normalize_identifier(user_goal, "user_goal"),
            "dataset_ids": list(
                _normalize_identifier_sequence(dataset_ids, "dataset_ids")
            ),
            "requested_outputs": list(normalized_outputs),
        }
        key = _normalize_idempotency_key(idempotency_key)
        body = self._request(
            "POST",
            "/v1/agent/runs",
            payload,
            headers={"Idempotency-Key": key},
        )
        return _parse_agent_run(body, "Agent run creation response")

    def get_agent_run(self, run_id: str) -> AgentRunSnapshot:
        normalized_run_id = _normalize_identifier(run_id, "run_id")
        body = self._request(
            "GET", f"/v1/agent/runs/{quote(normalized_run_id, safe='')}"
        )
        return _parse_matching_agent_run(body, normalized_run_id, "Agent run response")

    def cancel_agent_run(self, run_id: str) -> AgentRunSnapshot:
        normalized_run_id = _normalize_identifier(run_id, "run_id")
        body = self._request(
            "POST",
            f"/v1/agent/runs/{quote(normalized_run_id, safe='')}/cancel",
        )
        return _parse_matching_agent_run(
            body, normalized_run_id, "Agent run cancellation response"
        )

    def approve_agent_run(
        self,
        run_id: str,
        *,
        approval_id: str,
        reason: str | None,
    ) -> AgentRunSnapshot:
        return self._agent_approval_action(
            run_id,
            action="approve",
            approval_id=approval_id,
            reason=reason,
        )

    def reject_agent_run(
        self,
        run_id: str,
        *,
        approval_id: str,
        reason: str | None,
    ) -> AgentRunSnapshot:
        return self._agent_approval_action(
            run_id,
            action="reject",
            approval_id=approval_id,
            reason=reason,
        )

    def _agent_approval_action(
        self,
        run_id: str,
        *,
        action: Literal["approve", "reject"],
        approval_id: str,
        reason: str | None,
    ) -> AgentRunSnapshot:
        normalized_run_id = _normalize_identifier(run_id, "run_id")
        normalized_reason = reason.strip() if isinstance(reason, str) else None
        if normalized_reason == "":
            normalized_reason = None
        body = self._request(
            "POST",
            f"/v1/agent/runs/{quote(normalized_run_id, safe='')}/{action}",
            {
                "approval_id": _normalize_identifier(approval_id, "approval_id"),
                "reason": normalized_reason,
            },
        )
        return _parse_matching_agent_run(
            body, normalized_run_id, f"Agent run {action} response"
        )

    def get_agent_run_result(self, run_id: str, result_id: str) -> dict[str, object]:
        """Read a ToolResult only through its authorized Agent run scope."""

        normalized_result_id = _normalize_identifier(result_id, "result_id")
        body = self._request(
            "GET",
            (
                f"/v1/agent/runs/{_encoded_id(run_id, 'run_id')}"
                f"/results/{quote(normalized_result_id, safe='')}"
            ),
        )
        response = _require_mapping(body, "Agent run result response")
        returned_id = _require_nonblank_string(
            response, "result_id", "Agent run result response"
        )
        if returned_id != normalized_result_id:
            raise ApiResponseError("Agent run result_id does not match the request")
        return dict(response)

    def run_lifetime_workflow(
        self,
        *,
        record_batch_id: str,
        calibration_cohort_id: str,
        policy_id: str,
    ) -> LifetimeWorkflowResult:
        payload = {
            "record_batch_id": _normalize_identifier(record_batch_id, "record_batch_id"),
            "calibration_cohort_id": _normalize_identifier(
                calibration_cohort_id, "calibration_cohort_id"
            ),
            "policy_id": _normalize_identifier(policy_id, "policy_id"),
        }
        body = self._request("POST", "/v1/workflows/lifetime-decision", payload)
        response = _require_mapping(body, "lifetime workflow response")
        warnings_value = response.get("warnings", [])
        if not isinstance(warnings_value, list) or not all(
            isinstance(item, str) for item in warnings_value
        ):
            raise ApiResponseError("lifetime workflow warnings must be a JSON string list")
        return LifetimeWorkflowResult(
            status=_require_nonblank_string(response, "status", "lifetime workflow response"),
            quality_result_id=_require_nonblank_string(
                response, "quality_result_id", "lifetime workflow response"
            ),
            feature_result_id=_optional_string(response, "feature_result_id"),
            prediction_result_id=_optional_string(response, "prediction_result_id"),
            calibration_result_id=_optional_string(response, "calibration_result_id"),
            interval_result_id=_optional_string(response, "interval_result_id"),
            decision_result_id=_optional_string(response, "decision_result_id"),
            report_result_id=_optional_string(response, "report_result_id"),
            warnings=tuple(cast(list[str], warnings_value)),
        )

    def register_canonical_csv(
        self,
        payload: bytes,
        registration: Mapping[str, object],
    ) -> str:
        """Upload bytes plus the public registration contract without deriving values."""

        if not payload:
            raise ValueError("canonical CSV payload must not be empty")
        if not registration:
            raise ValueError("canonical CSV registration must not be empty")
        body = self._request(
            "POST",
            "/v1/batches/canonical-csv",
            {
                "payload_base64": b64encode(payload).decode("ascii"),
                "registration": dict(registration),
            },
        )
        response = _require_mapping(body, "canonical CSV registration response")
        return _require_nonblank_string(
            response,
            "record_batch_id",
            "canonical CSV registration response",
        )

    def get_audited_markdown(self, result_id: str) -> AuditedMarkdown:
        normalized_id = _normalize_identifier(result_id, "result_id")
        encoded_id = quote(normalized_id, safe="")
        body = self._request("GET", f"/v1/reports/{encoded_id}")
        response = _require_mapping(body, "audited report response")
        returned_id = _require_nonblank_string(
            response, "result_id", "audited report response"
        )
        if returned_id != normalized_id:
            raise ApiResponseError("audited report result_id does not match the request")
        markdown = _require_nonblank_string(response, "markdown", "audited report response")
        return AuditedMarkdown(result_id=returned_id, markdown=markdown)

    def get_tool_result(self, result_id: str) -> dict[str, object]:
        """Fetch one server-signed ToolResult without interpreting its values."""
        normalized_id = _normalize_identifier(result_id, "result_id")
        encoded_id = quote(normalized_id, safe="")
        body = self._request("GET", f"/v1/results/{encoded_id}")
        response = _require_mapping(body, "tool result response")
        returned_id = _require_nonblank_string(response, "result_id", "tool result response")
        if returned_id != normalized_id:
            raise ApiResponseError("tool result result_id does not match the request")
        return dict(response)

    def _request(
        self,
        method: str,
        path: str,
        json_body: Mapping[str, object] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> object:
        request_headers = dict(headers or {})
        if method.upper() == "POST":
            request_headers = {**request_headers, "Origin": self._trusted_origin}
        response = self._transport.request(
            method,
            f"{self._base_url}{path}",
            json_body,
            request_headers or None,
        )
        if not 200 <= response.status_code < 300:
            raise ApiHttpError(response.status_code, _error_detail(response.body))
        return response.body


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query string or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _normalize_origin(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("trusted origin must be an HTTP(S) scheme and authority only")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _required_setting(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _is_loopback_hostname(hostname: str | None) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _normalize_identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _encoded_id(value: str, field_name: str) -> str:
    return quote(_normalize_identifier(value, field_name), safe="")


def _normalize_identifier_sequence(
    values: tuple[str, ...],
    field_name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_normalize_identifier(value, field_name) for value in values)
    if require_nonempty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique identifiers")
    return normalized


def _normalize_idempotency_key(value: str) -> str:
    normalized = value.strip()
    if not 8 <= len(normalized) <= 200:
        raise ValueError("idempotency_key must contain between 8 and 200 characters")
    if "\r" in normalized or "\n" in normalized:
        raise ValueError("idempotency_key must not contain line breaks")
    return normalized


def _require_nonblank_secret(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _require_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ApiResponseError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _require_nonblank_string(
    value: Mapping[str, object], field_name: str, description: str
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise ApiResponseError(f"{description} requires non-blank {field_name}")
    return item


def _optional_string(value: Mapping[str, object], field_name: str) -> str | None:
    item = value.get(field_name)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ApiResponseError(f"lifetime workflow {field_name} must be null or non-blank")
    return item


def _require_bool(value: Mapping[str, object], field_name: str, description: str) -> bool:
    item = value.get(field_name)
    if not isinstance(item, bool):
        raise ApiResponseError(f"{description} requires boolean {field_name}")
    return item


def _parse_auth_principal(value: object, description: str) -> AuthPrincipal:
    payload = _require_mapping(value, description)
    return AuthPrincipal(
        user_id=_require_nonblank_string(payload, "user_id", description),
        username=_require_nonblank_string(payload, "username", description),
        role=_require_nonblank_string(payload, "role", description),
        must_change_password=_require_bool(payload, "must_change_password", description),
    )


def _parse_auth_session(value: object, description: str) -> AuthSession:
    payload = _require_mapping(value, description)
    principal = _parse_auth_principal(payload, description)
    return AuthSession(
        user_id=principal.user_id,
        username=principal.username,
        role=principal.role,
        must_change_password=principal.must_change_password,
        expires_at=_require_nonblank_string(payload, "expires_at", description),
    )


def _parse_agent_run(value: object, description: str) -> AgentRunSnapshot:
    payload = _require_mapping(value, description)
    intent = _require_mapping(payload.get("intent"), f"{description} intent")
    plan = _require_mapping(payload.get("plan"), f"{description} plan")
    return AgentRunSnapshot(
        run_id=_require_nonblank_string(payload, "run_id", description),
        project_id=_require_nonblank_string(payload, "project_id", description),
        created_by_user_id=_require_nonblank_string(
            payload, "created_by_user_id", description
        ),
        status=cast(
            AgentRunStatusValue,
            _require_enum_string(
                payload, "status", description, allowed=_AGENT_RUN_STATUSES
            ),
        ),
        planning_mode=cast(
            AgentPlanningModeValue,
            _require_enum_string(
                payload,
                "planning_mode",
                description,
                allowed=_AGENT_PLANNING_MODES,
            ),
        ),
        intent=dict(intent),
        plan=dict(plan),
        dispatch_status=cast(
            AgentDispatchStatusValue,
            _require_enum_string(
                payload,
                "dispatch_status",
                description,
                allowed=_AGENT_DISPATCH_STATUSES,
            ),
        ),
        dispatch_task_id=_optional_transport_string(
            payload, "dispatch_task_id", description
        ),
        created_at=_require_nonblank_string(payload, "created_at", description),
        updated_at=_require_nonblank_string(payload, "updated_at", description),
        completed_at=_optional_transport_string(payload, "completed_at", description),
    )


def _parse_matching_agent_run(
    value: object,
    expected_run_id: str,
    description: str,
) -> AgentRunSnapshot:
    run = _parse_agent_run(value, description)
    if run.run_id != expected_run_id:
        raise ApiResponseError(f"{description} run_id does not match the request")
    return run


def _require_enum_string(
    value: Mapping[str, object],
    field_name: str,
    description: str,
    *,
    allowed: frozenset[str],
) -> str:
    item = _require_nonblank_string(value, field_name, description)
    if item not in allowed:
        raise ApiResponseError(f"{description} has unsupported {field_name}")
    return item


def _optional_transport_string(
    value: Mapping[str, object], field_name: str, description: str
) -> str | None:
    item = value.get(field_name)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ApiResponseError(f"{description} {field_name} must be null or non-blank")
    return item


def _error_detail(value: object) -> str:
    if isinstance(value, dict):
        detail = value.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
    if isinstance(value, str) and value.strip():
        return value
    return "the API returned an unspecified error"
