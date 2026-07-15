"""Dependency-light client for the public FastAPI workbench endpoints.

The client intentionally knows only transport contracts.  It never imports a
domain model, loads a model artifact, or calculates an engineering value.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


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
    """Small injectable boundary used by unit tests and the urllib adapter."""

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, object] | None = None,
    ) -> HttpResponse: ...


class UrlLibTransport:
    """Standard-library JSON transport; no domain or Streamlit dependency."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, object] | None = None,
    ) -> HttpResponse:
        data = None
        headers = {"Accept": "application/json"}
        if json_body is not None:
            data = json.dumps(dict(json_body), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read()
                status_code = response.status
        except HTTPError as exc:
            payload = exc.read()
            return HttpResponse(exc.code, _decode_json_or_text(payload))
        except URLError as exc:
            raise ApiConnectionError(f"FastAPI service is unreachable: {exc.reason}") from exc
        return HttpResponse(status_code, _decode_json(payload))


@dataclass(frozen=True, slots=True)
class HealthStatus:
    status: str
    service: str

    def display_payload(self) -> dict[str, str]:
        return {"status": self.status, "service": self.service}


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
        transport: HttpTransport | None = None,
    ) -> None:
        self._base_url = _normalize_base_url(base_url)
        self._transport = transport or UrlLibTransport()

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

    def _request(
        self,
        method: str,
        path: str,
        json_body: Mapping[str, object] | None = None,
    ) -> object:
        response = self._transport.request(
            method,
            f"{self._base_url}{path}",
            json_body,
        )
        if not 200 <= response.status_code < 300:
            raise ApiHttpError(response.status_code, _error_detail(response.body))
        return response.body


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query string or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _normalize_identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


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


def _error_detail(value: object) -> str:
    if isinstance(value, dict):
        detail = value.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
    if isinstance(value, str) and value.strip():
        return value
    return "the API returned an unspecified error"


def _decode_json(payload: bytes) -> object:
    try:
        return cast(object, json.loads(payload.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiResponseError("the API returned invalid UTF-8 JSON") from exc


def _decode_json_or_text(payload: bytes) -> object:
    try:
        return _decode_json(payload)
    except ApiResponseError:
        return payload.decode("utf-8", errors="replace")
