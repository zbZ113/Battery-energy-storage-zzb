"""Secret-safe outbound Feishu client with injected transport and bounded retries."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_MEDIA_TYPE = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z"
)
_MEDIA_TYPE_PARAMETER = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*=[a-z0-9][a-z0-9!#$&^_.:+-]*\Z"
)


class FeishuErrorKind(StrEnum):
    AUTHENTICATION = "AUTHENTICATION"
    RATE_LIMIT = "RATE_LIMIT"
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    PROTOCOL = "PROTOCOL"


class FeishuClientError(RuntimeError):
    """Base outbound error that never embeds credentials or response bodies."""


class FeishuApiError(FeishuClientError):
    def __init__(
        self,
        *,
        kind: FeishuErrorKind,
        status_code: int | None,
        api_code: int | None,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        self.api_code = api_code
        super().__init__(
            "Feishu API request failed "
            f"(kind={kind.value}, status={status_code}, code={api_code})"
        )


class FeishuTransportError(FeishuClientError):
    """Transport failure without an HTTP response."""


class FeishuClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_id: str = Field(min_length=1, max_length=200)
    app_secret: SecretStr
    base_url: HttpUrl = HttpUrl("https://open.feishu.cn/open-apis")
    timeout_seconds: float = Field(default=10, gt=0, le=60, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=5)
    backoff_base_seconds: float = Field(default=0.25, ge=0, le=5, allow_inf_nan=False)
    token_refresh_skew_seconds: int = Field(default=60, ge=1, le=600)

    @field_validator("app_id")
    @classmethod
    def app_id_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Feishu app_id must not be blank")
        return normalized

    @field_validator("app_secret")
    @classmethod
    def app_secret_is_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Feishu app_secret must not be blank")
        return value


@dataclass(frozen=True, slots=True)
class FeishuMultipartFile:
    field_name: str
    filename: str
    content_type: str
    payload: bytes = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class FeishuHttpRequest:
    method: str
    url: str
    headers: dict[str, str]
    params: dict[str, str]
    json_body: dict[str, object] | None
    form: dict[str, str]
    files: tuple[FeishuMultipartFile, ...]
    timeout_seconds: float

    def __repr__(self) -> str:
        keys = sorted(self.json_body) if self.json_body is not None else []
        return (
            "FeishuHttpRequest("
            f"method={self.method!r}, url={self.url!r}, "
            f"headers={sorted(self.headers)!r}, params={self.params!r}, "
            f"json_keys={keys!r}, form_keys={sorted(self.form)!r}, "
            f"file_count={len(self.files)}, timeout_seconds={self.timeout_seconds!r})"
        )


@dataclass(frozen=True, slots=True)
class FeishuHttpResponse:
    status_code: int
    headers: Mapping[str, str]
    json_body: dict[str, object] | None = None
    body: bytes = b""


class FeishuTransport(Protocol):
    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse: ...


class FakeFeishuTransport:
    """Deterministic transport for tests and the explicit local sandbox."""

    def __init__(self, *, responses: Sequence[FeishuHttpResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[FeishuHttpRequest] = []

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        self.requests.append(request)
        if not self._responses:
            raise FeishuTransportError("Fake Feishu transport has no queued response")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class HttpxFeishuTransport:
    """Optional real HTTP transport; importing this module does not import httpx."""

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        try:
            import httpx
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise FeishuTransportError(
                "Feishu HTTP transport requires the API dependency group"
            ) from exc
        files = {
            item.field_name: (item.filename, item.payload, item.content_type)
            for item in request.files
        }
        try:
            response = httpx.request(
                request.method,
                request.url,
                headers=request.headers,
                params=request.params,
                json=request.json_body,
                data=request.form or None,
                files=files or None,
                timeout=request.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise FeishuTransportError("Feishu HTTP transport failed") from exc
        content_type = response.headers.get("content-type", "").lower()
        json_body: dict[str, object] | None = None
        if "application/json" in content_type:
            try:
                candidate = response.json()
            except ValueError as exc:
                raise FeishuTransportError("Feishu returned invalid JSON") from exc
            if isinstance(candidate, dict):
                json_body = candidate
        return FeishuHttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            json_body=json_body,
            body=response.content,
        )


@dataclass(frozen=True, slots=True)
class _CachedToken:
    value: str = field(repr=False)
    refresh_at: float


class FeishuClient:
    def __init__(
        self,
        config: FeishuClientConfig,
        *,
        transport: FeishuTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or HttpxFeishuTransport()
        self._clock = clock
        self._sleeper = sleeper
        self._token: _CachedToken | None = None
        self._token_lock = threading.RLock()

    def __repr__(self) -> str:
        return "FeishuClient(config=<redacted>, transport=<configured>)"

    def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request_json(
            "POST",
            "/im/v1/messages",
            params={"receive_id_type": _identifier(receive_id_type, "receive_id_type")},
            json_body={
                "receive_id": _identifier(receive_id, "receive_id"),
                "msg_type": _identifier(msg_type, "msg_type"),
                "content": _canonical_json(content),
            },
        )

    def reply_message(
        self,
        *,
        message_id: str,
        msg_type: str,
        content: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request_json(
            "POST",
            f"/im/v1/messages/{_identifier(message_id, 'message_id')}/reply",
            json_body={
                "msg_type": _identifier(msg_type, "msg_type"),
                "content": _canonical_json(content),
            },
        )

    def update_card(
        self,
        *,
        message_id: str,
        card: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request_json(
            "PATCH",
            f"/im/v1/messages/{_identifier(message_id, 'message_id')}",
            json_body={"content": _canonical_json(card)},
        )

    def upload_image(
        self,
        *,
        payload: bytes,
        content_type: str,
        filename: str = "image.png",
    ) -> dict[str, object]:
        return self._request_json(
            "POST",
            "/im/v1/images",
            form={"image_type": "message"},
            files=(
                FeishuMultipartFile(
                    field_name="image",
                    filename=_filename(filename),
                    content_type=_content_type(content_type),
                    payload=payload,
                ),
            ),
        )

    def upload_file(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
    ) -> dict[str, object]:
        checked_filename = _filename(filename)
        return self._request_json(
            "POST",
            "/im/v1/files",
            form={"file_type": "stream", "file_name": checked_filename},
            files=(
                FeishuMultipartFile(
                    field_name="file",
                    filename=checked_filename,
                    content_type=_content_type(content_type),
                    payload=payload,
                ),
            ),
        )

    def upload_bitable_media(
        self,
        *,
        app_token: str,
        filename: str,
        content_type: str,
        payload: bytes,
    ) -> dict[str, object]:
        checked_filename = _filename(filename)
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 20 * 1024 * 1024:
            raise ValueError("Bitable media payload size is invalid")
        return self._request_json(
            "POST",
            "/drive/v1/medias/upload_all",
            form={
                "file_name": checked_filename,
                "parent_type": "bitable_image",
                "parent_node": _identifier(app_token, "app_token"),
                "size": str(len(payload)),
            },
            files=(
                FeishuMultipartFile(
                    field_name="file",
                    filename=checked_filename,
                    content_type=_content_type(content_type),
                    payload=payload,
                ),
            ),
        )

    def download_message_resource(
        self,
        *,
        message_id: str,
        file_key: str,
        resource_type: str,
    ) -> bytes:
        return self.download_message_resource_response(
            message_id=message_id,
            file_key=file_key,
            resource_type=resource_type,
        ).body

    def download_message_resource_response(
        self,
        *,
        message_id: str,
        file_key: str,
        resource_type: str,
    ) -> FeishuHttpResponse:
        """Return exact bytes plus response metadata for attachment policy checks."""

        return self._request(
            "GET",
            (
                f"/im/v1/messages/{_identifier(message_id, 'message_id')}"
                f"/resources/{_identifier(file_key, 'file_key')}"
            ),
            params={"type": _identifier(resource_type, "resource_type")},
            expect_json=False,
        )

    def search_bitable_records(
        self,
        *,
        app_token: str,
        table_id: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, object]:
        checked_field_name = _bitable_field_name(field_name)
        checked_field_value = _identifier(field_value, "field_value")
        return self._request_json(
            "POST",
            self._bitable_records_path(app_token, table_id) + "/search",
            retry_safe=True,
            json_body={
                "filter": {
                    "conjunction": "and",
                    "conditions": [
                        {
                            "field_name": checked_field_name,
                            "operator": "is",
                            "value": [checked_field_value],
                        }
                    ],
                },
                "field_names": [],
                "sort": [],
                "view_id": None,
                "automatic_fields": False,
            },
        )

    def create_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request_json(
            "POST",
            self._bitable_records_path(app_token, table_id),
            json_body={"fields": dict(fields)},
        )

    def update_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        record_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request_json(
            "PUT",
            (
                self._bitable_records_path(app_token, table_id)
                + f"/{_identifier(record_id, 'record_id')}"
            ),
            json_body={"fields": dict(fields)},
        )

    @staticmethod
    def _bitable_records_path(app_token: str, table_id: str) -> str:
        return (
            f"/bitable/v1/apps/{_identifier(app_token, 'app_token')}"
            f"/tables/{_identifier(table_id, 'table_id')}/records"
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
        form: Mapping[str, str] | None = None,
        files: tuple[FeishuMultipartFile, ...] = (),
        retry_safe: bool | None = None,
    ) -> dict[str, object]:
        response = self._request(
            method,
            path,
            params=params,
            json_body=json_body,
            form=form,
            files=files,
            expect_json=True,
            retry_safe=retry_safe,
        )
        assert response.json_body is not None
        data = response.json_body.get("data", {})
        if not isinstance(data, dict):
            raise FeishuApiError(
                kind=FeishuErrorKind.PROTOCOL,
                status_code=response.status_code,
                api_code=_api_code(response),
            )
        return data

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
        form: Mapping[str, str] | None = None,
        files: tuple[FeishuMultipartFile, ...] = (),
        expect_json: bool,
        retry_safe: bool | None = None,
    ) -> FeishuHttpResponse:
        refreshed = False
        may_retry = method in {"GET", "PATCH", "PUT"} if retry_safe is None else retry_safe
        while True:
            token = self._tenant_token(force_refresh=refreshed)
            response = self._perform_with_retries(
                FeishuHttpRequest(
                    method=method,
                    url=self._url(path),
                    headers={"Authorization": f"Bearer {token}"},
                    params=dict(params or {}),
                    json_body=dict(json_body) if json_body is not None else None,
                    form=dict(form or {}),
                    files=files,
                    timeout_seconds=self._config.timeout_seconds,
                ),
                retry_safe=may_retry,
            )
            if response.status_code == 401 and not refreshed:
                self._invalidate_token()
                refreshed = True
                continue
            self._raise_for_response(response, expect_json=expect_json)
            return response

    def _tenant_token(self, *, force_refresh: bool = False) -> str:
        with self._token_lock:
            if (
                not force_refresh
                and self._token is not None
                and self._clock() < self._token.refresh_at
            ):
                return self._token.value
            response = self._perform_with_retries(
                FeishuHttpRequest(
                    method="POST",
                    url=self._url("/auth/v3/tenant_access_token/internal"),
                    headers={},
                    params={},
                    json_body={
                        "app_id": self._config.app_id,
                        "app_secret": self._config.app_secret.get_secret_value(),
                    },
                    form={},
                    files=(),
                    timeout_seconds=self._config.timeout_seconds,
                ),
                retry_safe=True,
            )
            self._raise_for_response(response, expect_json=True)
            assert response.json_body is not None
            token = response.json_body.get("tenant_access_token")
            expiry = response.json_body.get("expire")
            if (
                not isinstance(token, str)
                or not token.strip()
                or isinstance(expiry, bool)
                or not isinstance(expiry, int | float)
                or not math.isfinite(float(expiry))
                or expiry <= self._config.token_refresh_skew_seconds
            ):
                raise FeishuApiError(
                    kind=FeishuErrorKind.PROTOCOL,
                    status_code=response.status_code,
                    api_code=_api_code(response),
                )
            self._token = _CachedToken(
                value=token.strip(),
                refresh_at=(
                    self._clock()
                    + float(expiry)
                    - self._config.token_refresh_skew_seconds
                ),
            )
            return self._token.value

    def _invalidate_token(self) -> None:
        with self._token_lock:
            self._token = None

    def _perform_with_retries(
        self,
        request: FeishuHttpRequest,
        *,
        retry_safe: bool,
    ) -> FeishuHttpResponse:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = self._transport.request(request)
            except FeishuTransportError:
                if not retry_safe or attempt == self._config.max_attempts:
                    raise FeishuApiError(
                        kind=FeishuErrorKind.TRANSIENT,
                        status_code=None,
                        api_code=None,
                    ) from None
                self._backoff(attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                kind = (
                    FeishuErrorKind.RATE_LIMIT
                    if response.status_code == 429
                    else FeishuErrorKind.TRANSIENT
                )
                if not retry_safe or attempt == self._config.max_attempts:
                    raise FeishuApiError(
                        kind=kind,
                        status_code=response.status_code,
                        api_code=_api_code(response),
                    )
                self._backoff(attempt)
                continue
            return response
        raise AssertionError("retry loop must return or raise")

    def _backoff(self, attempt: int) -> None:
        self._sleeper(self._config.backoff_base_seconds * (2 ** (attempt - 1)))

    @staticmethod
    def _raise_for_response(
        response: FeishuHttpResponse,
        *,
        expect_json: bool,
    ) -> None:
        if response.status_code == 401:
            raise FeishuApiError(
                kind=FeishuErrorKind.AUTHENTICATION,
                status_code=401,
                api_code=_api_code(response),
            )
        if not 200 <= response.status_code < 300:
            raise FeishuApiError(
                kind=FeishuErrorKind.PERMANENT,
                status_code=response.status_code,
                api_code=_api_code(response),
            )
        if expect_json:
            code = _api_code(response)
            if response.json_body is None or code is None:
                raise FeishuApiError(
                    kind=FeishuErrorKind.PROTOCOL,
                    status_code=response.status_code,
                    api_code=code,
                )
            if code != 0:
                raise FeishuApiError(
                    kind=FeishuErrorKind.PERMANENT,
                    status_code=response.status_code,
                    api_code=code,
                )

    def _url(self, path: str) -> str:
        return str(self._config.base_url).rstrip("/") + "/" + path.lstrip("/")


def _api_code(response: FeishuHttpResponse) -> int | None:
    if response.json_body is None:
        return None
    code = response.json_body.get("code")
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _canonical_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Feishu content must be JSON-compatible") from exc


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe machine identifier")
    return normalized


def _bitable_field_name(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 100
        or normalized != value
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("field_name must be a safe Bitable field name")
    return normalized


def _filename(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 255
        or normalized != value
        or "/" in normalized
        or "\\" in normalized
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("filename must be a safe basename")
    return normalized


def _content_type(value: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    parts = [part.strip() for part in normalized.split(";")]
    if (
        not normalized
        or len(normalized) > 200
        or len(parts) > 9
        or _MEDIA_TYPE.fullmatch(parts[0]) is None
        or any(_MEDIA_TYPE_PARAMETER.fullmatch(part) is None for part in parts[1:])
    ):
        raise ValueError("content_type must be a bounded media type")
    return "; ".join(parts)


__all__ = [
    "FakeFeishuTransport",
    "FeishuApiError",
    "FeishuClient",
    "FeishuClientConfig",
    "FeishuClientError",
    "FeishuErrorKind",
    "FeishuHttpRequest",
    "FeishuHttpResponse",
    "FeishuMultipartFile",
    "FeishuTransport",
    "FeishuTransportError",
    "HttpxFeishuTransport",
]
