"""Local-only in-memory Feishu OpenAPI sandbox with no model execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


@dataclass(slots=True)
class _SandboxState:
    messages: dict[str, dict[str, str]] = field(default_factory=dict)
    resources: dict[tuple[str, str], tuple[bytes, str]] = field(default_factory=dict)
    bitable_records: dict[tuple[str, str, str], dict[str, object]] = field(
        default_factory=dict
    )
    bitable_media: set[str] = field(default_factory=set)
    lock: RLock = field(default_factory=RLock)


def create_fake_feishu_sandbox_app() -> FastAPI:
    """Create an isolated fake Feishu server for local protocol tests."""

    app = FastAPI(title="Fake Feishu Sandbox", version="v1")
    state = _SandboxState()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "fake-feishu-sandbox"}

    @app.get("/sandbox/state")
    def snapshot() -> dict[str, int]:
        with state.lock:
            return {
                "messages": len(state.messages),
                "resources": len(state.resources),
                "bitable_records": len(state.bitable_records),
            }

    @app.put(
        "/sandbox/resources/{message_id}/{file_key}",
        status_code=204,
    )
    async def seed_resource(
        message_id: str,
        file_key: str,
        request: Request,
    ) -> Response:
        checked_message_id = _reference(message_id, "message_id")
        checked_file_key = _reference(file_key, "file_key")
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=422, detail="sandbox_resource_empty")
        content_type = request.headers.get("content-type", "application/octet-stream")
        with state.lock:
            state.resources[(checked_message_id, checked_file_key)] = (
                payload,
                content_type,
            )
        return Response(status_code=204)

    @app.post("/open-apis/auth/v3/tenant_access_token/internal")
    def tenant_token() -> dict[str, object]:
        return {
            "code": 0,
            "tenant_access_token": "sandbox-token",
            "expire": 7200,
        }

    @app.post("/open-apis/im/v1/messages")
    async def send_message(request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await _json_object(request)
        message_id = _next_reference("om", len(state.messages) + 1)
        with state.lock:
            state.messages[message_id] = _message_reference(payload)
        return _ok({"message_id": message_id})

    @app.post("/open-apis/im/v1/messages/{message_id}/reply")
    async def reply_message(message_id: str, request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await _json_object(request)
        reply_id = _next_reference("om", len(state.messages) + 1)
        reference = _message_reference(payload)
        reference["reply_to"] = _reference(message_id, "message_id")
        with state.lock:
            state.messages[reply_id] = reference
        return _ok({"message_id": reply_id})

    @app.patch("/open-apis/im/v1/messages/{message_id}")
    async def update_message(message_id: str, request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        checked_message_id = _reference(message_id, "message_id")
        await _json_object(request)
        with state.lock:
            if checked_message_id not in state.messages:
                state.messages[checked_message_id] = {
                    "receive_id": "sandbox-existing-message",
                    "msg_type": "interactive",
                }
        return _ok({"message_id": checked_message_id})

    @app.post("/open-apis/im/v1/files")
    async def upload_file(request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=422, detail="sandbox_upload_empty")
        return _ok({"file_key": _next_reference("file", len(state.resources) + 1)})

    @app.post("/open-apis/im/v1/images")
    async def upload_image(request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=422, detail="sandbox_upload_empty")
        return _ok({"image_key": _next_reference("image", len(state.resources) + 1)})

    @app.post("/open-apis/drive/v1/medias/upload_all")
    async def upload_bitable_media(request: Request) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=422, detail="sandbox_upload_empty")
        with state.lock:
            file_token = _next_reference("media", len(state.bitable_media) + 1)
            state.bitable_media.add(file_token)
        return _ok({"file_token": file_token})

    @app.get(
        "/open-apis/im/v1/messages/{message_id}/resources/{file_key}"
    )
    def download_resource(
        message_id: str,
        file_key: str,
        request: Request,
    ) -> Response:
        _require_sandbox_bearer(request)
        key = (
            _reference(message_id, "message_id"),
            _reference(file_key, "file_key"),
        )
        with state.lock:
            resource = state.resources.get(key)
        if resource is None:
            raise HTTPException(status_code=404, detail="sandbox_resource_not_found")
        return Response(content=resource[0], media_type=resource[1])

    @app.post(
        "/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/search"
    )
    async def search_records(
        app_token: str,
        table_id: str,
        request: Request,
    ) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await _json_object(request)
        run_field_name, run_id = _run_id_filter(payload.get("filter"))
        prefix = (
            _reference(app_token, "app_token"),
            _reference(table_id, "table_id"),
        )
        with state.lock:
            items = [
                dict(record)
                for key, record in state.bitable_records.items()
                if key[:2] == prefix
                and _record_matches_run(record, run_field_name, run_id)
            ]
        return _ok({"items": items, "has_more": False})

    @app.post(
        "/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
    )
    async def create_record(
        app_token: str,
        table_id: str,
        request: Request,
    ) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await _json_object(request)
        fields = _scalar_fields(payload.get("fields"))
        with state.lock:
            record_id = _next_reference("rec", len(state.bitable_records) + 1)
            key = (
                _reference(app_token, "app_token"),
                _reference(table_id, "table_id"),
                record_id,
            )
            record: dict[str, object] = {
                "record_id": record_id,
                "fields": fields,
            }
            state.bitable_records[key] = record
        return _ok({"record": record})

    @app.put(
        "/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"
    )
    async def update_record(
        app_token: str,
        table_id: str,
        record_id: str,
        request: Request,
    ) -> dict[str, object]:
        _require_sandbox_bearer(request)
        payload = await _json_object(request)
        fields = _scalar_fields(payload.get("fields"))
        key = (
            _reference(app_token, "app_token"),
            _reference(table_id, "table_id"),
            _reference(record_id, "record_id"),
        )
        with state.lock:
            if key not in state.bitable_records:
                raise HTTPException(status_code=404, detail="sandbox_record_not_found")
            record: dict[str, object] = {
                "record_id": key[2],
                "fields": fields,
            }
            state.bitable_records[key] = record
        return _ok({"record": record})

    return app


def _require_sandbox_bearer(request: Request) -> None:
    if request.headers.get("authorization") != "Bearer sandbox-token":
        raise HTTPException(
            status_code=401,
            detail="fake_feishu_authentication_failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="sandbox_json_invalid") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="sandbox_json_invalid")
    return value


def _message_reference(payload: dict[str, Any]) -> dict[str, str]:
    receive_id = _reference(payload.get("receive_id", "sandbox-reply"), "receive_id")
    msg_type = _reference(payload.get("msg_type"), "msg_type")
    return {"receive_id": receive_id, "msg_type": msg_type}


def _scalar_fields(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="sandbox_bitable_fields_invalid")
    normalized: dict[str, object] = {}
    for field_name, item in value.items():
        checked_name = _field_name(field_name)
        if isinstance(item, dict):
            raise HTTPException(
                status_code=422,
                detail="sandbox_bitable_fields_invalid",
            )
        if isinstance(item, list):
            if not _is_attachment_value(item):
                raise HTTPException(
                    status_code=422,
                    detail="sandbox_bitable_fields_invalid",
                )
            normalized[checked_name] = [dict(entry) for entry in item]
        else:
            normalized[checked_name] = item
    return normalized


def _record_matches_run(
    record: dict[str, object],
    field_name: str,
    run_id: str,
) -> bool:
    fields = record.get("fields")
    return isinstance(fields, dict) and fields.get(field_name) == run_id


def _run_id_filter(value: object) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="sandbox_filter_invalid")
    conditions = value.get("conditions")
    if (
        value.get("conjunction") != "and"
        or not isinstance(conditions, list)
        or len(conditions) != 1
        or not isinstance(conditions[0], dict)
    ):
        raise HTTPException(status_code=422, detail="sandbox_filter_invalid")
    condition = conditions[0]
    items = condition.get("value")
    if (
        condition.get("operator") != "is"
        or not isinstance(items, list)
        or len(items) != 1
    ):
        raise HTTPException(status_code=422, detail="sandbox_filter_invalid")
    return (
        _field_name(condition.get("field_name")),
        _reference(items[0], "run_id"),
    )


def _is_attachment_value(value: list[object]) -> bool:
    if not value:
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) != {"file_token"}:
            return False
        try:
            _reference(item.get("file_token"), "file_token")
        except HTTPException:
            return False
    return True


def _field_name(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 100
        or any(ord(character) < 32 for character in normalized)
    ):
        raise HTTPException(status_code=422, detail="sandbox_field_name_invalid")
    return normalized


def _reference(value: object, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise HTTPException(status_code=422, detail=f"sandbox_{field_name}_invalid")
    return normalized


def _next_reference(prefix: str, ordinal: int) -> str:
    return f"{prefix}-{ordinal:06d}"


def _ok(data: dict[str, object]) -> dict[str, object]:
    return {"code": 0, "data": data}


__all__ = ["create_fake_feishu_sandbox_app"]
