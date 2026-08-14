from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from quanxin_life.integrations.feishu.client import (
    FakeFeishuTransport,
    FeishuApiError,
    FeishuClient,
    FeishuClientConfig,
    FeishuErrorKind,
    FeishuHttpResponse,
    FeishuTransportError,
)


def _client(
    transport: FakeFeishuTransport,
    *,
    sleeps: list[float] | None = None,
) -> FeishuClient:
    sleep_log = sleeps if sleeps is not None else []
    return FeishuClient(
        FeishuClientConfig(
            app_id="cli_test_app",
            app_secret=SecretStr("must-not-leak"),
            base_url="https://open.feishu.example/open-apis",
            timeout_seconds=3,
            max_attempts=3,
            backoff_base_seconds=0.1,
        ),
        transport=transport,
        clock=lambda: 1000.0,
        sleeper=sleep_log.append,
    )


def _token_response(token: str = "tenant-token") -> FeishuHttpResponse:
    return FeishuHttpResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        json_body={"code": 0, "tenant_access_token": token, "expire": 7200},
    )


def _ok(data: dict[str, object] | None = None) -> FeishuHttpResponse:
    return FeishuHttpResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        json_body={"code": 0, "data": data or {}},
    )


def test_tenant_token_is_cached_without_exposing_the_app_secret() -> None:
    transport = FakeFeishuTransport(
        responses=[_token_response(), _ok(), _ok()]
    )
    client = _client(transport)

    client.send_message(
        receive_id="oc_reviewed_chat",
        receive_id_type="chat_id",
        msg_type="text",
        content={"text": "status-only"},
    )
    client.send_message(
        receive_id="oc_reviewed_chat",
        receive_id_type="chat_id",
        msg_type="text",
        content={"text": "status-only"},
    )

    assert len(transport.requests) == 3
    assert transport.requests[0].url.endswith(
        "/auth/v3/tenant_access_token/internal"
    )
    assert transport.requests[1].headers["Authorization"] == "Bearer tenant-token"
    assert "must-not-leak" not in repr(client)
    assert "must-not-leak" not in repr(transport.requests)


def test_authentication_failure_refreshes_the_token_once() -> None:
    transport = FakeFeishuTransport(
        responses=[
            _token_response("tenant-old"),
            FeishuHttpResponse(
                status_code=401,
                headers={"content-type": "application/json"},
                json_body={"code": 99991663, "msg": "expired"},
            ),
            _token_response("tenant-new"),
            _ok({"message_id": "om_result"}),
        ]
    )

    result = _client(transport).reply_message(
        message_id="om_source",
        msg_type="text",
        content={"text": "status-only"},
    )

    assert result == {"message_id": "om_result"}
    assert transport.requests[-1].headers["Authorization"] == "Bearer tenant-new"
    assert len(transport.requests) == 4


def test_rate_limit_and_server_failure_use_bounded_exponential_backoff() -> None:
    sleeps: list[float] = []
    transport = FakeFeishuTransport(
        responses=[
            _token_response(),
            FeishuHttpResponse(
                status_code=429,
                headers={"content-type": "application/json"},
                json_body={"code": 99991400, "msg": "rate limited"},
            ),
            FeishuHttpResponse(
                status_code=503,
                headers={"content-type": "application/json"},
                json_body={"code": 1, "msg": "temporarily unavailable"},
            ),
            _ok(),
        ]
    )

    _client(transport, sleeps=sleeps).update_card(
        message_id="om_card",
        card={"config": {"wide_screen_mode": True}, "elements": []},
    )

    assert sleeps == [0.1, 0.2]
    assert len(transport.requests) == 4


def test_token_and_read_only_search_retain_bounded_retries() -> None:
    sleeps: list[float] = []
    transport = FakeFeishuTransport(
        responses=[
            FeishuHttpResponse(
                status_code=503,
                headers={"content-type": "application/json"},
                json_body={"code": 1, "msg": "temporarily unavailable"},
            ),
            _token_response(),
            FeishuTransportError("read-only search transport failure"),
            _ok({"items": []}),
        ]
    )

    result = _client(transport, sleeps=sleeps).search_bitable_records(
        app_token="app_table",
        table_id="tbl_runs",
        field_name="run_id",
        field_value="run-safe",
    )

    assert result == {"items": []}
    assert sleeps == [0.1, 0.1]
    assert len(transport.requests) == 4
    assert transport.requests[0].url.endswith(
        "/auth/v3/tenant_access_token/internal"
    )
    assert transport.requests[1].url == transport.requests[0].url
    assert transport.requests[2].url.endswith("/records/search")
    assert transport.requests[3].url == transport.requests[2].url


def test_upload_bitable_image_uses_the_drive_media_contract() -> None:
    transport = FakeFeishuTransport(
        responses=[_token_response(), _ok({"file_token": "file_curve_safe"})]
    )

    result = _client(transport).upload_bitable_media(
        app_token="app_table",
        filename="soh-curve.png",
        content_type="image/png",
        payload=b"audited-curve",
    )

    assert result == {"file_token": "file_curve_safe"}
    request = transport.requests[-1]
    assert request.method == "POST"
    assert request.url.endswith("/drive/v1/medias/upload_all")
    assert request.form == {
        "file_name": "soh-curve.png",
        "parent_type": "bitable_image",
        "parent_node": "app_table",
        "size": str(len(b"audited-curve")),
    }
    assert len(request.files) == 1
    assert request.files[0].field_name == "file"
    assert request.files[0].filename == "soh-curve.png"
    assert request.files[0].content_type == "image/png"
    assert request.files[0].payload == b"audited-curve"


def _invoke_non_idempotent_operation(client: FeishuClient, operation: str) -> None:
    if operation == "send_message":
        client.send_message(
            receive_id="oc_reviewed_chat",
            receive_id_type="chat_id",
            msg_type="text",
            content={"text": "status-only"},
        )
    elif operation == "reply_message":
        client.reply_message(
            message_id="om_source",
            msg_type="text",
            content={"text": "status-only"},
        )
    elif operation == "upload_image":
        client.upload_image(
            filename="audited-plot.png",
            content_type="image/png",
            payload=b"audited-image",
        )
    elif operation == "upload_file":
        client.upload_file(
            filename="audited-report.md",
            content_type="text/markdown",
            payload=b"audited-report",
        )
    elif operation == "upload_bitable_media":
        client.upload_bitable_media(
            app_token="app_table",
            filename="audited-plot.png",
            content_type="image/png",
            payload=b"audited-image",
        )
    elif operation == "create_bitable_record":
        client.create_bitable_record(
            app_token="app_table",
            table_id="tbl_runs",
            fields={"run_id": "run-safe"},
        )
    else:
        raise AssertionError(f"unknown test operation: {operation}")


@pytest.mark.parametrize(
    "operation",
    [
        "send_message",
        "reply_message",
        "upload_image",
        "upload_file",
        "upload_bitable_media",
        "create_bitable_record",
    ],
)
@pytest.mark.parametrize(
    ("failure", "expected_kind", "expected_status"),
    [
        (FeishuTransportError("ambiguous transport failure"), FeishuErrorKind.TRANSIENT, None),
        (
            FeishuHttpResponse(
                status_code=429,
                headers={"content-type": "application/json"},
                json_body={"code": 99991400, "msg": "rate limited"},
            ),
            FeishuErrorKind.RATE_LIMIT,
            429,
        ),
        (
            FeishuHttpResponse(
                status_code=503,
                headers={"content-type": "application/json"},
                json_body={"code": 1, "msg": "temporarily unavailable"},
            ),
            FeishuErrorKind.TRANSIENT,
            503,
        ),
    ],
)
def test_non_idempotent_posts_are_not_replayed_after_ambiguous_failures(
    operation: str,
    failure: FeishuHttpResponse | Exception,
    expected_kind: FeishuErrorKind,
    expected_status: int | None,
) -> None:
    sleeps: list[float] = []
    transport = FakeFeishuTransport(
        responses=[_token_response(), failure, _ok({"unexpected": "replay"})]
    )

    with pytest.raises(FeishuApiError) as captured:
        _invoke_non_idempotent_operation(_client(transport, sleeps=sleeps), operation)

    assert captured.value.kind is expected_kind
    assert captured.value.status_code == expected_status
    assert len(transport.requests) == 2
    assert sleeps == []


def test_permanent_api_error_is_structured_and_secret_free() -> None:
    transport = FakeFeishuTransport(
        responses=[
            _token_response(),
            FeishuHttpResponse(
                status_code=400,
                headers={"content-type": "application/json"},
                json_body={"code": 12345, "msg": "invalid request"},
            ),
        ]
    )

    with pytest.raises(FeishuApiError) as captured:
        _client(transport).send_message(
            receive_id="oc_reviewed_chat",
            receive_id_type="chat_id",
            msg_type="text",
            content={"text": "status-only"},
        )

    assert captured.value.kind is FeishuErrorKind.PERMANENT
    assert captured.value.status_code == 400
    assert captured.value.api_code == 12345
    assert "must-not-leak" not in str(captured.value)


def test_message_upload_download_and_bitable_request_shapes_are_deterministic() -> None:
    transport = FakeFeishuTransport(
        responses=[
            _token_response(),
            _ok({"message_id": "om_sent"}),
            _ok({"file_key": "file_uploaded"}),
            FeishuHttpResponse(
                status_code=200,
                headers={"content-type": "text/csv"},
                body=b"verified-bytes",
            ),
            _ok({"items": []}),
            _ok({"record": {"record_id": "rec_created"}}),
            _ok({"record": {"record_id": "rec_updated"}}),
        ]
    )
    client = _client(transport)

    sent = client.send_message(
        receive_id="oc_reviewed_chat",
        receive_id_type="chat_id",
        msg_type="interactive",
        content={"elements": []},
    )
    uploaded = client.upload_file(
        filename="audited-report.md",
        content_type="text/markdown; charset=utf-8",
        payload=b"report-from-tool-result",
    )
    downloaded = client.download_message_resource(
        message_id="om_source",
        file_key="file_source",
        resource_type="file",
    )
    found = client.search_bitable_records(
        app_token="app_table",
        table_id="tbl_runs",
        field_name="run_id",
        field_value="run-safe",
    )
    created = client.create_bitable_record(
        app_token="app_table",
        table_id="tbl_runs",
        fields={"run_id": "run-safe", "task_status": "QUEUED"},
    )
    updated = client.update_bitable_record(
        app_token="app_table",
        table_id="tbl_runs",
        record_id="rec_created",
        fields={"task_status": "COMPLETED"},
    )

    assert sent == {"message_id": "om_sent"}
    assert uploaded == {"file_key": "file_uploaded"}
    assert downloaded == b"verified-bytes"
    assert found == {"items": []}
    assert created == {"record": {"record_id": "rec_created"}}
    assert updated == {"record": {"record_id": "rec_updated"}}
    message_request = transport.requests[1]
    assert message_request.params == {"receive_id_type": "chat_id"}
    assert message_request.json_body == {
        "receive_id": "oc_reviewed_chat",
        "msg_type": "interactive",
        "content": json.dumps(
            {"elements": []},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    assert transport.requests[2].files[0].filename == "audited-report.md"
    assert (
        transport.requests[2].files[0].content_type
        == "text/markdown; charset=utf-8"
    )
    assert transport.requests[3].params == {"type": "file"}
    assert transport.requests[4].json_body == {
        "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "run_id",
                    "operator": "is",
                    "value": ["run-safe"],
                }
            ],
        },
        "field_names": [],
        "sort": [],
        "view_id": None,
        "automatic_fields": False,
    }


def test_download_resource_response_preserves_verified_http_metadata() -> None:
    transport = FakeFeishuTransport(
        responses=[
            _token_response(),
            FeishuHttpResponse(
                status_code=200,
                headers={
                    "content-type": "text/csv; charset=utf-8",
                    "content-length": "14",
                },
                body=b"verified-bytes",
            ),
        ]
    )
    client = _client(transport)

    response = client.download_message_resource_response(
        message_id="om_source",
        file_key="file_source",
        resource_type="file",
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-length"] == "14"
    assert response.body == b"verified-bytes"
