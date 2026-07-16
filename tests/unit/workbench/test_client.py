from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from workbench.client import (
    ApiClient,
    ApiHttpError,
    ApiResponseError,
    HttpResponse,
    WorkbenchConfig,
)


class AuthRecordingTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[
            tuple[str, str, Mapping[str, Any] | None, Mapping[str, str] | None]
        ] = []

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((method, url, json_body, headers))
        return self._responses.pop(0)


class RecordingTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[object, ...]] = []

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        call: tuple[object, ...] = (method, url, json_body)
        if headers is not None:
            call = (*call, headers)
        self.calls.append(call)
        return self._responses.pop(0)


def test_health_and_tool_discovery_use_only_http_api() -> None:
    transport = RecordingTransport(
        [
            HttpResponse(200, {"status": "ok", "service": "quanxin-life-tool-api"}),
            HttpResponse(
                200,
                [
                    {
                        "name": "predict_cycle_life",
                        "description": "server-owned schema",
                    }
                ],
            ),
        ]
    )
    client = ApiClient(
        "http://127.0.0.1:8000/",
        trusted_origin="http://127.0.0.1:8501",
        transport=transport,
    )

    health = client.health()
    tools = client.list_tools()

    assert health.status == "ok"
    assert health.service == "quanxin-life-tool-api"
    assert health.display_payload() == {
        "status": "ok",
        "service": "quanxin-life-tool-api",
    }
    assert tools == (
        {"name": "predict_cycle_life", "description": "server-owned schema"},
    )
    assert transport.calls == [
        ("GET", "http://127.0.0.1:8000/health", None),
        ("GET", "http://127.0.0.1:8000/v1/tools", None),
    ]


def test_lifetime_workflow_submits_identifiers_only() -> None:
    transport = RecordingTransport(
        [
            HttpResponse(
                200,
                {
                    "status": "completed",
                    "quality_result_id": "quality-id",
                    "feature_result_id": "feature-id",
                    "prediction_result_id": "prediction-id",
                    "calibration_result_id": "calibration-id",
                    "interval_result_id": "interval-id",
                    "decision_result_id": "decision-id",
                    "report_result_id": "report-id",
                    "warnings": ["server warning"],
                },
            )
        ]
    )
    client = ApiClient(
        "https://api.example.test",
        trusted_origin="https://app.example.test",
        transport=transport,
    )

    result = client.run_lifetime_workflow(
        record_batch_id=" batch-A ",
        calibration_cohort_id=" cohort-A ",
        policy_id=" policy-A ",
    )

    assert result.status == "completed"
    assert result.report_result_id == "report-id"
    assert result.warnings == ("server warning",)
    assert transport.calls == [
        (
            "POST",
            "https://api.example.test/v1/workflows/lifetime-decision",
            {
                "record_batch_id": "batch-A",
                "calibration_cohort_id": "cohort-A",
                "policy_id": "policy-A",
            },
            {"Origin": "https://app.example.test"},
        )
    ]


def test_canonical_csv_registration_sends_bytes_and_contract_metadata_only() -> None:
    transport = RecordingTransport(
        [HttpResponse(200, {"record_batch_id": "canonical-csv-batch"})]
    )
    client = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=transport,
    )
    registration = {"metadata": {"cell_id": "cell-1"}, "data_version": "v1"}

    batch_id = client.register_canonical_csv(b"a,b\n1,2\n", registration)

    assert batch_id == "canonical-csv-batch"
    assert transport.calls == [
        (
            "POST",
            "http://localhost:8000/v1/batches/canonical-csv",
            {
                "payload_base64": "YSxiCjEsMgo=",
                "registration": registration,
            },
            {"Origin": "http://localhost:8501"},
        )
    ]


def test_audited_markdown_is_returned_unchanged_for_display_and_download() -> None:
    markdown = "# Audited report\n\nEvidence stays on the server.\n"
    transport = RecordingTransport(
        [HttpResponse(200, {"result_id": "report/id", "markdown": markdown})]
    )
    client = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=transport,
    )

    report = client.get_audited_markdown("report/id")

    assert report.markdown == markdown
    assert report.download_bytes() == markdown.encode("utf-8")
    assert transport.calls == [
        (
            "GET",
            "http://localhost:8000/v1/reports/report%2Fid",
            None,
        )
    ]


def test_signed_tool_result_is_fetched_by_id_without_rewriting_payload() -> None:
    signed_payload = {
        "result_id": "prediction/id",
        "tool_name": "predict_cycle_life",
        "values": {"artifact": {"life_prediction": {"predicted_eol_cycle": 0}}},
        "warnings": [],
    }
    transport = RecordingTransport([HttpResponse(200, signed_payload)])
    client = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=transport,
    )

    result = client.get_tool_result("prediction/id")

    assert result == signed_payload
    assert transport.calls == [
        ("GET", "http://localhost:8000/v1/results/prediction%2Fid", None)
    ]


def test_http_and_malformed_response_fail_explicitly() -> None:
    http_error = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=RecordingTransport([HttpResponse(503, {"detail": "workflow unavailable"})]),
    )
    malformed = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=RecordingTransport([HttpResponse(200, {"status": "ok"})]),
    )

    with pytest.raises(ApiHttpError, match="workflow unavailable"):
        http_error.health()
    with pytest.raises(ApiResponseError, match="health response"):
        malformed.health()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "record_batch_id": " ",
            "calibration_cohort_id": "cohort",
            "policy_id": "policy",
        },
        {
            "record_batch_id": "batch",
            "calibration_cohort_id": "",
            "policy_id": "policy",
        },
        {
            "record_batch_id": "batch",
            "calibration_cohort_id": "cohort",
            "policy_id": "\t",
        },
    ],
)
def test_lifetime_workflow_rejects_blank_identifiers_before_http(
    kwargs: dict[str, str],
) -> None:
    transport = RecordingTransport([])
    client = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        transport=transport,
    )

    with pytest.raises(ValueError, match="must not be blank"):
        client.run_lifetime_workflow(**kwargs)

    assert transport.calls == []


@pytest.mark.parametrize(
    ("environment", "base_url", "origin", "message"),
    [
        ("production", "http://api.example.test", "https://app.example.test", "HTTPS"),
        ("production", "https://api.example.test", "http://app.example.test", "HTTPS"),
        ("development", "http://192.168.1.8:8000", "http://localhost:8501", "localhost"),
        ("development", "http://localhost:8000", "http://192.168.1.8:8501", "localhost"),
    ],
)
def test_workbench_config_rejects_insecure_or_remote_endpoints(
    environment: str,
    base_url: str,
    origin: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WorkbenchConfig.from_mapping(
            {
                "QUANXIN_ENVIRONMENT": environment,
                "QUANXIN_API_BASE_URL": base_url,
                "QUANXIN_TRUSTED_ORIGIN": origin,
            }
        )


def test_workbench_config_requires_explicit_environment_and_urls() -> None:
    for missing in (
        "QUANXIN_ENVIRONMENT",
        "QUANXIN_API_BASE_URL",
        "QUANXIN_TRUSTED_ORIGIN",
    ):
        values = {
            "QUANXIN_ENVIRONMENT": "development",
            "QUANXIN_API_BASE_URL": "http://localhost:8000",
            "QUANXIN_TRUSTED_ORIGIN": "http://localhost:8501",
        }
        del values[missing]
        with pytest.raises(ValueError, match=missing):
            WorkbenchConfig.from_mapping(values)


def test_workbench_config_rejects_credentials_embedded_in_api_url() -> None:
    with pytest.raises(ValueError, match="credentials"):
        WorkbenchConfig.from_mapping(
            {
                "QUANXIN_ENVIRONMENT": "production",
                "QUANXIN_API_BASE_URL": "https://user:secret@api.example.test",
                "QUANXIN_TRUSTED_ORIGIN": "https://app.example.test",
            }
        )


def test_httpx_client_keeps_session_cookie_and_posts_trusted_origin() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/auth/login":
            return httpx.Response(
                200,
                headers={"set-cookie": "quanxin_dev_session=opaque; HttpOnly; Path=/"},
                json={
                    "user_id": "user-1",
                    "username": "member@example.test",
                    "role": "MEMBER",
                    "must_change_password": True,
                    "expires_at": "2026-07-17T00:00:00Z",
                },
            )
        assert request.headers["cookie"] == "quanxin_dev_session=opaque"
        return httpx.Response(
            200,
            json={
                "user_id": "user-1",
                "username": "member@example.test",
                "role": "MEMBER",
                "must_change_password": True,
            },
        )

    client = ApiClient(
        "http://localhost:8000",
        trusted_origin="http://localhost:8501",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    session = client.login(username="member@example.test", password="not-logged")
    principal = client.me()

    assert session.must_change_password is True
    assert principal.username == "member@example.test"
    assert requests[0].headers["origin"] == "http://localhost:8501"
    assert "origin" not in requests[1].headers


def test_change_password_and_logout_use_exact_auth_contract_and_origin() -> None:
    transport = AuthRecordingTransport(
        [
            HttpResponse(
                200,
                {
                    "user_id": "user-1",
                    "username": "member@example.test",
                    "role": "MEMBER",
                    "must_change_password": False,
                    "expires_at": "2026-07-17T00:00:00Z",
                },
            ),
            HttpResponse(204, None),
        ]
    )
    client = ApiClient(
        "http://127.0.0.1:8000",
        trusted_origin="http://127.0.0.1:8501",
        transport=transport,
    )

    changed = client.change_password(
        current_password=" temporary-secret ",
        new_password="replacement-secret",
    )
    client.logout()

    assert changed.must_change_password is False
    assert transport.calls == [
        (
            "POST",
            "http://127.0.0.1:8000/v1/auth/change-password",
            {
                "current_password": " temporary-secret ",
                "new_password": "replacement-secret",
            },
            {"Origin": "http://127.0.0.1:8501"},
        ),
        (
            "POST",
            "http://127.0.0.1:8000/v1/auth/logout",
            None,
            {"Origin": "http://127.0.0.1:8501"},
        ),
    ]


def test_password_bytes_are_not_trimmed_or_normalized_by_client() -> None:
    transport = AuthRecordingTransport(
        [
            HttpResponse(
                200,
                {
                    "user_id": "user-1",
                    "username": "member@example.test",
                    "role": "MEMBER",
                    "must_change_password": False,
                    "expires_at": "2026-07-17T00:00:00Z",
                },
            )
        ]
    )
    client = ApiClient(
        "https://api.example.test",
        trusted_origin="https://app.example.test",
        transport=transport,
    )

    client.login(username=" member@example.test ", password=" padded secret ")

    assert transport.calls[0][2] == {
        "username": "member@example.test",
        "password": " padded secret ",
    }
