from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from workbench.client import (
    ApiClient,
    ApiHttpError,
    ApiResponseError,
    HttpResponse,
)


class RecordingTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        json_body: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        self.calls.append((method, url, json_body))
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
    client = ApiClient("http://127.0.0.1:8000/", transport=transport)

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
    client = ApiClient("https://api.example.test", transport=transport)

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
        )
    ]


def test_canonical_csv_registration_sends_bytes_and_contract_metadata_only() -> None:
    transport = RecordingTransport(
        [HttpResponse(200, {"record_batch_id": "canonical-csv-batch"})]
    )
    client = ApiClient("http://localhost:8000", transport=transport)
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
        )
    ]


def test_audited_markdown_is_returned_unchanged_for_display_and_download() -> None:
    markdown = "# Audited report\n\nEvidence stays on the server.\n"
    transport = RecordingTransport(
        [HttpResponse(200, {"result_id": "report/id", "markdown": markdown})]
    )
    client = ApiClient("http://localhost:8000", transport=transport)

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
    client = ApiClient("http://localhost:8000", transport=transport)

    result = client.get_tool_result("prediction/id")

    assert result == signed_payload
    assert transport.calls == [
        ("GET", "http://localhost:8000/v1/results/prediction%2Fid", None)
    ]


def test_http_and_malformed_response_fail_explicitly() -> None:
    http_error = ApiClient(
        "http://localhost:8000",
        transport=RecordingTransport([HttpResponse(503, {"detail": "workflow unavailable"})]),
    )
    malformed = ApiClient(
        "http://localhost:8000",
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
    client = ApiClient("http://localhost:8000", transport=transport)

    with pytest.raises(ValueError, match="must not be blank"):
        client.run_lifetime_workflow(**kwargs)

    assert transport.calls == []
