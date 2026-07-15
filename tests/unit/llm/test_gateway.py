from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from quanxin_life.core import LlmProviderConfig
from quanxin_life.llm import (
    LlmCapabilityUnavailableError,
    LlmRuntimeCredentials,
    LlmStructuredRequest,
    LlmTaskPurpose,
    OpenAiCompatibleGateway,
)

CHECKED_AT = datetime(2026, 7, 15, 15, tzinfo=UTC)


def _config(**overrides: object) -> LlmProviderConfig:
    values: dict[str, object] = {
        "config_version": "llm-provider-v1",
        "provider_id": "test-provider",
        "base_url": "https://llm.example.test/v1",
        "primary_model": "primary-model",
        "economy_model": "economy-model",
        "embedding_model": "embedding-model",
        "timeout_seconds": 10,
        "monthly_budget_cny": 100,
    }
    values.update(overrides)
    return LlmProviderConfig.model_validate(values)


def _json_response(content: str = '{"steps": []}') -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def test_runtime_credentials_never_expose_the_api_key() -> None:
    credentials = LlmRuntimeCredentials(api_key="secret-value-that-must-not-leak")

    assert "secret-value-that-must-not-leak" not in repr(credentials)
    assert "secret-value-that-must-not-leak" not in str(credentials)
    assert credentials.api_key.get_secret_value() == "secret-value-that-must-not-leak"


def test_capability_probe_is_independent_and_secret_safe() -> None:
    seen_authorization: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers["Authorization"])
        payload = request.read().decode("utf-8")
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})
        if '"stream":true' in payload:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"choices":[]}\n\ndata: [DONE]\n\n',
            )
        if '"tools"' in payload:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "capability_probe",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_json_response("{}"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = OpenAiCompatibleGateway(
        config=_config(),
        credentials=LlmRuntimeCredentials(api_key="probe-secret"),
        client=client,
        clock=lambda: CHECKED_AT,
    )

    probed = gateway.probe_capabilities()

    assert probed.supports_json_schema is True
    assert probed.supports_tool_calling is True
    assert probed.supports_streaming is True
    assert probed.supports_embeddings is True
    assert probed.capability_checked_at == CHECKED_AT
    assert probed.capability_warnings == ()
    assert seen_authorization == ["Bearer probe-secret"] * 4
    assert "probe-secret" not in repr(probed)


def test_probe_rejects_http_200_when_provider_ignores_requested_capabilities() -> None:
    gateway = OpenAiCompatibleGateway(
        config=_config(),
        credentials=LlmRuntimeCredentials(api_key="secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=_json_response())
            )
        ),
        clock=lambda: CHECKED_AT,
    )

    probed = gateway.probe_capabilities()

    assert probed.supports_json_schema is False
    assert probed.supports_tool_calling is False
    assert probed.supports_streaming is False
    assert probed.supports_embeddings is False
    assert "JSON_SCHEMA_PROBE_FAILED:INVALID_RESPONSE" in probed.capability_warnings
    assert "TOOL_CALLING_PROBE_FAILED:INVALID_RESPONSE" in probed.capability_warnings
    assert "STREAMING_PROBE_FAILED:INVALID_RESPONSE" in probed.capability_warnings
    assert "EMBEDDINGS_PROBE_FAILED:INVALID_RESPONSE" in probed.capability_warnings


def test_failed_probe_returns_codes_without_response_body_or_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="provider echoed forbidden-secret in an unsafe response body",
        )

    gateway = OpenAiCompatibleGateway(
        config=_config(),
        credentials=LlmRuntimeCredentials(api_key="forbidden-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: CHECKED_AT,
    )

    probed = gateway.probe_capabilities()

    assert probed.supports_json_schema is False
    assert probed.supports_tool_calling is False
    assert probed.supports_streaming is False
    assert probed.supports_embeddings is False
    assert len(probed.capability_warnings) == 4
    serialized = repr(probed.capability_warnings)
    assert "HTTP_401" in serialized
    assert "forbidden-secret" not in serialized
    assert "unsafe response body" not in serialized


def test_structured_completion_uses_declared_schema_and_returns_usage_only() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(__import__("json").loads(request.read())))
        return httpx.Response(200, json=_json_response('{"task_type":"diagnosis"}'))

    gateway = OpenAiCompatibleGateway(
        config=_config(supports_json_schema=True),
        credentials=LlmRuntimeCredentials(api_key="completion-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = LlmStructuredRequest(
        purpose=LlmTaskPurpose.INTENT,
        prompt_version="intent-prompt-v1",
        system_instruction="只输出结构化任务意图, 不生成任何电池工程数值。",
        user_instruction="分析已登记的数据批次。",
        response_schema={
            "type": "object",
            "properties": {"task_type": {"type": "string"}},
            "required": ["task_type"],
            "additionalProperties": False,
        },
        max_output_tokens=200,
    )

    response = gateway.complete_json(request)

    assert response.payload == {"task_type": "diagnosis"}
    assert response.model == "primary-model"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 5
    assert "completion-secret" not in repr(response)
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert captured[0]["response_format"]["json_schema"]["schema"] == request.response_schema


def test_structured_completion_refuses_unprobed_or_invalid_json() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_json_response("not-json"))
        )
    )
    credentials = LlmRuntimeCredentials(api_key="secret")

    request = LlmStructuredRequest(
        purpose=LlmTaskPurpose.PLAN,
        prompt_version="plan-prompt-v1",
        system_instruction="输出计划。",
        user_instruction="生成安全计划。",
        response_schema={"type": "object"},
        max_output_tokens=200,
    )
    with pytest.raises(LlmCapabilityUnavailableError, match="JSON Schema"):
        OpenAiCompatibleGateway(
            config=_config(supports_json_schema=None),
            credentials=credentials,
            client=client,
        ).complete_json(request)

    with pytest.raises(LlmCapabilityUnavailableError, match="valid JSON object"):
        OpenAiCompatibleGateway(
            config=_config(supports_json_schema=True),
            credentials=credentials,
            client=client,
        ).complete_json(request)


def test_structured_completion_validates_payload_against_schema_locally() -> None:
    request = LlmStructuredRequest(
        purpose=LlmTaskPurpose.INTENT,
        prompt_version="intent-prompt-v1",
        system_instruction="输出任务意图。",
        user_instruction="分析批次。",
        response_schema={
            "type": "object",
            "properties": {"task_type": {"type": "string"}},
            "required": ["task_type"],
            "additionalProperties": False,
        },
        max_output_tokens=200,
    )
    gateway = OpenAiCompatibleGateway(
        config=_config(supports_json_schema=True),
        credentials=LlmRuntimeCredentials(api_key="secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=_json_response("{}"))
            )
        ),
    )

    with pytest.raises(LlmCapabilityUnavailableError, match="declared JSON Schema"):
        gateway.complete_json(request)
