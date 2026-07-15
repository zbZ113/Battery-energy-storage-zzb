"""Secret-safe OpenAI-compatible gateway for structured Agent planning.

The gateway never computes battery engineering values.  It only probes remote
capabilities and requests strict JSON objects for intent, plan and replan
tasks.  Callers must validate returned payloads against public core contracts
before the deterministic orchestrator is allowed to execute anything.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, SecretStr, field_validator

from quanxin_life.core import LlmProviderConfig, sha256_canonical
from quanxin_life.core.schemas import ContractModel, JsonMapping, _json_mapping

Clock = Callable[[], datetime]
ProbeValidator = Callable[[httpx.Response], bool]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LlmGatewayError(RuntimeError):
    """Base failure for the optional remote planning layer."""


class LlmCapabilityUnavailableError(LlmGatewayError):
    """Raised when a required capability or structured response is unavailable."""


class LlmProviderUnavailableError(LlmGatewayError):
    """Raised without response content when the remote provider cannot serve a request."""


class LlmTaskPurpose(StrEnum):
    """LLM tasks allowed by the competition safety boundary."""

    INTENT = "INTENT"
    PLAN = "PLAN"
    REPLAN = "REPLAN"


class LlmRuntimeCredentials(ContractModel):
    """Runtime-only credentials; never add this model to persistence schemas."""

    api_key: SecretStr

    @field_validator("api_key")
    @classmethod
    def api_key_is_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("LLM API key must not be blank")
        return value


class LlmStructuredRequest(ContractModel):
    """A bounded prompt for one JSON-only planning operation."""

    purpose: LlmTaskPurpose
    prompt_version: str = Field(min_length=1)
    system_instruction: str = Field(min_length=2, max_length=8_000)
    user_instruction: str = Field(min_length=2, max_length=12_000)
    response_schema: JsonMapping
    use_economy_model: bool = False
    max_output_tokens: int = Field(default=1_000, ge=1, le=4_000)

    @field_validator("prompt_version", "system_instruction", "user_instruction")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("LLM request text must not be blank")
        return normalized

    @field_validator("response_schema")
    @classmethod
    def response_schema_is_json_object(cls, value: JsonMapping) -> JsonMapping:
        _json_mapping(value)
        if value.get("type") != "object":
            raise ValueError("LLM response schema must describe a JSON object")
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as exc:
            raise ValueError("LLM response schema must be a valid JSON Schema") from exc
        return value


class LlmStructuredResponse(ContractModel):
    """Validated JSON payload plus provider usage, without raw provider content."""

    purpose: LlmTaskPurpose
    prompt_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    payload: JsonMapping
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("payload")
    @classmethod
    def payload_is_json_safe(cls, value: JsonMapping) -> JsonMapping:
        _json_mapping(value)
        return value


class OpenAiCompatibleGateway:
    """Small HTTP gateway whose public outputs never contain credentials or raw text."""

    def __init__(
        self,
        *,
        config: LlmProviderConfig,
        credentials: LlmRuntimeCredentials,
        client: httpx.Client | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._config = LlmProviderConfig.model_validate(config.model_dump(mode="json"))
        self._credentials = credentials
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def probe_capabilities(self) -> LlmProviderConfig:
        """Probe each optional capability independently and return a secret-free report."""

        warnings: list[str] = []
        json_schema = self._probe(
            "JSON_SCHEMA",
            "/chat/completions",
            self._json_schema_probe_payload(),
            warnings,
            validator=self._valid_json_schema_probe,
        )
        tool_calling = self._probe(
            "TOOL_CALLING",
            "/chat/completions",
            self._tool_calling_probe_payload(),
            warnings,
            validator=self._valid_tool_calling_probe,
        )
        streaming = self._probe(
            "STREAMING",
            "/chat/completions",
            self._streaming_probe_payload(),
            warnings,
            validator=self._valid_streaming_probe,
        )
        if self._config.embedding_model is None:
            embeddings = False
            warnings.append("EMBEDDINGS_PROBE_SKIPPED:NO_MODEL")
        else:
            embeddings = self._probe(
                "EMBEDDINGS",
                "/embeddings",
                {"model": self._config.embedding_model, "input": "capability probe"},
                warnings,
                validator=self._valid_embedding_probe,
            )
        checked_at = self._clock()
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError("LLM capability clock must return a timezone-aware datetime")
        return self._config.model_copy(
            update={
                "supports_json_schema": json_schema,
                "supports_tool_calling": tool_calling,
                "supports_streaming": streaming,
                "supports_embeddings": embeddings,
                "capability_checked_at": checked_at.astimezone(UTC),
                "capability_warnings": tuple(warnings),
            }
        )

    def complete_json(self, request: LlmStructuredRequest) -> LlmStructuredResponse:
        """Request one strict JSON object or fail explicitly for deterministic fallback."""

        validated = LlmStructuredRequest.model_validate(request.model_dump(mode="json"))
        if self._config.supports_json_schema is not True:
            raise LlmCapabilityUnavailableError(
                "JSON Schema capability is not verified; use the fixed workflow fallback"
            )
        model = (
            self._config.economy_model
            if validated.use_economy_model and self._config.economy_model is not None
            else self._config.primary_model
        )
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": validated.system_instruction},
                {"role": "user", "content": validated.user_instruction},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"quanxin_{validated.purpose.value.lower()}",
                    "strict": True,
                    "schema": validated.response_schema,
                },
            },
            "max_tokens": validated.max_output_tokens,
            "temperature": 0,
        }
        response = self._post("/chat/completions", request_payload)
        response_body = self._safe_response_json(response)
        content = self._extract_content(response_body)
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LlmCapabilityUnavailableError(
                "provider did not return a valid JSON object; use the fixed workflow fallback"
            ) from exc
        if not isinstance(parsed, dict):
            raise LlmCapabilityUnavailableError(
                "provider did not return a valid JSON object; use the fixed workflow fallback"
            )
        try:
            Draft202012Validator(validated.response_schema).validate(parsed)
        except JsonSchemaValidationError as exc:
            raise LlmCapabilityUnavailableError(
                "provider payload does not satisfy the declared JSON Schema"
            ) from exc
        usage = response_body.get("usage")
        if not isinstance(usage, dict):
            raise LlmCapabilityUnavailableError("provider response omitted structured token usage")
        prompt_tokens = self._usage_integer(usage, "prompt_tokens")
        completion_tokens = self._usage_integer(usage, "completion_tokens")
        total_tokens = self._usage_integer(usage, "total_tokens")
        return LlmStructuredResponse(
            purpose=validated.purpose,
            prompt_version=validated.prompt_version,
            model=model,
            payload=parsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_hash=sha256_canonical(validated.model_dump(mode="json")),
        )

    def _probe(
        self,
        name: str,
        endpoint: str,
        payload: JsonMapping,
        warnings: list[str],
        *,
        validator: ProbeValidator,
    ) -> bool:
        try:
            response = self._client.post(
                self._url(endpoint),
                headers=self._headers(),
                json=payload,
                timeout=self._config.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            warnings.append(f"{name}_PROBE_FAILED:HTTP_ERROR:{type(exc).__name__}")
            return False
        if not response.is_success:
            warnings.append(f"{name}_PROBE_FAILED:HTTP_{response.status_code}")
            return False
        try:
            valid_response = validator(response)
        except (KeyError, IndexError, TypeError, ValueError, LlmGatewayError):
            valid_response = False
        if not valid_response:
            warnings.append(f"{name}_PROBE_FAILED:INVALID_RESPONSE")
            return False
        return True

    def _post(self, endpoint: str, payload: JsonMapping) -> httpx.Response:
        try:
            response = self._client.post(
                self._url(endpoint),
                headers=self._headers(),
                json=payload,
                timeout=self._config.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise LlmProviderUnavailableError(
                f"LLM provider request failed:{type(exc).__name__}"
            ) from exc
        if not response.is_success:
            raise LlmProviderUnavailableError(
                f"LLM provider request failed:HTTP_{response.status_code}"
            )
        return response

    def _url(self, endpoint: str) -> str:
        return f"{str(self._config.base_url).rstrip('/')}{endpoint}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._credentials.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _json_schema_probe_payload(self) -> JsonMapping:
        return {
            "model": self._config.primary_model,
            "messages": [{"role": "user", "content": "Return an empty JSON object."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "quanxin_capability_probe",
                    "strict": True,
                    "schema": {"type": "object", "additionalProperties": False},
                },
            },
            "max_tokens": 8,
            "temperature": 0,
        }

    def _tool_calling_probe_payload(self) -> JsonMapping:
        return {
            "model": self._config.primary_model,
            "messages": [{"role": "user", "content": "Call the capability probe tool."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "capability_probe",
                        "description": "No-op capability probe",
                        "parameters": {"type": "object", "additionalProperties": False},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "capability_probe"}},
            "max_tokens": 8,
            "temperature": 0,
        }

    def _streaming_probe_payload(self) -> JsonMapping:
        return {
            "model": self._config.primary_model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "stream": True,
            "max_tokens": 2,
            "temperature": 0,
        }

    @classmethod
    def _valid_json_schema_probe(cls, response: httpx.Response) -> bool:
        envelope = cls._safe_response_json(response)
        content = cls._extract_content(envelope)
        value = json.loads(content)
        return isinstance(value, dict) and not value

    @classmethod
    def _valid_tool_calling_probe(cls, response: httpx.Response) -> bool:
        envelope = cls._safe_response_json(response)
        try:
            tool_calls = envelope["choices"][0]["message"]["tool_calls"]
        except (KeyError, IndexError, TypeError):
            return False
        if not isinstance(tool_calls, list) or not tool_calls:
            return False
        first = tool_calls[0]
        return (
            isinstance(first, dict)
            and isinstance(first.get("function"), dict)
            and first["function"].get("name") == "capability_probe"
        )

    @staticmethod
    def _valid_streaming_probe(response: httpx.Response) -> bool:
        content_type = response.headers.get("content-type", "").lower()
        return "text/event-stream" in content_type and any(
            line.startswith("data:") for line in response.text.splitlines()
        )

    @classmethod
    def _valid_embedding_probe(cls, response: httpx.Response) -> bool:
        envelope = cls._safe_response_json(response)
        try:
            embedding = envelope["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError):
            return False
        return (
            isinstance(embedding, list)
            and bool(embedding)
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in embedding
            )
        )

    @staticmethod
    def _safe_response_json(response: httpx.Response) -> JsonMapping:
        try:
            value = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise LlmCapabilityUnavailableError(
                "provider response was not a valid JSON envelope"
            ) from exc
        if not isinstance(value, dict):
            raise LlmCapabilityUnavailableError("provider response was not a JSON object envelope")
        return value

    @staticmethod
    def _extract_content(response: JsonMapping) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmCapabilityUnavailableError(
                "provider response omitted structured message content"
            ) from exc
        if not isinstance(content, str):
            raise LlmCapabilityUnavailableError(
                "provider response message content must be a JSON string"
            )
        return content

    @staticmethod
    def _usage_integer(usage: JsonMapping, field_name: str) -> int:
        value: Any = usage.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise LlmCapabilityUnavailableError(
                f"provider response contains invalid {field_name} usage"
            )
        return value
