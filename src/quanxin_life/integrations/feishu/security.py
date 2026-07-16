"""Request authentication for Feishu event callbacks.

The verifier operates on the exact request body bytes.  It does not log or
return credentials, payloads, or calculated signatures.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class FeishuSignatureError(ValueError):
    """Raised when a Feishu callback cannot be authenticated."""


class FeishuWebhookSecrets(BaseModel):
    """Secret callback configuration with redacted representations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verification_token: SecretStr
    encrypt_key: SecretStr
    max_clock_skew_seconds: int = Field(default=300, ge=1, le=900)

    @field_validator("verification_token", "encrypt_key")
    @classmethod
    def secret_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Feishu credentials must not be blank")
        return value


class FeishuWebhookVerifier:
    """Verify callback freshness and the Feishu SHA-256 request signature."""

    def __init__(self, config: FeishuWebhookSecrets) -> None:
        self._config = config

    def verify(
        self,
        *,
        timestamp: str,
        nonce: str,
        signature: str,
        body: bytes,
        now: datetime,
    ) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Feishu verification clock must include a timezone")
        if not nonce.strip() or not signature.strip():
            raise FeishuSignatureError("Feishu signature headers must not be blank")
        try:
            request_seconds = int(timestamp)
            request_time = datetime.fromtimestamp(request_seconds, tz=UTC)
        except (ValueError, OverflowError, OSError) as exc:
            raise FeishuSignatureError("Feishu timestamp must be a valid Unix timestamp") from exc
        normalized_now = now.astimezone(UTC)
        skew = abs((normalized_now - request_time).total_seconds())
        if skew > self._config.max_clock_skew_seconds:
            raise FeishuSignatureError("Feishu timestamp is outside the allowed window")

        signed = (
            timestamp.encode("utf-8")
            + nonce.encode("utf-8")
            + self._config.encrypt_key.get_secret_value().encode("utf-8")
            + body
        )
        expected = hashlib.sha256(signed).hexdigest()
        if not hmac.compare_digest(expected, signature.lower()):
            raise FeishuSignatureError("Feishu signature mismatch")
