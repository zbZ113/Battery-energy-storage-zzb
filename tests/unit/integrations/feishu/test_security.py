from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr, ValidationError

from quanxin_life.integrations.feishu.security import (
    FeishuSignatureError,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)

NOW = datetime(2026, 7, 16, 8, 30, tzinfo=UTC)
BODY = b'{"schema":"2.0","header":{"event_id":"evt-001"}}'


def _signature(*, timestamp: str, nonce: str, encrypt_key: str, body: bytes) -> str:
    payload = timestamp.encode() + nonce.encode() + encrypt_key.encode() + body
    return hashlib.sha256(payload).hexdigest()


def _verifier() -> FeishuWebhookVerifier:
    return FeishuWebhookVerifier(
        FeishuWebhookSecrets(
            verification_token=SecretStr("verification-secret"),
            encrypt_key=SecretStr("encrypt-secret"),
            max_clock_skew_seconds=300,
        )
    )


def test_secret_configuration_repr_never_exposes_credentials() -> None:
    config = FeishuWebhookSecrets(
        verification_token=SecretStr("verification-secret"),
        encrypt_key=SecretStr("encrypt-secret"),
    )

    rendered = repr(config)

    assert "verification-secret" not in rendered
    assert "encrypt-secret" not in rendered
    assert "**********" in rendered


def test_secret_configuration_rejects_blank_credentials() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        FeishuWebhookSecrets(
            verification_token=SecretStr("   "),
            encrypt_key=SecretStr("encrypt-secret"),
        )


def test_verifier_accepts_an_authentic_fresh_request() -> None:
    timestamp = str(int(NOW.timestamp()))
    nonce = "nonce-001"

    _verifier().verify(
        timestamp=timestamp,
        nonce=nonce,
        signature=_signature(
            timestamp=timestamp,
            nonce=nonce,
            encrypt_key="encrypt-secret",
            body=BODY,
        ),
        body=BODY,
        now=NOW,
    )


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (str(int((NOW - timedelta(seconds=301)).timestamp())), "outside the allowed window"),
        (str(int((NOW + timedelta(seconds=301)).timestamp())), "outside the allowed window"),
        ("not-a-timestamp", "valid Unix timestamp"),
    ],
)
def test_verifier_rejects_stale_future_or_invalid_timestamps(
    timestamp: str,
    expected: str,
) -> None:
    with pytest.raises(FeishuSignatureError, match=expected):
        _verifier().verify(
            timestamp=timestamp,
            nonce="nonce-001",
            signature="0" * 64,
            body=BODY,
            now=NOW,
        )


def test_verifier_rejects_a_tampered_signature() -> None:
    timestamp = str(int(NOW.timestamp()))

    with pytest.raises(FeishuSignatureError, match="signature mismatch"):
        _verifier().verify(
            timestamp=timestamp,
            nonce="nonce-001",
            signature="0" * 64,
            body=BODY,
            now=NOW,
        )


@pytest.mark.parametrize("timestamp", [str(10**100), str(-(10**100))])
def test_verifier_normalizes_timestamp_overflow_to_signature_error(
    timestamp: str,
) -> None:
    with pytest.raises(FeishuSignatureError, match="valid Unix timestamp"):
        _verifier().verify(
            timestamp=timestamp,
            nonce="nonce-001",
            signature="0" * 64,
            body=BODY,
            now=NOW,
        )
