from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from quanxin_life.integrations.feishu.events import (
    FeishuEventError,
    FeishuEventProcessor,
    FeishuPayloadDecryptionUnavailable,
    FeishuPayloadDecryptor,
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
    FeishuReceiptStore,
)
from quanxin_life.integrations.feishu.security import (
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)

NOW = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)
TIMESTAMP = str(int(NOW.timestamp()))
NONCE = "nonce-001"


class _MemoryReceiptStore:
    def __init__(self) -> None:
        self.claims: list[dict[str, object]] = []
        self.processed: list[tuple[str, str, datetime]] = []

    def claim(
        self,
        *,
        event_id: str,
        event_type: str,
        payload_sha256: str,
        received_at: datetime,
    ) -> FeishuReceiptClaim:
        previous = next(
            (item for item in self.claims if item["event_id"] == event_id), None
        )
        if previous is not None:
            if (
                previous["event_type"] != event_type
                or previous["payload_sha256"] != payload_sha256
            ):
                return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)
            return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.PROCESSED)
        self.claims.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "payload_sha256": payload_sha256,
                "received_at": received_at,
            }
        )
        return FeishuReceiptClaim(
            status=FeishuReceiptClaimStatus.NEW,
            claim_token="c" * 64,
            attempt=1,
        )

    def mark_processed(
        self, *, event_id: str, claim_token: str, processed_at: datetime
    ) -> None:
        self.processed.append((event_id, claim_token, processed_at))

    def mark_failed(
        self, *, event_id: str, claim_token: str, failed_at: datetime
    ) -> None:
        del event_id, claim_token, failed_at

    def renew_claim(
        self, *, event_id: str, claim_token: str, renewed_at: datetime
    ) -> datetime:
        self.renewed = (event_id, claim_token, renewed_at)
        return renewed_at


def _processor(
    store: FeishuReceiptStore,
    *,
    decryptor: FeishuPayloadDecryptor | None = None,
) -> FeishuEventProcessor:
    secrets = FeishuWebhookSecrets(
        verification_token=SecretStr("verification-secret"),
        encrypt_key=SecretStr("encrypt-secret"),
    )
    return FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(secrets),
        verification_token=secrets.verification_token,
        receipts=store,
        decryptor=decryptor,
    )


def _signed_headers(body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(
        TIMESTAMP.encode() + NONCE.encode() + b"encrypt-secret" + body
    ).hexdigest()
    return {
        "X-Lark-Request-Timestamp": TIMESTAMP,
        "X-Lark-Request-Nonce": NONCE,
        "X-Lark-Signature": digest,
    }


def test_url_verification_returns_only_the_verified_challenge_without_signature() -> None:
    store = _MemoryReceiptStore()
    body = json.dumps(
        {
            "type": "url_verification",
            "token": "verification-secret",
            "challenge": "challenge-value",
        },
        separators=(",", ":"),
    ).encode()

    outcome = _processor(store).handle(headers={}, body=body, now=NOW)

    assert outcome.response == {"challenge": "challenge-value"}
    assert outcome.kind == "url_verification"
    assert outcome.event_id is None
    assert store.claims == []


def test_url_verification_rejects_the_wrong_verification_token() -> None:
    body = json.dumps(
        {
            "type": "url_verification",
            "token": "wrong-token",
            "challenge": "challenge-value",
        },
        separators=(",", ":"),
    ).encode()

    with pytest.raises(FeishuEventError, match="verification token mismatch"):
        _processor(_MemoryReceiptStore()).handle(
            headers={}, body=body, now=NOW
        )


def test_event_claim_uses_existing_receipt_row_semantics_and_blocks_replay() -> None:
    store = _MemoryReceiptStore()
    body = json.dumps(
        {
            "schema": "2.0",
            "header": {
                "event_id": "evt-001",
                "event_type": "im.message.receive_v1",
                "token": "verification-secret",
            },
            "event": {"message": {"message_id": "om-001"}},
        },
        separators=(",", ":"),
    ).encode()
    processor = _processor(store)

    first = processor.handle(headers=_signed_headers(body), body=body, now=NOW)
    replay = processor.handle(headers=_signed_headers(body), body=body, now=NOW)

    assert first.kind == "event"
    assert first.event_id == "evt-001"
    assert first.duplicate is False
    assert replay.duplicate is True
    assert first.receipt_status is FeishuReceiptClaimStatus.NEW
    assert replay.receipt_status is FeishuReceiptClaimStatus.PROCESSED
    assert first.claim_token == "c" * 64
    assert replay.claim_token is None
    assert len(store.claims) == 1
    assert store.claims[0] == {
        "event_id": "evt-001",
        "event_type": "im.message.receive_v1",
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "received_at": NOW,
    }


def test_event_can_be_marked_processed_without_storing_the_payload() -> None:
    store = _MemoryReceiptStore()
    processor = _processor(store)

    processor.mark_processed(
        event_id="evt-001", claim_token="c" * 64, processed_at=NOW
    )

    assert store.processed == [("evt-001", "c" * 64, NOW)]


def test_event_claim_can_be_renewed_without_exposing_payload() -> None:
    store = _MemoryReceiptStore()
    renewed_until = _processor(store).renew_claim(
        event_id="evt-001", claim_token="c" * 64, renewed_at=NOW
    )

    assert renewed_until == NOW
    assert store.renewed == ("evt-001", "c" * 64, NOW)


@pytest.mark.parametrize(
    ("claim_status", "duplicate"),
    [
        (FeishuReceiptClaimStatus.NEW, False),
        (FeishuReceiptClaimStatus.RETRYABLE, False),
        (FeishuReceiptClaimStatus.PROCESSED, True),
        (FeishuReceiptClaimStatus.IN_PROGRESS, True),
    ],
)
def test_receipt_claim_status_controls_processing(
    claim_status: FeishuReceiptClaimStatus,
    duplicate: bool,
) -> None:
    class _StatusStore(_MemoryReceiptStore):
        def claim(self, **kwargs: object) -> FeishuReceiptClaim:
            del kwargs
            return FeishuReceiptClaim(
                status=claim_status,
                claim_token="c" * 64
                if claim_status
                in {FeishuReceiptClaimStatus.NEW, FeishuReceiptClaimStatus.RETRYABLE}
                else None,
                attempt=1,
            )

    body = _event_body(event_id="evt-status")
    outcome = _processor(_StatusStore()).handle(
        headers=_signed_headers(body), body=body, now=NOW
    )

    assert outcome.receipt_status is claim_status
    assert outcome.duplicate is duplicate


def test_receipt_conflict_rejects_reused_event_id_with_changed_payload() -> None:
    store = _MemoryReceiptStore()
    first = _event_body(event_id="evt-conflict", message_id="om-001")
    changed = _event_body(event_id="evt-conflict", message_id="om-002")
    processor = _processor(store)
    processor.handle(headers=_signed_headers(first), body=first, now=NOW)

    with pytest.raises(FeishuEventError, match="conflicts"):
        processor.handle(headers=_signed_headers(changed), body=changed, now=NOW)


def test_failed_event_is_marked_retryable_by_store_port() -> None:
    class _FailedStore(_MemoryReceiptStore):
        def mark_failed(
            self, *, event_id: str, claim_token: str, failed_at: datetime
        ) -> None:
            self.failed = (event_id, claim_token, failed_at)

    store = _FailedStore()
    _processor(store).mark_failed(
        event_id="evt-001", claim_token="c" * 64, failed_at=NOW
    )

    assert store.failed == ("evt-001", "c" * 64, NOW)


class _FakeDecryptor:
    def __init__(self, plaintext: bytes) -> None:
        self.plaintext = plaintext
        self.calls: list[str] = []

    def decrypt(self, encrypted: str) -> bytes:
        self.calls.append(encrypted)
        return self.plaintext


def test_encrypted_event_uses_injected_decryptor_after_signature_verification() -> None:
    plaintext = _event_body(event_id="evt-encrypted")
    envelope = b'{"encrypt":"opaque-ciphertext"}'
    decryptor = _FakeDecryptor(plaintext)

    outcome = _processor(_MemoryReceiptStore(), decryptor=decryptor).handle(
        headers=_signed_headers(envelope), body=envelope, now=NOW
    )

    assert outcome.event_id == "evt-encrypted"
    assert decryptor.calls == ["opaque-ciphertext"]


def test_encrypted_event_fails_explicitly_without_a_decryptor() -> None:
    envelope = b'{"encrypt":"opaque-ciphertext"}'

    with pytest.raises(FeishuPayloadDecryptionUnavailable, match="not configured"):
        _processor(_MemoryReceiptStore()).handle(
            headers=_signed_headers(envelope), body=envelope, now=NOW
        )


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"schema":"2.0","header":{"event_id":"evt-001"}}',
    ],
)
def test_event_processor_rejects_malformed_or_incomplete_payloads(body: bytes) -> None:
    with pytest.raises(FeishuEventError):
        _processor(_MemoryReceiptStore()).handle(
            headers=_signed_headers(body), body=body, now=NOW
        )


def _event_body(*, event_id: str, message_id: str = "om-001") -> bytes:
    return json.dumps(
        {
            "schema": "2.0",
            "header": {
                "event_id": event_id,
                "event_type": "im.message.receive_v1",
                "token": "verification-secret",
            },
            "event": {"message": {"message_id": message_id}},
        },
        separators=(",", ":"),
    ).encode()
