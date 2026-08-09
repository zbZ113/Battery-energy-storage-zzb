from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from pydantic import SecretStr

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.feishu import (
    MAX_FEISHU_CALLBACK_BYTES,
    FeishuEventRouter,
    FeishuEventRouteStatus,
    create_feishu_http_adapter,
)
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.integrations.feishu import (
    FeishuEventProcessor,
    FeishuEventReference,
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)

NOW = datetime(2026, 7, 16, 20, 0, tzinfo=UTC)
TOKEN = "reviewed-verification-token"
ENCRYPT_KEY = "reviewed-encrypt-key"


class _MemoryReceiptStore:
    def __init__(self, *, fail_first_completion: bool = False) -> None:
        self.statuses: dict[str, FeishuReceiptClaimStatus] = {}
        self.identities: dict[str, tuple[str, str]] = {}
        self.claim_tokens: dict[str, str] = {}
        self.attempts: dict[str, int] = {}
        self.fail_first_completion = fail_first_completion
        self.completion_attempts = 0

    def claim(
        self,
        *,
        event_id: str,
        event_type: str,
        payload_sha256: str,
        received_at: datetime,
    ) -> FeishuReceiptClaim:
        del received_at
        identity = (event_type, payload_sha256)
        previous = self.identities.get(event_id)
        if previous is not None and previous != identity:
            return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)
        self.identities[event_id] = identity
        current = self.statuses.get(event_id)
        if current is FeishuReceiptClaimStatus.PROCESSED:
            return FeishuReceiptClaim(status=current)
        if current is FeishuReceiptClaimStatus.IN_PROGRESS:
            return FeishuReceiptClaim(status=current)
        attempt = self.attempts.get(event_id, 0) + 1
        token = f"{attempt:064x}"
        self.attempts[event_id] = attempt
        self.statuses[event_id] = FeishuReceiptClaimStatus.IN_PROGRESS
        self.claim_tokens[event_id] = token
        return FeishuReceiptClaim(
            status=(
                FeishuReceiptClaimStatus.NEW
                if current is None
                else FeishuReceiptClaimStatus.RETRYABLE
            ),
            claim_token=token,
            attempt=attempt,
        )

    def mark_processed(
        self, *, event_id: str, claim_token: str, processed_at: datetime
    ) -> None:
        del processed_at
        self.completion_attempts += 1
        if self.fail_first_completion and self.completion_attempts == 1:
            raise RuntimeError("simulated durable completion failure")
        assert self.claim_tokens[event_id] == claim_token
        self.statuses[event_id] = FeishuReceiptClaimStatus.PROCESSED
        del self.claim_tokens[event_id]

    def mark_failed(
        self, *, event_id: str, claim_token: str, failed_at: datetime
    ) -> None:
        del failed_at
        assert self.claim_tokens[event_id] == claim_token
        self.statuses[event_id] = FeishuReceiptClaimStatus.RETRYABLE
        del self.claim_tokens[event_id]

    def renew_claim(
        self, *, event_id: str, claim_token: str, renewed_at: datetime
    ) -> datetime:
        assert self.claim_tokens[event_id] == claim_token
        return renewed_at


@dataclass
class _RecordingRouter:
    fail_first: bool = False

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.created_task_events: set[str] = set()

    def route(
        self, *, event_id: str, event_type: str, claim_token: str
    ) -> FeishuEventRouteStatus:
        self.calls.append((event_id, event_type, claim_token))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("sensitive downstream detail")
        if event_id in self.created_task_events:
            return FeishuEventRouteStatus.ALREADY_ENQUEUED
        self.created_task_events.add(event_id)
        return FeishuEventRouteStatus.ENQUEUED


class _NoConfirmationRouter:
    def route(self, *, event_id: str, event_type: str, claim_token: str) -> None:
        del event_id, event_type, claim_token


@dataclass
class _RecordingSanitizedRouter:
    def __post_init__(self) -> None:
        self.calls: list[tuple[FeishuEventReference, str]] = []

    def route_event(
        self, *, event: FeishuEventReference, claim_token: str
    ) -> FeishuEventRouteStatus:
        self.calls.append((event, claim_token))
        return FeishuEventRouteStatus.ENQUEUED


def _client(
    *,
    router: FeishuEventRouter | None = None,
    fail_first_completion: bool = False,
) -> tuple[TestClient, _MemoryReceiptStore]:
    receipts = _MemoryReceiptStore(fail_first_completion=fail_first_completion)
    secrets = FeishuWebhookSecrets(
        verification_token=SecretStr(TOKEN),
        encrypt_key=SecretStr(ENCRYPT_KEY),
    )
    processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(secrets),
        verification_token=SecretStr(TOKEN),
        receipts=receipts,
    )
    adapter = create_feishu_http_adapter(
        processor,
        router=router,
        now_factory=lambda: NOW,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        feishu_adapter=adapter,
    )
    return TestClient(app), receipts


def _event_body(*, event_id: str = "evt-safe-001", token: str = TOKEN) -> bytes:
    return json.dumps(
        {
            "schema": "2.0",
            "header": {
                "event_id": event_id,
                "event_type": "im.message.receive_v1",
                "token": token,
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou-sender"}},
                "message": {
                    "message_id": "om-source",
                    "chat_id": "oc-chat",
                    "message_type": "text",
                    "content": '{"text":"must-not-be-forwarded"}',
                },
            },
        },
        separators=(",", ":"),
    ).encode()


def _signed_headers(body: bytes, *, signature: str | None = None) -> dict[str, str]:
    timestamp = str(int(NOW.timestamp()))
    nonce = "safe-nonce"
    expected = hashlib.sha256(
        timestamp.encode() + nonce.encode() + ENCRYPT_KEY.encode() + body
    ).hexdigest()
    return {
        "X-Lark-Request-Timestamp": timestamp,
        "X-Lark-Request-Nonce": nonce,
        "X-Lark-Signature": signature or expected,
        "Content-Type": "application/json",
    }


def test_url_verification_returns_only_the_challenge() -> None:
    client, _ = _client()

    response = client.post(
        "/v1/integrations/feishu/events",
        content=json.dumps(
            {"type": "url_verification", "token": TOKEN, "challenge": "challenge-1"}
        ).encode(),
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "challenge-1"}


def test_valid_event_routes_once_and_duplicate_delivery_is_acknowledged() -> None:
    router = _RecordingRouter()
    client, _ = _client(router=router)
    body = _event_body()

    first = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )
    repeated = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert first.status_code == 200, first.text
    assert repeated.status_code == 200, repeated.text
    assert first.json() == {}
    assert repeated.json() == {}
    assert len(router.calls) == 1
    event_id, event_type, claim_token = router.calls[0]
    assert (event_id, event_type) == ("evt-safe-001", "im.message.receive_v1")
    assert len(claim_token) == 64


def test_new_router_receives_sanitized_reference_without_message_body() -> None:
    router = _RecordingSanitizedRouter()
    client, _ = _client(router=router)  # type: ignore[arg-type]
    body = _event_body(event_id="evt-sanitized-001")

    response = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert response.status_code == 200
    assert len(router.calls) == 1
    reference, claim_token = router.calls[0]
    assert reference.message_id == "om-source"
    assert reference.chat_id == "oc-chat"
    assert len(claim_token) == 64
    assert "must-not-be-forwarded" not in repr(reference)


def test_in_progress_delivery_returns_retryable_error_instead_of_false_ack() -> None:
    router = _RecordingRouter()
    client, receipts = _client(router=router)
    body = _event_body(event_id="evt-in-progress-001")
    digest = hashlib.sha256(body).hexdigest()
    receipts.identities["evt-in-progress-001"] = ("im.message.receive_v1", digest)
    receipts.statuses["evt-in-progress-001"] = FeishuReceiptClaimStatus.IN_PROGRESS

    response = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "feishu_event_in_progress"}
    assert router.calls == []


def test_invalid_signature_token_and_json_return_secret_free_errors() -> None:
    client, _ = _client(router=_RecordingRouter())
    body = _event_body()

    invalid_signature = client.post(
        "/v1/integrations/feishu/events",
        headers=_signed_headers(body, signature="0" * 64),
        content=body,
    )
    invalid_json = client.post(
        "/v1/integrations/feishu/events",
        headers={"Content-Type": "application/json"},
        content=b"{not-json",
    )
    wrong_token_body = _event_body(token="wrong-callback-token")
    invalid_token = client.post(
        "/v1/integrations/feishu/events",
        headers=_signed_headers(wrong_token_body),
        content=wrong_token_body,
    )

    assert invalid_signature.status_code == 401
    assert invalid_signature.json() == {"detail": "feishu_callback_authentication_failed"}
    assert invalid_json.status_code == 422
    assert invalid_json.json() == {"detail": "invalid_feishu_callback"}
    assert invalid_token.status_code == 422
    assert invalid_token.json() == {"detail": "invalid_feishu_callback"}
    combined = invalid_signature.text + invalid_json.text + invalid_token.text
    assert TOKEN not in combined
    assert ENCRYPT_KEY not in combined
    assert "must-not-be-forwarded" not in combined


def test_invalid_callback_logs_only_a_safe_reason_code(
    caplog,  # type: ignore[no-untyped-def]
) -> None:
    client, _ = _client(router=_RecordingRouter())
    body = _event_body(token="wrong-callback-token")

    with caplog.at_level(logging.WARNING, logger="quanxin_life.api.feishu"):
        response = client.post(
            "/v1/integrations/feishu/events",
            headers=_signed_headers(body),
            content=body,
        )

    assert response.status_code == 422
    assert "reason=verification_token_mismatch" in caplog.text
    assert "error_type=FeishuEventError" in caplog.text
    assert "detail=Feishu verification token mismatch" in caplog.text
    assert TOKEN not in caplog.text
    assert ENCRYPT_KEY not in caplog.text


def test_unsafe_event_reference_is_rejected_before_receipt_claim() -> None:
    client, receipts = _client(router=_RecordingRouter())
    payload = json.loads(_event_body(event_id="evt-unsafe-reference"))
    payload["event"]["message"]["message_id"] = "../unsafe"
    body = json.dumps(payload, separators=(",", ":")).encode()

    response = client.post(
        "/v1/integrations/feishu/events",
        headers=_signed_headers(body),
        content=body,
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_feishu_callback"}
    assert receipts.identities == {}


def test_encrypted_payload_without_reviewed_decryptor_returns_503() -> None:
    client, _ = _client(router=_RecordingRouter())
    body = json.dumps({"encrypt": "ciphertext-must-not-leak"}).encode()

    response = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "feishu_callback_decryption_unavailable"}
    assert "ciphertext-must-not-leak" not in response.text


def test_callback_body_larger_than_one_mib_is_rejected_before_parsing() -> None:
    client, _ = _client(router=_RecordingRouter())

    response = client.post(
        "/v1/integrations/feishu/events",
        headers={"Content-Type": "application/octet-stream"},
        content=b"x" * (MAX_FEISHU_CALLBACK_BYTES + 1),
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "feishu_callback_too_large"}


def test_obviously_oversized_content_length_is_rejected_before_body_read() -> None:
    client, _ = _client(router=_RecordingRouter())

    response = client.post(
        "/v1/integrations/feishu/events",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(MAX_FEISHU_CALLBACK_BYTES + 1),
        },
        content=b"{}",
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "feishu_callback_too_large"}


def test_router_failure_is_released_for_retry_and_never_leaks_exception() -> None:
    router = _RecordingRouter(fail_first=True)
    client, receipts = _client(router=router)
    body = _event_body(event_id="evt-retry-001")

    failed = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )
    retried = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert failed.status_code == 503
    assert failed.json() == {"detail": "feishu_event_routing_failed"}
    assert "sensitive downstream detail" not in failed.text
    assert retried.status_code == 200, retried.text
    assert len(router.calls) == 2
    assert receipts.statuses["evt-retry-001"] is FeishuReceiptClaimStatus.PROCESSED


def test_router_must_confirm_durable_idempotent_enqueue() -> None:
    client, receipts = _client(router=_NoConfirmationRouter())  # type: ignore[arg-type]
    body = _event_body(event_id="evt-unconfirmed-001")

    response = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "feishu_event_routing_failed"}
    assert receipts.statuses["evt-unconfirmed-001"] is FeishuReceiptClaimStatus.RETRYABLE


def test_completion_failure_retries_idempotent_route_without_creating_second_task() -> None:
    router = _RecordingRouter()
    client, receipts = _client(router=router, fail_first_completion=True)
    body = _event_body(event_id="evt-completion-retry-001")

    failed_completion = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )
    retried = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert failed_completion.status_code == 503
    assert retried.status_code == 200, retried.text
    assert len(router.calls) == 2
    assert router.created_task_events == {"evt-completion-retry-001"}
    assert receipts.statuses["evt-completion-retry-001"] is FeishuReceiptClaimStatus.PROCESSED


def test_missing_router_releases_claim_and_returns_explicit_503() -> None:
    client, receipts = _client()
    body = _event_body(event_id="evt-no-router-001")

    response = client.post(
        "/v1/integrations/feishu/events", headers=_signed_headers(body), content=body
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "feishu_event_router_unavailable"}
    assert receipts.statuses["evt-no-router-001"] is FeishuReceiptClaimStatus.RETRYABLE
