from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.api.feishu_runner import (
    FeishuCallbackSecurityMode,
    create_feishu_callback_app,
)
from quanxin_life.integrations.feishu import (
    FeishuEventProcessor,
    FeishuEventReference,
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
)
from scripts.run_feishu_callback import build_local_app

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
TOKEN = "runner-verification-token"
ENCRYPT_KEY = "runner-encrypt-key"


class _MemoryReceiptStore:
    def __init__(self) -> None:
        self.statuses: dict[str, FeishuReceiptClaimStatus] = {}
        self.identities: dict[str, tuple[str, str]] = {}
        self.tokens: dict[str, str] = {}

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
        status = self.statuses.get(event_id)
        if status is FeishuReceiptClaimStatus.PROCESSED:
            return FeishuReceiptClaim(status=status)
        token = "a" * 64
        self.tokens[event_id] = token
        self.statuses[event_id] = FeishuReceiptClaimStatus.IN_PROGRESS
        return FeishuReceiptClaim(
            status=(
                FeishuReceiptClaimStatus.NEW
                if status is None
                else FeishuReceiptClaimStatus.RETRYABLE
            ),
            claim_token=token,
            attempt=1,
        )

    def mark_processed(
        self, *, event_id: str, claim_token: str, processed_at: datetime
    ) -> None:
        del processed_at
        assert self.tokens[event_id] == claim_token
        self.statuses[event_id] = FeishuReceiptClaimStatus.PROCESSED

    def mark_failed(
        self, *, event_id: str, claim_token: str, failed_at: datetime
    ) -> None:
        del failed_at
        assert self.tokens[event_id] == claim_token
        self.statuses[event_id] = FeishuReceiptClaimStatus.RETRYABLE

    def renew_claim(
        self, *, event_id: str, claim_token: str, renewed_at: datetime
    ) -> datetime:
        assert self.tokens[event_id] == claim_token
        return renewed_at


class _RecordingRouter:
    def __init__(self) -> None:
        self.events: list[FeishuEventReference] = []

    def route_event(
        self, *, event: FeishuEventReference, claim_token: str
    ) -> FeishuEventRouteStatus:
        assert len(claim_token) == 64
        self.events.append(event)
        return FeishuEventRouteStatus.ENQUEUED


def _event_body(*, event_id: str = "evt-runner-001", token: str = TOKEN) -> bytes:
    return json.dumps(
        {
            "schema": "2.0",
            "header": {
                "event_id": event_id,
                "event_type": "im.message.receive_v1",
                "token": token,
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou-runner"}},
                "message": {
                    "message_id": "om-runner",
                    "chat_id": "oc-runner",
                    "message_type": "text",
                    "content": '{"text":"must-not-enter-router"}',
                },
            },
        },
        separators=(",", ":"),
    ).encode()


def _file_event_body(*, event_id: str, token: str = TOKEN) -> bytes:
    payload = json.loads(_event_body(event_id=event_id, token=token))
    payload["event"]["message"]["message_type"] = "file"
    payload["event"]["message"]["content"] = json.dumps(
        {"file_key": "file-runner", "file_name": "runner.csv"}
    )
    return json.dumps(payload, separators=(",", ":")).encode()


def _headers(body: bytes) -> dict[str, str]:
    timestamp = str(int(NOW.timestamp()))
    nonce = "runner-nonce"
    signature = hashlib.sha256(
        timestamp.encode() + nonce.encode() + ENCRYPT_KEY.encode() + body
    ).hexdigest()
    return {
        "x-lark-request-timestamp": timestamp,
        "x-lark-request-nonce": nonce,
        "x-lark-signature": signature,
        "content-type": "application/json",
    }


def _encrypted_body(plaintext: bytes) -> bytes:
    key = hashlib.sha256(ENCRYPT_KEY.encode("utf-8")).digest()
    iv = b"i" * 16
    padding = 16 - (len(plaintext) % 16)
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ciphertext = iv + encryptor.update(plaintext + bytes([padding]) * padding)
    ciphertext += encryptor.finalize()
    return json.dumps(
        {"encrypt": base64.b64encode(ciphertext).decode("ascii")},
        separators=(",", ":"),
    ).encode()


def _app(*, mode: FeishuCallbackSecurityMode = FeishuCallbackSecurityMode.LOCAL_PLAINTEXT):
    receipts = _MemoryReceiptStore()
    processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(
            FeishuWebhookSecrets(
                verification_token=SecretStr(TOKEN),
                encrypt_key=SecretStr(ENCRYPT_KEY),
            )
        ),
        verification_token=SecretStr(TOKEN),
        receipts=receipts,
    )
    router = _RecordingRouter()
    app = create_feishu_callback_app(
        processor,
        router=router,
        security_mode=mode,
        now_factory=lambda: NOW,
    )
    return TestClient(app), router, receipts


def test_runner_exposes_health_and_existing_signed_callback_route() -> None:
    client, router, _ = _app()

    assert client.get("/health").json() == {
        "status": "ok",
        "service": "feishu-callback-runner",
        "security_mode": "LOCAL_PLAINTEXT",
    }
    body = _event_body()
    response = client.post("/v1/integrations/feishu/events", headers=_headers(body), content=body)

    assert response.status_code == 200
    assert len(router.events) == 1
    assert "must-not-enter-router" not in repr(router.events[0])


def test_runner_preserves_url_verification_contract() -> None:
    client, _, _ = _app()

    response = client.post(
        "/v1/integrations/feishu/events",
        json={"type": "url_verification", "token": TOKEN, "challenge": "runner-challenge"},
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "runner-challenge"}


def test_production_runner_rejects_plaintext_only_security_mode() -> None:
    with pytest.raises(ValueError, match="reviewed decryptor"):
        _app(mode=FeishuCallbackSecurityMode.PRODUCTION_REVIEWED)


def test_production_runner_rejects_a_declared_decryptor_when_processor_has_none() -> None:
    receipts = _MemoryReceiptStore()
    processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(
            FeishuWebhookSecrets(
                verification_token=SecretStr(TOKEN),
                encrypt_key=SecretStr(ENCRYPT_KEY),
            )
        ),
        verification_token=SecretStr(TOKEN),
        receipts=receipts,
    )

    with pytest.raises(ValueError, match="processor decryptor"):
        create_feishu_callback_app(
            processor,
            router=_RecordingRouter(),
            security_mode=FeishuCallbackSecurityMode.PRODUCTION_REVIEWED,
            reviewed_decryptor=object(),  # type: ignore[arg-type]
        )


def test_local_entrypoint_uses_sql_receipts_and_routes_one_sanitized_file_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    database_url = f"sqlite:///{(tmp_path / 'receipts.sqlite3').as_posix()}"
    app = build_local_app(database_url, now_factory=lambda: NOW)
    client = TestClient(app)
    body = _file_event_body(event_id="evt-runner-file-001")

    first = client.post(
        "/v1/integrations/feishu/events", headers=_headers(body), content=body
    )
    duplicate = client.post(
        "/v1/integrations/feishu/events", headers=_headers(body), content=body
    )
    wrong_token_body = _file_event_body(
        event_id="evt-runner-wrong-token", token="wrong-token"
    )
    wrong_token = client.post(
        "/v1/integrations/feishu/events",
        headers=_headers(wrong_token_body),
        content=wrong_token_body,
    )

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert wrong_token.status_code == 422
    sink = app.state.reference_sink
    assert sink.events.qsize() == 1
    event = sink.events.get_nowait()
    assert event.file_key == "file-runner"
    assert event.file_name == "runner.csv"
    assert event.message_id == "om-runner"
    assert "must-not-enter-router" not in repr(event)


def test_local_entrypoint_creates_a_missing_sqlite_parent_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    database_path = tmp_path / "missing" / "receipts.sqlite3"

    app = build_local_app(
        f"sqlite:///{database_path.as_posix()}", now_factory=lambda: NOW
    )

    assert database_path.is_file()
    assert TestClient(app).get("/health").status_code == 200


def test_local_entrypoint_decrypts_encrypted_url_verification(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    app = build_local_app(
        f"sqlite:///{(tmp_path / 'encrypted.sqlite3').as_posix()}",
        now_factory=lambda: NOW,
    )
    client = TestClient(app)
    plaintext = json.dumps(
        {
            "type": "url_verification",
            "token": TOKEN,
            "challenge": "encrypted-challenge",
        },
        separators=(",", ":"),
    ).encode()
    body = _encrypted_body(plaintext)

    response = client.post(
        "/v1/integrations/feishu/events", headers=_headers(body), content=body
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "encrypted-challenge"}
    assert client.get("/health").json()["security_mode"] == "LOCAL_ENCRYPTED"


def test_local_entrypoint_accepts_encrypted_url_verification_without_signature_headers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    app = build_local_app(
        f"sqlite:///{(tmp_path / 'encrypted-no-signature.sqlite3').as_posix()}"
    )
    client = TestClient(app)
    plaintext = json.dumps(
        {
            "type": "url_verification",
            "token": TOKEN,
            "challenge": "encrypted-no-signature-challenge",
        },
        separators=(",", ":"),
    ).encode()
    body = _encrypted_body(plaintext)

    response = client.post(
        "/v1/integrations/feishu/events",
        headers={"content-type": "application/json"},
        content=body,
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "encrypted-no-signature-challenge"}


def test_local_entrypoint_rejects_encrypted_business_event_without_signature_headers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    app = build_local_app(
        f"sqlite:///{(tmp_path / 'encrypted-event-no-signature.sqlite3').as_posix()}"
    )
    client = TestClient(app)
    body = _encrypted_body(_event_body(event_id="evt-no-signature"))

    response = client.post(
        "/v1/integrations/feishu/events",
        headers={"content-type": "application/json"},
        content=body,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "feishu_callback_authentication_failed"}
