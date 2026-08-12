from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

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
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt
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


class _PersistentQueue:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_ids.append(job_id)
        return FeishuJobDispatchReceipt(job_id=job_id, task_id="task-callback")


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


def _scenario_card_action_body(*, event_id: str, scenario_context_id: str) -> bytes:
    return json.dumps(
        {
            "schema": "2.0",
            "header": {
                "event_id": event_id,
                "event_type": "card.action.trigger",
                "token": TOKEN,
            },
            "event": {
                "operator": {"open_id": "ou-runner"},
                "context": {
                    "open_chat_id": "oc-runner",
                    "open_message_id": "om-scenario-runner",
                },
                "action": {
                    "value": {
                        "task_type": (
                            FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value
                        ),
                        "scenario_context_id": scenario_context_id,
                    }
                },
            },
        },
        separators=(",", ":"),
    ).encode()


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


def test_callback_persists_one_sanitized_job_and_acks_without_running_worker() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(session_factory)
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
    queue = _PersistentQueue()
    router = SqlAlchemyFeishuJobRouter(
        SqlAlchemyFeishuJobStore(session_factory),
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    )
    app = create_feishu_callback_app(
        processor,
        router=router,
        security_mode=FeishuCallbackSecurityMode.LOCAL_PLAINTEXT,
        now_factory=lambda: NOW,
    )
    client = TestClient(app)
    body = _file_event_body(event_id="evt-durable-job")

    first = client.post(
        "/v1/integrations/feishu/events", headers=_headers(body), content=body
    )
    duplicate = client.post(
        "/v1/integrations/feishu/events", headers=_headers(body), content=body
    )

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert len(queue.job_ids) == 1
    with session_factory() as session:
        assert session.scalar(select(func.count(FeishuEventReceipt.job_id))) == 1
        row = session.scalar(
            select(FeishuEventReceipt).where(
                FeishuEventReceipt.event_id == "evt-durable-job"
            )
        )
        assert row is not None
        assert row.message_id == "om-runner"
        assert row.file_key == "file-runner"
        assert row.job_stage == "RECEIVED"
        assert row.analysis_result_id is None


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


def test_local_entrypoint_persists_and_dispatches_one_sanitized_file_job(
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
    queue = app.state.job_queue
    assert queue.job_ids.qsize() == 1
    job = app.state.job_store.get(queue.job_ids.get_nowait())
    assert job.job_status is FeishuAnalysisJobStatus.PENDING
    assert job.task_type is FeishuAnalysisTask.PREDICT_CYCLE_LIFE
    assert job.file_key == "file-runner"
    assert job.file_name == "runner.csv"
    assert job.message_id == "om-runner"
    assert job.scenario_context_id is None
    assert "must-not-enter-router" not in repr(job)


def test_local_entrypoint_routes_scenario_card_by_context_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", TOKEN)
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", ENCRYPT_KEY)
    app = build_local_app(
        f"sqlite:///{(tmp_path / 'scenario.sqlite3').as_posix()}",
        now_factory=lambda: NOW,
    )
    context_id = "ea28ad37-d072-4d42-9d2a-fbc4ab5158db"
    body = _scenario_card_action_body(
        event_id="evt-runner-scenario-001",
        scenario_context_id=context_id,
    )

    response = TestClient(app).post(
        "/v1/integrations/feishu/events",
        headers=_headers(body),
        content=body,
    )

    assert response.status_code == 200
    job = app.state.job_store.get(app.state.job_queue.job_ids.get_nowait())
    assert job.task_type is FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS
    assert job.scenario_context_id == context_id
    assert job.file_key is None


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
