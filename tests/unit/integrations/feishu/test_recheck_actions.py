from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from quanxin_life.core import Decision, sha256_canonical
from quanxin_life.integrations.feishu.bitable import (
    BitableConflictError,
    BitableProtocolError,
    BitableValidationError,
)
from quanxin_life.integrations.feishu.events import (
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
)
from quanxin_life.integrations.feishu.recheck_actions import (
    CHINESE_RECHECK_ACTION_FIELD_PROFILE,
    FeishuRecheckActionService,
    RecheckActionAuthorizationError,
    RecheckActionInProgressError,
    derive_recheck_action_key,
)

_PLUS_EIGHT = timezone(timedelta(hours=8))


class _FakeBitableClient:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.searches: list[str] = []
        self.created = 0

    def search_bitable_records(
        self,
        *,
        app_token: str,
        table_id: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, object]:
        assert app_token == "app_actions"
        assert table_id == "tbl_rechecks"
        assert field_name == "action_key"
        self.searches.append(field_value)
        record = self.records.get(field_value)
        return {"items": [] if record is None else [record]}

    def create_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        assert app_token == "app_actions"
        assert table_id == "tbl_rechecks"
        self.created += 1
        record_id = f"rec_{self.created}"
        action_key = fields["action_key"]
        assert isinstance(action_key, str)
        self.records[action_key] = {
            "record_id": record_id,
            "fields": dict(fields),
        }
        return {"record": {"record_id": record_id}}


class _ReceiptStore:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()
        self.claims: list[tuple[str, str, str]] = []
        self.processed = 0
        self.failed = 0

    def claim(
        self,
        *,
        event_id: str,
        event_type: str,
        payload_sha256: str,
        received_at: datetime,
    ) -> FeishuReceiptClaim:
        assert received_at.tzinfo is not None
        with self._lock:
            self.claims.append((event_id, event_type, payload_sha256))
            row = self._rows.get(event_id)
            if row is None:
                token = "a" * 64
                self._rows[event_id] = {
                    "event_type": event_type,
                    "payload_sha256": payload_sha256,
                    "status": FeishuReceiptClaimStatus.IN_PROGRESS,
                    "claim_token": token,
                    "attempt": 1,
                }
                return FeishuReceiptClaim(
                    status=FeishuReceiptClaimStatus.NEW,
                    claim_token=token,
                    attempt=1,
                )
            if (
                row["event_type"] != event_type
                or row["payload_sha256"] != payload_sha256
            ):
                return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)
            status = row["status"]
            assert isinstance(status, FeishuReceiptClaimStatus)
            if status is FeishuReceiptClaimStatus.PROCESSED:
                return FeishuReceiptClaim(status=status)
            if status is FeishuReceiptClaimStatus.IN_PROGRESS:
                return FeishuReceiptClaim(status=status)
            assert status is FeishuReceiptClaimStatus.RETRYABLE
            previous_attempt = row["attempt"]
            assert isinstance(previous_attempt, int)
            attempt = previous_attempt + 1
            token = "b" * 64
            row.update(
                status=FeishuReceiptClaimStatus.IN_PROGRESS,
                claim_token=token,
                attempt=attempt,
            )
            return FeishuReceiptClaim(
                status=FeishuReceiptClaimStatus.RETRYABLE,
                claim_token=token,
                attempt=attempt,
            )

    def mark_processed(
        self,
        *,
        event_id: str,
        claim_token: str,
        processed_at: datetime,
    ) -> None:
        assert processed_at.tzinfo is not None
        with self._lock:
            row = self._owned_row(event_id, claim_token)
            row.update(
                status=FeishuReceiptClaimStatus.PROCESSED,
                claim_token=None,
            )
            self.processed += 1

    def mark_failed(
        self,
        *,
        event_id: str,
        claim_token: str,
        failed_at: datetime,
    ) -> None:
        assert failed_at.tzinfo is not None
        with self._lock:
            row = self._owned_row(event_id, claim_token)
            row.update(
                status=FeishuReceiptClaimStatus.RETRYABLE,
                claim_token=None,
            )
            self.failed += 1

    def renew_claim(
        self,
        *,
        event_id: str,
        claim_token: str,
        renewed_at: datetime,
    ) -> datetime:
        with self._lock:
            self._owned_row(event_id, claim_token)
        return renewed_at + timedelta(minutes=5)

    def _owned_row(self, event_id: str, claim_token: str) -> dict[str, object]:
        row = self._rows[event_id]
        assert row["status"] is FeishuReceiptClaimStatus.IN_PROGRESS
        assert row["claim_token"] == claim_token
        return row


class _AuthorizationVerifier:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[dict[str, object]] = []

    def verify_recheck_action(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        action_type: Decision,
        responsibility_reference: str,
        permission_reference: str,
    ) -> bool:
        self.calls.append(
            {
                "source_run_id": source_run_id,
                "source_result_id": source_result_id,
                "action_type": action_type,
                "responsibility_reference": responsibility_reference,
                "permission_reference": permission_reference,
            }
        )
        return self.allowed


def _service(
    client: object,
    verifier: object,
    *,
    receipts: object | None = None,
    clock: object | None = None,
    field_profile: object | None = None,
) -> FeishuRecheckActionService:
    kwargs: dict[str, object] = {}
    if clock is not None:
        kwargs["clock"] = clock
    if field_profile is not None:
        kwargs["field_profile"] = field_profile
    return FeishuRecheckActionService(
        client=client,  # type: ignore[arg-type]
        app_token="app_actions",
        table_id="tbl_rechecks",
        authorization_verifier=verifier,  # type: ignore[arg-type]
        receipt_store=receipts or _ReceiptStore(),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _references() -> tuple[str, str]:
    return str(uuid4()), str(uuid4())


def test_creates_once_and_returns_the_same_remote_recheck_record() -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    verifier = _AuthorizationVerifier()
    receipts = _ReceiptStore()
    service = _service(
        client,
        verifier,
        receipts=receipts,
        clock=lambda: datetime(2026, 8, 14, 9, 30, tzinfo=_PLUS_EIGHT),
    )

    first = service.create_recheck_action(
        source_run_id=source_run_id,
        source_result_id=source_result_id,
        responsibility_reference="ou_battery_owner",
        permission_reference="perm_recheck_live_v1",
    )
    second = service.create_recheck_action(
        source_run_id=source_run_id,
        source_result_id=source_result_id,
        responsibility_reference="ou_battery_owner",
        permission_reference="perm_recheck_live_v1",
    )

    expected_key = sha256_canonical(
        {
            "action_type": Decision.RECHECK.value,
            "source_result_id": source_result_id,
            "source_run_id": source_run_id,
        }
    )
    expected_request_hash = sha256_canonical(
        {
            "action_key": expected_key,
            "action_type": Decision.RECHECK.value,
            "permission_reference": "perm_recheck_live_v1",
            "responsibility_reference": "ou_battery_owner",
            "source_result_id": source_result_id,
            "source_run_id": source_run_id,
        }
    )
    assert derive_recheck_action_key(source_run_id, source_result_id) == expected_key
    assert first.action_key == second.action_key == expected_key
    assert first.record_id == second.record_id == "rec_1"
    assert first.action_type is second.action_type is Decision.RECHECK
    assert first.created_at == second.created_at == datetime(2026, 8, 14, 1, 30, tzinfo=UTC)
    assert first.updated_at == second.updated_at == datetime(2026, 8, 14, 1, 30, tzinfo=UTC)
    assert client.created == 1
    assert client.searches == [expected_key, expected_key]
    assert receipts.claims == [
        (
            f"recheck:{expected_key}",
            "quanxin_life.recheck_action.v1",
            expected_request_hash,
        ),
        (
            f"recheck:{expected_key}",
            "quanxin_life.recheck_action.v1",
            expected_request_hash,
        ),
    ]
    assert receipts.processed == 1
    assert receipts.failed == 0
    assert len(verifier.calls) == 2
    assert verifier.calls[0]["source_run_id"] == source_run_id
    assert verifier.calls[0]["source_result_id"] == source_result_id
    assert verifier.calls[0]["action_type"] is Decision.RECHECK
    stored = client.records[expected_key]["fields"]
    assert isinstance(stored, dict)
    assert stored == {
        "action_key": expected_key,
        "source_run_id": source_run_id,
        "source_result_id": source_result_id,
        "action_type": Decision.RECHECK.value,
        "responsibility_reference": "ou_battery_owner",
        "permission_reference": "perm_recheck_live_v1",
        "status": "PENDING",
        "created_at_utc": "2026-08-14T01:30:00+00:00",
        "updated_at_utc": "2026-08-14T01:30:00+00:00",
    }


def test_chinese_profile_writes_engineer_facing_recheck_fields() -> None:
    class ChineseClient:
        def __init__(self) -> None:
            self.search_field_name: str | None = None
            self.remote_fields: dict[str, object] = {}

        def search_bitable_records(self, **kwargs: object) -> dict[str, object]:
            self.search_field_name = str(kwargs["field_name"])
            return {"items": []}

        def create_bitable_record(self, **kwargs: object) -> dict[str, object]:
            fields = kwargs["fields"]
            assert isinstance(fields, Mapping)
            self.remote_fields = dict(fields)
            return {"record": {"record_id": "rec_chinese_recheck"}}

    source_run_id, source_result_id = _references()
    client = ChineseClient()
    receipt = _service(
        client,
        _AuthorizationVerifier(),
        field_profile=CHINESE_RECHECK_ACTION_FIELD_PROFILE,
    ).create_recheck_action(
        source_run_id=source_run_id,
        source_result_id=source_result_id,
        responsibility_reference="ou_battery_owner",
        permission_reference="perm_recheck_live_v1",
    )

    assert receipt.status == "PENDING"
    assert client.search_field_name == "复检建单键"
    assert client.remote_fields["来源任务ID"] == source_run_id
    assert client.remote_fields["来源建议结果ID"] == source_result_id
    assert client.remote_fields["动作类型"] == "发起复检"
    assert client.remote_fields["责任人"] == "ou_battery_owner"
    assert client.remote_fields["权限依据"] == "perm_recheck_live_v1"
    assert client.remote_fields["任务状态"] == "待处理"


def test_bitable_failure_releases_the_durable_claim_for_retry() -> None:
    class FailOnceClient(_FakeBitableClient):
        def create_bitable_record(
            self,
            *,
            app_token: str,
            table_id: str,
            fields: Mapping[str, object],
        ) -> dict[str, object]:
            if self.created == 0:
                self.created += 1
                raise RuntimeError("temporary Feishu failure")
            return super().create_bitable_record(
                app_token=app_token,
                table_id=table_id,
                fields=fields,
            )

    source_run_id, source_result_id = _references()
    client = FailOnceClient()
    receipts = _ReceiptStore()
    service = _service(client, _AuthorizationVerifier(), receipts=receipts)

    with pytest.raises(RuntimeError, match="temporary Feishu failure"):
        service.create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    restored = service.create_recheck_action(
        source_run_id=source_run_id,
        source_result_id=source_result_id,
        responsibility_reference="ou_battery_owner",
        permission_reference="perm_recheck_live_v1",
    )

    assert restored.record_id == "rec_2"
    assert receipts.failed == 1
    assert receipts.processed == 1
    assert len(receipts.claims) == 2


def test_shared_receipt_claim_prevents_two_workers_from_creating_the_same_action() -> None:
    class BlockingClient(_FakeBitableClient):
        def __init__(self) -> None:
            super().__init__()
            self.entered_create = threading.Event()
            self.release_create = threading.Event()

        def create_bitable_record(
            self,
            *,
            app_token: str,
            table_id: str,
            fields: Mapping[str, object],
        ) -> dict[str, object]:
            self.entered_create.set()
            assert self.release_create.wait(timeout=2)
            return super().create_bitable_record(
                app_token=app_token,
                table_id=table_id,
                fields=fields,
            )

    source_run_id, source_result_id = _references()
    client = BlockingClient()
    receipts = _ReceiptStore()
    verifier = _AuthorizationVerifier()
    first_worker = _service(client, verifier, receipts=receipts)
    second_worker = _service(client, verifier, receipts=receipts)
    request = {
        "source_run_id": source_run_id,
        "source_result_id": source_result_id,
        "responsibility_reference": "ou_battery_owner",
        "permission_reference": "perm_recheck_live_v1",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(first_worker.create_recheck_action, **request)
        assert client.entered_create.wait(timeout=2)
        with pytest.raises(RecheckActionInProgressError, match="in progress"):
            second_worker.create_recheck_action(**request)
        client.release_create.set()
        created = owner.result(timeout=2)

    restored = second_worker.create_recheck_action(**request)
    assert created.record_id == restored.record_id == "rec_1"
    assert client.created == 1
    assert receipts.processed == 1


@pytest.mark.parametrize(
    ("responsibility_reference", "permission_reference", "message"),
    [
        ("", "perm_recheck_live_v1", "responsibility_reference"),
        ("ou_battery_owner", "", "permission_reference"),
        (None, "perm_recheck_live_v1", "responsibility_reference"),
        ("ou_battery_owner", None, "permission_reference"),
    ],
)
def test_rejects_missing_responsibility_or_permission_before_any_feishu_call(
    responsibility_reference: object,
    permission_reference: object,
    message: str,
) -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    verifier = _AuthorizationVerifier()

    with pytest.raises(BitableValidationError, match=message):
        _service(client, verifier).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference=responsibility_reference,  # type: ignore[arg-type]
            permission_reference=permission_reference,  # type: ignore[arg-type]
        )

    assert verifier.calls == []
    assert client.searches == []
    assert client.created == 0


def test_rejects_denied_or_invalid_authorization_before_any_feishu_call() -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    denied = _AuthorizationVerifier(allowed=False)

    with pytest.raises(RecheckActionAuthorizationError, match="not authorized"):
        _service(client, denied).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    class InvalidVerifier(_AuthorizationVerifier):
        def verify_recheck_action(self, **kwargs: object) -> bool:
            super().verify_recheck_action(**kwargs)  # type: ignore[arg-type]
            return "yes"  # type: ignore[return-value]

    with pytest.raises(RecheckActionAuthorizationError, match="invalid"):
        _service(client, InvalidVerifier()).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    assert client.searches == []
    assert client.created == 0


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("responsibility_reference", "ou_other_owner"),
        ("permission_reference", "perm_recheck_live_v2"),
    ],
)
def test_same_action_key_rejects_changed_responsibility_or_permission(
    changed_field: str,
    changed_value: str,
) -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    service = _service(client, _AuthorizationVerifier())
    original = {
        "source_run_id": source_run_id,
        "source_result_id": source_result_id,
        "responsibility_reference": "ou_battery_owner",
        "permission_reference": "perm_recheck_live_v1",
    }
    service.create_recheck_action(**original)

    with pytest.raises(BitableConflictError, match="responsibility or permission"):
        service.create_recheck_action(**{**original, changed_field: changed_value})

    assert client.created == 1


def test_rejects_conflicting_stored_source_identity() -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    service = _service(client, _AuthorizationVerifier())
    receipt = service.create_recheck_action(
        source_run_id=source_run_id,
        source_result_id=source_result_id,
        responsibility_reference="ou_battery_owner",
        permission_reference="perm_recheck_live_v1",
    )
    stored = client.records[receipt.action_key]["fields"]
    assert isinstance(stored, dict)
    stored["source_result_id"] = str(uuid4())

    with pytest.raises(BitableConflictError, match="source_result_id"):
        service.create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    assert client.created == 1


@pytest.mark.parametrize("response_kind", ["duplicate", "paginated"])
def test_rejects_remote_uniqueness_uncertainty(response_kind: str) -> None:
    class UncertainClient(_FakeBitableClient):
        def search_bitable_records(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            if response_kind == "duplicate":
                return {
                    "items": [
                        {"record_id": "rec_1", "fields": {}},
                        {"record_id": "rec_2", "fields": {}},
                    ]
                }
            return {"items": [], "has_more": True}

    client = UncertainClient()
    source_run_id, source_result_id = _references()

    with pytest.raises(BitableConflictError):
        _service(client, _AuthorizationVerifier()).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    assert client.created == 0


def test_rejects_unsafe_identifiers_and_naive_clock_before_network_access() -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    verifier = _AuthorizationVerifier()

    with pytest.raises(BitableValidationError, match="source_run_id"):
        _service(client, verifier).create_recheck_action(
            source_run_id='run"forged',
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )
    with pytest.raises(BitableValidationError, match="responsibility_reference"):
        _service(client, verifier).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="owner with spaces",
            permission_reference="perm_recheck_live_v1",
        )
    with pytest.raises(BitableValidationError, match="timezone"):
        _service(
            client,
            verifier,
            clock=lambda: datetime(2026, 8, 14, 1, 30),
        ).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    assert client.searches == []
    assert client.created == 0


def test_rejects_malformed_existing_record_without_updating_or_recreating() -> None:
    source_run_id, source_result_id = _references()
    client = _FakeBitableClient()
    action_key = derive_recheck_action_key(source_run_id, source_result_id)
    client.records[action_key] = {
        "record_id": "rec_existing",
        "fields": "not-a-mapping",
    }

    with pytest.raises(BitableProtocolError, match="fields"):
        _service(client, _AuthorizationVerifier()).create_recheck_action(
            source_run_id=source_run_id,
            source_result_id=source_result_id,
            responsibility_reference="ou_battery_owner",
            permission_reference="perm_recheck_live_v1",
        )

    assert client.created == 0


def test_constructor_requires_an_explicit_authorization_verifier() -> None:
    with pytest.raises(TypeError, match="authorization_verifier"):
        _service(_FakeBitableClient(), None)
