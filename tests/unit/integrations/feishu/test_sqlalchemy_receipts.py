from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from quanxin_life.integrations.feishu.events import (
    FeishuReceiptClaim,
    FeishuReceiptClaimStatus,
)
from quanxin_life.integrations.feishu.sqlalchemy_receipts import (
    FeishuReceiptOwnershipError,
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.persistence.database import create_session_factory
from quanxin_life.persistence.models import Base, FeishuEventReceipt

NOW = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)


@pytest.fixture
def store() -> SqlAlchemyFeishuReceiptStore:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SqlAlchemyFeishuReceiptStore(create_session_factory(engine))


def _claim(
    store: SqlAlchemyFeishuReceiptStore,
    *,
    event_type: str = "im.message.receive_v1",
    payload_sha256: str = "a" * 64,
) -> FeishuReceiptClaim:
    return store.claim(
        event_id="evt-001",
        event_type=event_type,
        payload_sha256=payload_sha256,
        received_at=NOW,
    )


def test_new_claim_is_in_progress_and_same_replay_is_not_reprocessed(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    replay = _claim(store)

    assert first.status is FeishuReceiptClaimStatus.NEW
    assert first.claim_token is not None
    assert replay.status is FeishuReceiptClaimStatus.IN_PROGRESS
    assert replay.claim_token is None


def test_processed_claim_is_reported_as_processed(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    store.mark_processed(
        event_id="evt-001",
        claim_token=first.claim_token,
        processed_at=NOW + timedelta(seconds=1),
    )

    assert _claim(store).status is FeishuReceiptClaimStatus.PROCESSED


def test_failed_claim_can_be_acquired_once_for_retry(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    store.mark_failed(
        event_id="evt-001",
        claim_token=first.claim_token,
        failed_at=NOW + timedelta(seconds=1),
    )

    retry = _claim(store)
    concurrent = _claim(store)
    assert retry.status is FeishuReceiptClaimStatus.RETRYABLE
    assert retry.claim_token is not None
    assert retry.claim_token != first.claim_token
    assert concurrent.status is FeishuReceiptClaimStatus.IN_PROGRESS
    assert concurrent.claim_token is None


@pytest.mark.parametrize(
    ("event_type", "payload_sha256"),
    [
        ("different.event", "a" * 64),
        ("im.message.receive_v1", "b" * 64),
    ],
)
def test_reused_event_id_with_different_identity_is_conflict(
    store: SqlAlchemyFeishuReceiptStore,
    event_type: str,
    payload_sha256: str,
) -> None:
    assert _claim(store).status is FeishuReceiptClaimStatus.NEW

    assert (
        _claim(store, event_type=event_type, payload_sha256=payload_sha256).status
        is FeishuReceiptClaimStatus.CONFLICT
    )


def test_marking_unknown_receipt_fails_explicitly(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        store.mark_failed(event_id="unknown", claim_token="a" * 64, failed_at=NOW)


def test_store_persists_no_callback_payload(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    assert _claim(store).status is FeishuReceiptClaimStatus.NEW
    with store.session_factory() as session:
        row = session.query(FeishuEventReceipt).one()

    assert row.event_id == "evt-001"
    assert row.payload_sha256 == "a" * 64
    assert not hasattr(row, "payload")


def test_expired_in_progress_lease_is_recovered_with_a_new_token(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)

    recovered = store.claim(
        event_id="evt-001",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW + timedelta(minutes=6),
    )

    assert recovered.status is FeishuReceiptClaimStatus.RETRYABLE
    assert recovered.claim_token is not None
    assert recovered.claim_token != first.claim_token
    assert recovered.attempt == 2


def test_late_failure_from_old_claim_cannot_overwrite_successful_retry(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    retry = store.claim(
        event_id="evt-001",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW + timedelta(minutes=6),
    )
    store.mark_processed(
        event_id="evt-001",
        claim_token=retry.claim_token,
        processed_at=NOW + timedelta(minutes=6, seconds=1),
    )

    with pytest.raises(FeishuReceiptOwnershipError, match="no longer owns"):
        store.mark_failed(
            event_id="evt-001",
            claim_token=first.claim_token,
            failed_at=NOW + timedelta(minutes=6, seconds=2),
        )
    assert _claim(store).status is FeishuReceiptClaimStatus.PROCESSED


def test_late_success_from_old_claim_cannot_finish_a_new_attempt(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    retry = store.claim(
        event_id="evt-001",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW + timedelta(minutes=6),
    )

    with pytest.raises(FeishuReceiptOwnershipError, match="no longer owns"):
        store.mark_processed(
            event_id="evt-001",
            claim_token=first.claim_token,
            processed_at=NOW + timedelta(minutes=6, seconds=1),
        )
    store.mark_processed(
        event_id="evt-001",
        claim_token=retry.claim_token,
        processed_at=NOW + timedelta(minutes=6, seconds=2),
    )


def test_current_owner_can_renew_a_live_lease(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)

    renewed_until = store.renew_claim(
        event_id="evt-001",
        claim_token=first.claim_token,
        renewed_at=NOW + timedelta(minutes=4),
    )
    still_running = store.claim(
        event_id="evt-001",
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        received_at=NOW + timedelta(minutes=6),
    )

    assert renewed_until == NOW + timedelta(minutes=9)
    assert still_running.status is FeishuReceiptClaimStatus.IN_PROGRESS


def test_expired_owner_cannot_renew_after_its_lease(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)

    with pytest.raises(FeishuReceiptOwnershipError, match="no longer owns"):
        store.renew_claim(
            event_id="evt-001",
            claim_token=first.claim_token,
            renewed_at=NOW + timedelta(minutes=6),
        )


@pytest.mark.parametrize("completion", ["processed", "failed"])
def test_expired_owner_cannot_complete_or_fail_its_claim(
    store: SqlAlchemyFeishuReceiptStore,
    completion: str,
) -> None:
    first = _claim(store)
    expired_at = NOW + timedelta(minutes=6)

    with pytest.raises(FeishuReceiptOwnershipError, match="no longer owns"):
        if completion == "processed":
            store.mark_processed(
                event_id="evt-001",
                claim_token=first.claim_token,
                processed_at=expired_at,
            )
        else:
            store.mark_failed(
                event_id="evt-001",
                claim_token=first.claim_token,
                failed_at=expired_at,
            )


@pytest.mark.parametrize("completion", ["processed", "failed"])
def test_live_owner_can_complete_or_fail_its_claim(
    store: SqlAlchemyFeishuReceiptStore,
    completion: str,
) -> None:
    first = _claim(store)
    completed_at = NOW + timedelta(minutes=4)

    if completion == "processed":
        store.mark_processed(
            event_id="evt-001",
            claim_token=first.claim_token,
            processed_at=completed_at,
        )
        expected = FeishuReceiptClaimStatus.PROCESSED
    else:
        store.mark_failed(
            event_id="evt-001",
            claim_token=first.claim_token,
            failed_at=completed_at,
        )
        expected = FeishuReceiptClaimStatus.RETRYABLE

    with store.session_factory() as session:
        row = session.query(FeishuEventReceipt).one()
    assert row.status == expected.value


def test_attempt_count_and_failed_timestamp_are_persisted(
    store: SqlAlchemyFeishuReceiptStore,
) -> None:
    first = _claim(store)
    failed_at = NOW + timedelta(seconds=1)
    store.mark_failed(
        event_id="evt-001", claim_token=first.claim_token, failed_at=failed_at
    )
    retry = _claim(store)

    with store.session_factory() as session:
        row = session.query(FeishuEventReceipt).one()

    assert retry.attempt == 2
    assert row.attempt_count == 2
    assert row.failed_at is not None
    assert row.claim_token == retry.claim_token
    assert row.lease_expires_at is not None


def test_two_sqlite_sessions_cannot_both_acquire_the_same_retry(
    tmp_path: Path,
) -> None:
    """SQLite smoke test; PostgreSQL row-lock behavior needs its integration suite."""

    database = tmp_path / "feishu-receipts.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{database}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    store = SqlAlchemyFeishuReceiptStore(create_session_factory(engine))
    first = _claim(store)
    store.mark_failed(event_id="evt-001", claim_token=first.claim_token, failed_at=NOW)

    def acquire() -> FeishuReceiptClaim:
        return _claim(store)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: acquire(), range(2)))

    assert sorted(result.status for result in results) == [
        FeishuReceiptClaimStatus.IN_PROGRESS,
        FeishuReceiptClaimStatus.RETRYABLE,
    ]
    assert len([result for result in results if result.claim_token is not None]) == 1
