"""Durable Feishu receipt claims with leases and fenced completion tokens."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.elements import ColumnElement

from quanxin_life.persistence.database import SessionFactory
from quanxin_life.persistence.models import FeishuEventReceipt

from .events import FeishuReceiptClaim, FeishuReceiptClaimStatus

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeishuReceiptOwnershipError(RuntimeError):
    """Raised when an expired or superseded worker tries to finish a claim."""


class SqlAlchemyFeishuReceiptStore:
    """Claim one event attempt at a time using atomic compare-and-set updates.

    PostgreSQL additionally takes a row lock while inspecting an existing
    receipt. The conditional UPDATE is still the correctness boundary, so the
    same state machine remains safe on databases where ``FOR UPDATE`` is a
    no-op (notably SQLite test databases).
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        lease_seconds: int = 300,
    ) -> None:
        if lease_seconds < 1 or lease_seconds > 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        self._session_factory = session_factory
        self._lease = timedelta(seconds=lease_seconds)

    @property
    def session_factory(self) -> SessionFactory:
        """Expose the configured factory for assembly and narrow diagnostics."""

        return self._session_factory

    def claim(
        self,
        *,
        event_id: str,
        event_type: str,
        payload_sha256: str,
        received_at: datetime,
    ) -> FeishuReceiptClaim:
        checked_event_id = _text(event_id, field="event_id", max_length=200)
        checked_event_type = _text(event_type, field="event_type", max_length=100)
        checked_hash = payload_sha256.strip().lower()
        if _SHA256.fullmatch(checked_hash) is None:
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
        now = _utc(received_at)

        for _ in range(4):
            session = self._session_factory()
            try:
                receipt = session.scalar(
                    select(FeishuEventReceipt)
                    .where(FeishuEventReceipt.event_id == checked_event_id)
                    .with_for_update()
                )
                if receipt is None:
                    token = _new_claim_token()
                    session.add(
                        FeishuEventReceipt(
                            id=uuid.uuid4().hex,
                            event_id=checked_event_id,
                            event_type=checked_event_type,
                            payload_sha256=checked_hash,
                            status=FeishuReceiptClaimStatus.IN_PROGRESS.value,
                            claim_token=token,
                            attempt_count=1,
                            lease_expires_at=now + self._lease,
                            received_at=now,
                            processed_at=None,
                            failed_at=None,
                        )
                    )
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        continue
                    return FeishuReceiptClaim(
                        status=FeishuReceiptClaimStatus.NEW,
                        claim_token=token,
                        attempt=1,
                    )

                if (
                    receipt.event_type != checked_event_type
                    or receipt.payload_sha256 != checked_hash
                ):
                    session.rollback()
                    return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)

                try:
                    status = FeishuReceiptClaimStatus(receipt.status)
                except ValueError:
                    session.rollback()
                    return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)

                if status is FeishuReceiptClaimStatus.PROCESSED:
                    session.rollback()
                    return FeishuReceiptClaim(status=status)

                if status is FeishuReceiptClaimStatus.IN_PROGRESS and not _lease_expired(
                    receipt.lease_expires_at, now=now
                ):
                    session.rollback()
                    return FeishuReceiptClaim(status=status)

                if status not in {
                    FeishuReceiptClaimStatus.IN_PROGRESS,
                    FeishuReceiptClaimStatus.RETRYABLE,
                }:
                    session.rollback()
                    return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.CONFLICT)

                token = _new_claim_token()
                next_attempt = receipt.attempt_count + 1
                fencing_predicate = _fencing_predicate(receipt, now=now)
                statement = (
                    update(FeishuEventReceipt)
                    .where(
                        FeishuEventReceipt.id == receipt.id,
                        FeishuEventReceipt.event_type == checked_event_type,
                        FeishuEventReceipt.payload_sha256 == checked_hash,
                        fencing_predicate,
                    )
                    .values(
                        status=FeishuReceiptClaimStatus.IN_PROGRESS.value,
                        claim_token=token,
                        attempt_count=FeishuEventReceipt.attempt_count + 1,
                        lease_expires_at=now + self._lease,
                    )
                )
                updated = cast(CursorResult[Any], session.execute(statement))
                if updated.rowcount != 1:
                    session.rollback()
                    continue
                session.commit()
                return FeishuReceiptClaim(
                    status=FeishuReceiptClaimStatus.RETRYABLE,
                    claim_token=token,
                    attempt=next_attempt,
                )
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        # Every failed CAS means another worker advanced the state. Re-read on
        # the next delivery rather than returning an unsafe ownership token.
        return FeishuReceiptClaim(status=FeishuReceiptClaimStatus.IN_PROGRESS)

    def mark_processed(
        self,
        *,
        event_id: str,
        claim_token: str,
        processed_at: datetime,
    ) -> None:
        checked_at = _utc(processed_at)
        self._complete_owned_claim(
            event_id=event_id,
            claim_token=claim_token,
            owned_at=checked_at,
            values={
                "status": FeishuReceiptClaimStatus.PROCESSED.value,
                "processed_at": checked_at,
                "claim_token": None,
                "lease_expires_at": None,
            },
        )

    def mark_failed(
        self,
        *,
        event_id: str,
        claim_token: str,
        failed_at: datetime,
    ) -> None:
        checked_at = _utc(failed_at)
        self._complete_owned_claim(
            event_id=event_id,
            claim_token=claim_token,
            owned_at=checked_at,
            values={
                "status": FeishuReceiptClaimStatus.RETRYABLE.value,
                "failed_at": checked_at,
                "claim_token": None,
                "lease_expires_at": None,
            },
        )

    def renew_claim(
        self,
        *,
        event_id: str,
        claim_token: str,
        renewed_at: datetime,
    ) -> datetime:
        checked_event_id = _text(event_id, field="event_id", max_length=200)
        checked_token = _claim_token(claim_token)
        checked_at = _utc(renewed_at)
        new_expiry = checked_at + self._lease
        session = self._session_factory()
        try:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(FeishuEventReceipt)
                    .where(
                        FeishuEventReceipt.event_id == checked_event_id,
                        FeishuEventReceipt.status
                        == FeishuReceiptClaimStatus.IN_PROGRESS.value,
                        FeishuEventReceipt.claim_token == checked_token,
                        FeishuEventReceipt.lease_expires_at > checked_at,
                    )
                    .values(lease_expires_at=new_expiry)
                ),
            )
            if result.rowcount != 1:
                session.rollback()
                raise FeishuReceiptOwnershipError(
                    f"Feishu claim no longer owns event: {checked_event_id}"
                )
            session.commit()
            return new_expiry
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _complete_owned_claim(
        self,
        *,
        event_id: str,
        claim_token: str,
        owned_at: datetime,
        values: dict[str, object],
    ) -> None:
        checked_event_id = _text(event_id, field="event_id", max_length=200)
        checked_token = _claim_token(claim_token)
        session = self._session_factory()
        try:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(FeishuEventReceipt)
                    .where(
                        FeishuEventReceipt.event_id == checked_event_id,
                        FeishuEventReceipt.status == FeishuReceiptClaimStatus.IN_PROGRESS.value,
                        FeishuEventReceipt.claim_token == checked_token,
                        FeishuEventReceipt.lease_expires_at.is_not(None),
                        FeishuEventReceipt.lease_expires_at > owned_at,
                    )
                    .values(**values)
                ),
            )
            if result.rowcount != 1:
                session.rollback()
                exists = session.scalar(
                    select(FeishuEventReceipt.id).where(
                        FeishuEventReceipt.event_id == checked_event_id
                    )
                )
                if exists is None:
                    raise ValueError(f"Feishu receipt not found: {checked_event_id}")
                raise FeishuReceiptOwnershipError(
                    f"Feishu claim no longer owns event: {checked_event_id}"
                )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _fencing_predicate(receipt: FeishuEventReceipt, *, now: datetime) -> ColumnElement[bool]:
    if receipt.status == FeishuReceiptClaimStatus.RETRYABLE.value:
        return and_(
            FeishuEventReceipt.status == FeishuReceiptClaimStatus.RETRYABLE.value,
            FeishuEventReceipt.claim_token.is_(None),
        )
    return and_(
        FeishuEventReceipt.status == FeishuReceiptClaimStatus.IN_PROGRESS.value,
        FeishuEventReceipt.claim_token == receipt.claim_token,
        or_(
            FeishuEventReceipt.lease_expires_at.is_(None),
            FeishuEventReceipt.lease_expires_at <= now,
        ),
    )


def _lease_expired(expires_at: datetime | None, *, now: datetime) -> bool:
    if expires_at is None:
        return True
    normalized = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
    return normalized.astimezone(UTC) <= now


def _new_claim_token() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex


def _claim_token(value: str) -> str:
    checked = value.strip().lower()
    if _SHA256.fullmatch(checked) is None:
        raise ValueError("claim_token must be a 64-character hexadecimal token")
    return checked


def _text(value: str, *, field: str, max_length: int) -> str:
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field} must not be blank")
    if len(checked) > max_length:
        raise ValueError(f"{field} exceeds {max_length} characters")
    return checked


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Feishu receipt timestamps must include a timezone")
    return value.astimezone(UTC)


__all__ = ["FeishuReceiptOwnershipError", "SqlAlchemyFeishuReceiptStore"]
