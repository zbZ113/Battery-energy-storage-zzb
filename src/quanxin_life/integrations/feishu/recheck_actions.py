"""Authorized, idempotent Feishu recheck actions backed by Bitable."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from quanxin_life.core import Decision, sha256_canonical
from quanxin_life.integrations.feishu.bitable import (
    BitableConflictError,
    BitableProtocolError,
    BitableValidationError,
    FeishuBitableClient,
)
from quanxin_life.integrations.feishu.events import (
    FeishuReceiptClaimStatus,
    FeishuReceiptStore,
)

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_INITIAL_STATUS = "PENDING"
_RECHECK_EVENT_TYPE = "quanxin_life.recheck_action.v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RecheckActionAuthorizationError(RuntimeError):
    """Raised when live run/result binding or Feishu permission is not verified."""


class RecheckActionInProgressError(RuntimeError):
    """Raised when another worker owns the same durable action claim."""


class RecheckActionAuthorizationVerifier(Protocol):
    """Verify run/result binding, responsibility, and live Feishu permission."""

    def verify_recheck_action(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        action_type: Decision,
        responsibility_reference: str,
        permission_reference: str,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecheckActionReceipt:
    """Remote record identity and the exact non-sensitive action fields."""

    action_key: str
    record_id: str
    source_run_id: str
    source_result_id: str
    action_type: Decision
    responsibility_reference: str
    permission_reference: str
    status: str
    created_at: datetime
    updated_at: datetime


class FeishuRecheckActionService:
    """Create one Bitable recheck row after explicit live authorization."""

    def __init__(
        self,
        *,
        client: FeishuBitableClient,
        app_token: str,
        table_id: str,
        authorization_verifier: RecheckActionAuthorizationVerifier,
        receipt_store: FeishuReceiptStore,
        clock: Clock = _utc_now,
    ) -> None:
        if not callable(getattr(client, "search_bitable_records", None)):
            raise TypeError("client must support Bitable record search")
        if not callable(getattr(client, "create_bitable_record", None)):
            raise TypeError("client must support Bitable record creation")
        if not callable(getattr(authorization_verifier, "verify_recheck_action", None)):
            raise TypeError("authorization_verifier must verify recheck actions")
        for method_name in ("claim", "mark_processed", "mark_failed"):
            if not callable(getattr(receipt_store, method_name, None)):
                raise TypeError("receipt_store must persist recheck action claims")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._client = client
        self._app_token = _safe_identifier(app_token, field_name="app_token")
        self._table_id = _safe_identifier(table_id, field_name="table_id")
        self._authorization_verifier = authorization_verifier
        self._receipt_store = receipt_store
        self._clock = clock
        self._lock = threading.RLock()

    def create_recheck_action(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        responsibility_reference: str,
        permission_reference: str,
    ) -> RecheckActionReceipt:
        """Create once or return the identical authorized remote action record."""

        checked_run_id = _uuid_reference(source_run_id, field_name="source_run_id")
        checked_result_id = _uuid_reference(source_result_id, field_name="source_result_id")
        checked_responsibility = _safe_identifier(
            responsibility_reference,
            field_name="responsibility_reference",
        )
        checked_permission = _safe_identifier(
            permission_reference,
            field_name="permission_reference",
        )
        action_type = Decision.RECHECK
        self._verify_authorization(
            source_run_id=checked_run_id,
            source_result_id=checked_result_id,
            action_type=action_type,
            responsibility_reference=checked_responsibility,
            permission_reference=checked_permission,
        )
        now = _utc_timestamp(self._clock())
        action_key = derive_recheck_action_key(checked_run_id, checked_result_id)
        expected_fields = {
            "action_key": action_key,
            "source_run_id": checked_run_id,
            "source_result_id": checked_result_id,
            "action_type": action_type.value,
            "responsibility_reference": checked_responsibility,
            "permission_reference": checked_permission,
            "status": _INITIAL_STATUS,
            "created_at_utc": now.isoformat(),
            "updated_at_utc": now.isoformat(),
        }
        event_id = f"recheck:{action_key}"
        request_hash = sha256_canonical(
            {
                "action_key": action_key,
                "action_type": action_type.value,
                "permission_reference": checked_permission,
                "responsibility_reference": checked_responsibility,
                "source_result_id": checked_result_id,
                "source_run_id": checked_run_id,
            }
        )
        claim = self._receipt_store.claim(
            event_id=event_id,
            event_type=_RECHECK_EVENT_TYPE,
            payload_sha256=request_hash,
            received_at=now,
        )
        if claim.status is FeishuReceiptClaimStatus.CONFLICT:
            raise BitableConflictError(
                "recheck action identity is already bound to different responsibility "
                "or permission evidence"
            )
        if claim.status is FeishuReceiptClaimStatus.IN_PROGRESS:
            raise RecheckActionInProgressError("recheck action is already in progress")
        if claim.status is FeishuReceiptClaimStatus.PROCESSED:
            with self._lock:
                return self._restore_processed_action(
                    action_key=action_key,
                    expected_fields=expected_fields,
                    action_type=action_type,
                )
        claim_token = claim.claim_token
        if claim_token is None:  # pragma: no cover - FeishuReceiptClaim invariant
            raise RecheckActionInProgressError("recheck action claim has no ownership token")

        try:
            with self._lock:
                receipt = self._create_or_restore_action(
                    action_key=action_key,
                    expected_fields=expected_fields,
                    action_type=action_type,
                    created_at=now,
                )
        except Exception:
            self._receipt_store.mark_failed(
                event_id=event_id,
                claim_token=claim_token,
                failed_at=now,
            )
            raise
        self._receipt_store.mark_processed(
            event_id=event_id,
            claim_token=claim_token,
            processed_at=now,
        )
        return receipt

    def _create_or_restore_action(
        self,
        *,
        action_key: str,
        expected_fields: Mapping[str, str],
        action_type: Decision,
        created_at: datetime,
    ) -> RecheckActionReceipt:
        items = self._search(action_key)
        if items:
            return _existing_receipt(
                items[0],
                expected_fields=expected_fields,
                action_type=action_type,
            )
        response = self._client.create_bitable_record(
            app_token=self._app_token,
            table_id=self._table_id,
            fields=expected_fields,
        )
        return RecheckActionReceipt(
            action_key=action_key,
            record_id=_record_response_id(response),
            source_run_id=expected_fields["source_run_id"],
            source_result_id=expected_fields["source_result_id"],
            action_type=action_type,
            responsibility_reference=expected_fields["responsibility_reference"],
            permission_reference=expected_fields["permission_reference"],
            status=_INITIAL_STATUS,
            created_at=created_at,
            updated_at=created_at,
        )

    def _restore_processed_action(
        self,
        *,
        action_key: str,
        expected_fields: Mapping[str, str],
        action_type: Decision,
    ) -> RecheckActionReceipt:
        items = self._search(action_key)
        if not items:
            raise BitableProtocolError(
                "processed recheck receipt has no matching Bitable record"
            )
        return _existing_receipt(
            items[0],
            expected_fields=expected_fields,
            action_type=action_type,
        )

    def _search(self, action_key: str) -> list[Mapping[str, object]]:
        search = self._client.search_bitable_records(
            app_token=self._app_token,
            table_id=self._table_id,
            field_name="action_key",
            field_value=action_key,
        )
        items = _search_items(search)
        if len(items) > 1:
            raise BitableConflictError(
                "action_key matched multiple records; uniqueness is not proven"
            )
        return items

    def _verify_authorization(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        action_type: Decision,
        responsibility_reference: str,
        permission_reference: str,
    ) -> None:
        try:
            authorized = self._authorization_verifier.verify_recheck_action(
                source_run_id=source_run_id,
                source_result_id=source_result_id,
                action_type=action_type,
                responsibility_reference=responsibility_reference,
                permission_reference=permission_reference,
            )
        except Exception as exc:
            raise RecheckActionAuthorizationError(
                "recheck action authorization verification failed"
            ) from exc
        if not isinstance(authorized, bool):
            raise RecheckActionAuthorizationError(
                "recheck action authorization result is invalid"
            )
        if not authorized:
            raise RecheckActionAuthorizationError("recheck action is not authorized")


def derive_recheck_action_key(source_run_id: str, source_result_id: str) -> str:
    """Hash source identity plus the fixed existing ``Decision.RECHECK`` action type."""

    checked_run_id = _uuid_reference(source_run_id, field_name="source_run_id")
    checked_result_id = _uuid_reference(source_result_id, field_name="source_result_id")
    return sha256_canonical(
        {
            "action_type": Decision.RECHECK.value,
            "source_result_id": checked_result_id,
            "source_run_id": checked_run_id,
        }
    )


def _existing_receipt(
    item: Mapping[str, object],
    *,
    expected_fields: Mapping[str, str],
    action_type: Decision,
) -> RecheckActionReceipt:
    record_id = _record_id(item, response_label="search response")
    stored_fields = item.get("fields")
    if not isinstance(stored_fields, Mapping):
        raise BitableProtocolError("Bitable search response record fields are invalid")
    identity_fields = (
        "action_key",
        "source_run_id",
        "source_result_id",
        "action_type",
        "responsibility_reference",
        "permission_reference",
    )
    for field_name in identity_fields:
        if stored_fields.get(field_name) != expected_fields[field_name]:
            raise BitableConflictError(
                f"stored recheck action {field_name} conflicts with the requested action"
            )
    status = _safe_stored_text(stored_fields.get("status"), field_name="status")
    created_at = _stored_timestamp(
        stored_fields.get("created_at_utc"),
        field_name="created_at_utc",
    )
    updated_at = _stored_timestamp(
        stored_fields.get("updated_at_utc"),
        field_name="updated_at_utc",
    )
    if created_at > updated_at:
        raise BitableProtocolError("stored recheck action timestamps are inconsistent")
    return RecheckActionReceipt(
        action_key=expected_fields["action_key"],
        record_id=record_id,
        source_run_id=expected_fields["source_run_id"],
        source_result_id=expected_fields["source_result_id"],
        action_type=action_type,
        responsibility_reference=expected_fields["responsibility_reference"],
        permission_reference=expected_fields["permission_reference"],
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def _search_items(response: object) -> list[Mapping[str, object]]:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable search response is invalid")
    has_more = response.get("has_more")
    if has_more is True:
        raise BitableConflictError(
            "Bitable search has additional records; uniqueness is not proven"
        )
    if has_more not in {None, False}:
        raise BitableProtocolError("Bitable search pagination state is invalid")
    items = response.get("items")
    if not isinstance(items, list):
        raise BitableProtocolError("Bitable search response has no item list")
    if any(not isinstance(item, Mapping) for item in items):
        raise BitableProtocolError("Bitable search response contains an invalid record")
    return items


def _record_response_id(response: object) -> str:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable record response is invalid")
    return _record_id(response.get("record"), response_label="record response")


def _record_id(value: object, *, response_label: str) -> str:
    if not isinstance(value, Mapping):
        raise BitableProtocolError(f"Bitable {response_label} is invalid")
    record_id = value.get("record_id")
    if not isinstance(record_id, str) or _SAFE_IDENTIFIER.fullmatch(record_id) is None:
        raise BitableProtocolError(f"Bitable {response_label} has no safe record_id")
    return record_id


def _uuid_reference(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BitableValidationError(f"{field_name} must be a UUID string")
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise BitableValidationError(f"{field_name} must be a UUID string") from exc


def _safe_identifier(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_IDENTIFIER.fullmatch(normalized) is None:
        raise BitableValidationError(f"{field_name} must be a safe identifier")
    return normalized


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise BitableValidationError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise BitableValidationError("clock datetime must include a timezone")
    return value.astimezone(UTC)


def _stored_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise BitableProtocolError(f"stored {field_name} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise BitableProtocolError(f"stored {field_name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BitableProtocolError(f"stored {field_name} has no timezone")
    return parsed.astimezone(UTC)


def _safe_stored_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BitableProtocolError(f"stored {field_name} is invalid")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise BitableProtocolError(f"stored {field_name} is invalid")
    return normalized


__all__ = [
    "FeishuRecheckActionService",
    "RecheckActionAuthorizationError",
    "RecheckActionAuthorizationVerifier",
    "RecheckActionInProgressError",
    "RecheckActionReceipt",
    "derive_recheck_action_key",
]
