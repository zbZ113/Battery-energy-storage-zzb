"""Authorized, idempotent Feishu recheck actions backed by Bitable."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
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
RECHECK_ACTION_EVENT_TYPE = "quanxin_life.recheck_action.v1"
Clock = Callable[[], datetime]
_ACTION_FIELDS = frozenset(
    {
        "action_key",
        "source_run_id",
        "source_result_id",
        "action_type",
        "responsibility_reference",
        "permission_reference",
        "status",
        "created_at_utc",
        "updated_at_utc",
    }
)


def _profile_identifier(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_IDENTIFIER.fullmatch(normalized) is None:
        raise BitableValidationError(f"{field_name} must be a safe identifier")
    return normalized


def _profile_field_name(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 100
        or any(ord(character) < 32 for character in normalized)
    ):
        raise BitableValidationError(f"{field_name} must be a safe field name")
    return normalized


def _profile_text(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 200:
        raise BitableValidationError(f"{field_name} must be safe text")
    return normalized


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
class RecheckActionFieldProfile:
    profile_id: str
    field_names: Mapping[str, str]
    value_labels: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        _profile_identifier(self.profile_id, field_name="profile_id")
        if set(self.field_names) != _ACTION_FIELDS:
            raise BitableValidationError(
                "recheck action field profile must map every action field"
            )
        names = {
            key: _profile_field_name(value, field_name=f"field_names.{key}")
            for key, value in self.field_names.items()
        }
        if len(set(names.values())) != len(names):
            raise BitableValidationError(
                "recheck action field profile names must be unique"
            )
        labels: dict[str, Mapping[str, str]] = {}
        for field_name, mapping in self.value_labels.items():
            if field_name not in _ACTION_FIELDS or not isinstance(mapping, Mapping):
                raise BitableValidationError(
                    "recheck action value labels are invalid"
                )
            labels[field_name] = MappingProxyType(
                {
                    _profile_text(key, field_name="value label key"): (
                        _profile_text(value, field_name="value label value")
                    )
                    for key, value in mapping.items()
                }
            )
        object.__setattr__(self, "field_names", MappingProxyType(names))
        object.__setattr__(self, "value_labels", MappingProxyType(labels))


IDENTITY_RECHECK_ACTION_FIELD_PROFILE = RecheckActionFieldProfile(
    profile_id="recheck-fields-identity-v1",
    field_names={name: name for name in _ACTION_FIELDS},
    value_labels={},
)
CHINESE_RECHECK_ACTION_FIELD_PROFILE = RecheckActionFieldProfile(
    profile_id="recheck-fields-zh-cn-v1",
    field_names={
        "action_key": "复检建单键",
        "source_run_id": "来源任务ID",
        "source_result_id": "来源建议结果ID",
        "action_type": "动作类型",
        "responsibility_reference": "责任人",
        "permission_reference": "权限依据",
        "status": "任务状态",
        "created_at_utc": "创建时间UTC",
        "updated_at_utc": "更新时间UTC",
    },
    value_labels={
        "action_type": {Decision.RECHECK.value: "发起复检"},
        "status": {_INITIAL_STATUS: "待处理"},
    },
)


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
        field_profile: RecheckActionFieldProfile = (
            IDENTITY_RECHECK_ACTION_FIELD_PROFILE
        ),
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
        if not isinstance(field_profile, RecheckActionFieldProfile):
            raise TypeError("field_profile must be a RecheckActionFieldProfile")
        self._client = client
        self._app_token = _safe_identifier(app_token, field_name="app_token")
        self._table_id = _safe_identifier(table_id, field_name="table_id")
        self._authorization_verifier = authorization_verifier
        self._receipt_store = receipt_store
        self._clock = clock
        self._field_profile = field_profile
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
            event_type=RECHECK_ACTION_EVENT_TYPE,
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
                field_profile=self._field_profile,
            )
        response = self._client.create_bitable_record(
            app_token=self._app_token,
            table_id=self._table_id,
            fields=_encode_action_fields(
                expected_fields,
                field_profile=self._field_profile,
            ),
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
            field_profile=self._field_profile,
        )

    def _search(self, action_key: str) -> list[Mapping[str, object]]:
        search = self._client.search_bitable_records(
            app_token=self._app_token,
            table_id=self._table_id,
            field_name=self._field_profile.field_names["action_key"],
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


def _encode_action_fields(
    fields: Mapping[str, str],
    *,
    field_profile: RecheckActionFieldProfile,
) -> dict[str, str]:
    if set(fields) != _ACTION_FIELDS:
        raise BitableValidationError("recheck action fields are incomplete")
    encoded: dict[str, str] = {}
    for field_name, value in fields.items():
        labels = field_profile.value_labels.get(field_name, {})
        encoded[field_profile.field_names[field_name]] = labels.get(value, value)
    return encoded


def _decoded_action_value(
    stored_fields: Mapping[str, object],
    *,
    field_name: str,
    field_profile: RecheckActionFieldProfile,
) -> object:
    remote_value = stored_fields.get(field_profile.field_names[field_name])
    labels = field_profile.value_labels.get(field_name, {})
    reverse = {label: value for value, label in labels.items()}
    if isinstance(remote_value, str):
        return reverse.get(remote_value, remote_value)
    return remote_value


def _existing_receipt(
    item: Mapping[str, object],
    *,
    expected_fields: Mapping[str, str],
    action_type: Decision,
    field_profile: RecheckActionFieldProfile,
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
        if _decoded_action_value(
            stored_fields,
            field_name=field_name,
            field_profile=field_profile,
        ) != expected_fields[field_name]:
            raise BitableConflictError(
                f"stored recheck action {field_name} conflicts with the requested action"
            )
    status = _safe_stored_text(
        _decoded_action_value(
            stored_fields,
            field_name="status",
            field_profile=field_profile,
        ),
        field_name="status",
    )
    created_at = _stored_timestamp(
        _decoded_action_value(
            stored_fields,
            field_name="created_at_utc",
            field_profile=field_profile,
        ),
        field_name="created_at_utc",
    )
    updated_at = _stored_timestamp(
        _decoded_action_value(
            stored_fields,
            field_name="updated_at_utc",
            field_profile=field_profile,
        ),
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
    "CHINESE_RECHECK_ACTION_FIELD_PROFILE",
    "IDENTITY_RECHECK_ACTION_FIELD_PROFILE",
    "RECHECK_ACTION_EVENT_TYPE",
    "FeishuRecheckActionService",
    "RecheckActionAuthorizationError",
    "RecheckActionAuthorizationVerifier",
    "RecheckActionFieldProfile",
    "RecheckActionInProgressError",
    "RecheckActionReceipt",
    "derive_recheck_action_key",
]
