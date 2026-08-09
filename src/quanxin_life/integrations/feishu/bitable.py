"""Idempotent Feishu Bitable summaries keyed only by a safe ``run_id``."""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

BITABLE_RUN_FIELD_NAMES = frozenset(
    {
        "run_id",
        "task_type",
        "task_status",
        "data_batch_id",
        "input_file_sha256",
        "cell_reference",
        "primary_result_id",
        "model_route",
        "model_version",
        "data_version",
        "feature_version",
        "evidence_level",
        "warnings",
        "report_link",
        "created_at_utc",
        "updated_at_utc",
    }
)
_REQUIRED_FIELDS = frozenset(
    {"run_id", "task_type", "task_status", "created_at_utc", "updated_at_utc"}
)
_TIMESTAMP_FIELDS = frozenset({"created_at_utc", "updated_at_utc"})
_MAX_TEXT_LENGTH = 2_000


class BitableWriterError(RuntimeError):
    """Base failure for deterministic Bitable summary persistence."""


class BitableValidationError(BitableWriterError):
    """Raised before network access when one summary is unsafe."""


class BitableConflictError(BitableWriterError):
    """Raised when ``run_id`` no longer identifies one remote record."""


class BitableProtocolError(BitableWriterError):
    """Raised when Feishu returns an unusable response shape."""


class BitableWriteAction(StrEnum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"


@dataclass(frozen=True, slots=True)
class BitableWriteResult:
    run_id: str
    record_id: str
    action: BitableWriteAction


class FeishuBitableClient(Protocol):
    def search_bitable_records(
        self,
        *,
        app_token: str,
        table_id: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, object]: ...

    def create_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]: ...

    def update_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        record_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]: ...


class FeishuBitableWriter:
    """Create or update exactly one scalar-only Bitable row per ``run_id``."""

    def __init__(
        self,
        client: FeishuBitableClient,
        *,
        app_token: str,
        table_id: str,
    ) -> None:
        if not callable(getattr(client, "search_bitable_records", None)):
            raise TypeError("client must support Bitable record search")
        if not callable(getattr(client, "create_bitable_record", None)):
            raise TypeError("client must support Bitable record creation")
        if not callable(getattr(client, "update_bitable_record", None)):
            raise TypeError("client must support Bitable record updates")
        self._client = client
        self._app_token = _safe_identifier(app_token, field_name="app_token")
        self._table_id = _safe_identifier(table_id, field_name="table_id")
        self._lock = threading.RLock()

    def upsert(self, fields: Mapping[str, object]) -> BitableWriteResult:
        """Persist one complete summary without accepting arrays or nested objects."""

        normalized = _normalize_fields(fields)
        run_id = normalized["run_id"]
        assert isinstance(run_id, str)
        with self._lock:
            search = self._client.search_bitable_records(
                app_token=self._app_token,
                table_id=self._table_id,
                field_name="run_id",
                field_value=run_id,
            )
            items = _search_items(search)
            if len(items) > 1:
                raise BitableConflictError(
                    "run_id matched multiple records; refusing a concurrent overwrite"
                )
            if not items:
                response = self._client.create_bitable_record(
                    app_token=self._app_token,
                    table_id=self._table_id,
                    fields=normalized,
                )
                action = BitableWriteAction.CREATED
            else:
                record_id = _record_id(items[0], response_label="search response")
                response = self._client.update_bitable_record(
                    app_token=self._app_token,
                    table_id=self._table_id,
                    record_id=record_id,
                    fields=normalized,
                )
                action = BitableWriteAction.UPDATED
            return BitableWriteResult(
                run_id=run_id,
                record_id=_record_response_id(response),
                action=action,
            )


def _normalize_fields(fields: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(fields, Mapping):
        raise BitableValidationError("Bitable fields must be a mapping")
    unknown = set(fields) - BITABLE_RUN_FIELD_NAMES
    if unknown:
        raise BitableValidationError("Bitable field is not allowlisted")
    missing = _REQUIRED_FIELDS - set(fields)
    if missing:
        raise BitableValidationError("Bitable summary is missing required fields")

    normalized: dict[str, object] = {}
    for name, value in fields.items():
        if name in _TIMESTAMP_FIELDS:
            normalized[name] = _utc_timestamp(value, field_name=name)
        elif value is None:
            normalized[name] = None
        elif not isinstance(value, str):
            raise BitableValidationError(
                f"{name} must be a scalar string, datetime, or null"
            )
        else:
            normalized[name] = _bounded_text(value, field_name=name)

    run_id = normalized["run_id"]
    assert isinstance(run_id, str)
    normalized["run_id"] = _safe_identifier(run_id, field_name="run_id")
    sha256 = normalized.get("input_file_sha256")
    if sha256 is not None and (
        not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None
    ):
        raise BitableValidationError("input_file_sha256 must be lowercase SHA-256")
    report_link = normalized.get("report_link")
    if report_link is not None and (
        not isinstance(report_link, str) or not report_link.startswith("https://")
    ):
        raise BitableValidationError("report_link must use HTTPS")
    created = normalized["created_at_utc"]
    updated = normalized["updated_at_utc"]
    assert isinstance(created, str)
    assert isinstance(updated, str)
    if created > updated:
        raise BitableValidationError("created_at_utc must not follow updated_at_utc")
    return normalized


def _utc_timestamp(value: object, *, field_name: str) -> str:
    if not isinstance(value, datetime):
        raise BitableValidationError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise BitableValidationError(f"{field_name} must include a timezone")
    return value.astimezone(UTC).isoformat()


def _bounded_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_TEXT_LENGTH
        or any(ord(character) < 32 and character not in "\n\t" for character in normalized)
    ):
        raise BitableValidationError(f"{field_name} must be bounded safe text")
    return normalized


def _safe_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_IDENTIFIER.fullmatch(normalized) is None:
        raise BitableValidationError(f"{field_name} must be a safe identifier")
    return normalized


def _search_items(response: object) -> list[object]:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable search response is invalid")
    if response.get("has_more") is True:
        raise BitableConflictError(
            "Bitable search has additional records; uniqueness is not proven"
        )
    items = response.get("items")
    if not isinstance(items, list):
        raise BitableProtocolError("Bitable search response has no item list")
    if any(not isinstance(item, Mapping) for item in items):
        raise BitableProtocolError("Bitable search response contains an invalid record")
    return items


def _record_response_id(response: object) -> str:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable record response is invalid")
    record = response.get("record")
    return _record_id(record, response_label="record response")


def _record_id(value: object, *, response_label: str) -> str:
    if not isinstance(value, Mapping):
        raise BitableProtocolError(f"Bitable {response_label} is invalid")
    record_id = value.get("record_id")
    if not isinstance(record_id, str) or _SAFE_IDENTIFIER.fullmatch(record_id) is None:
        raise BitableProtocolError(f"Bitable {response_label} has no safe record_id")
    return record_id


__all__ = [
    "BITABLE_RUN_FIELD_NAMES",
    "BitableConflictError",
    "BitableProtocolError",
    "BitableValidationError",
    "BitableWriteAction",
    "BitableWriteResult",
    "BitableWriterError",
    "FeishuBitableWriter",
]
