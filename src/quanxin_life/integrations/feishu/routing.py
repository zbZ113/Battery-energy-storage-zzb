"""Sanitized Feishu event references for durable routing without message bodies."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class FeishuEventReferenceError(ValueError):
    """Raised when an inbound event cannot produce safe routing references."""


class FeishuInboundEventKind(StrEnum):
    MESSAGE = "MESSAGE"
    FILE = "FILE"
    CARD_ACTION = "CARD_ACTION"


class FeishuEventReference(BaseModel):
    """Bounded machine references; raw message content is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    event_type: str
    kind: FeishuInboundEventKind
    chat_id: str | None = None
    user_id: str | None = None
    message_id: str | None = None
    file_key: str | None = None
    file_name: str | None = None
    receive_id_type: str = "chat_id"
    event_time: datetime | None = None
    action_value: dict[str, str] = Field(default_factory=dict, max_length=16)

    @field_validator(
        "event_id",
        "event_type",
        "chat_id",
        "user_id",
        "message_id",
        "file_key",
        "receive_id_type",
    )
    @classmethod
    def references_are_safe(cls, value: str | None) -> str | None:
        return _optional_reference(value)

    @field_validator("file_name")
    @classmethod
    def filename_is_safe(cls, value: str | None) -> str | None:
        return _optional_filename(value)

    @field_validator("event_time")
    @classmethod
    def event_time_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Feishu event_time must include a timezone")
        return value.astimezone(UTC)

    @field_validator("action_value")
    @classmethod
    def action_values_are_machine_references(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        return {
            _reference(key, field_name="action key"): _reference(
                item, field_name="action value"
            )
            for key, item in value.items()
        }


def parse_feishu_event_reference(payload: dict[str, object]) -> FeishuEventReference:
    """Parse supported events without retaining free-form user content."""

    header = _mapping(payload.get("header"), field_name="header")
    event_id = _reference(header.get("event_id"), field_name="event_id")
    event_type = _reference(header.get("event_type"), field_name="event_type")
    event = _mapping(payload.get("event"), field_name="event")
    if event_type == "im.message.receive_v1":
        return _parse_message(
            event_id=event_id,
            event_type=event_type,
            event=event,
            header_event_time=header.get("create_time"),
        )
    if event_type == "card.action.trigger":
        return _parse_card_action(
            event_id=event_id,
            event_type=event_type,
            event=event,
            header_event_time=header.get("create_time"),
        )
    raise FeishuEventReferenceError("Feishu event type is not supported")


def _parse_message(
    *,
    event_id: str,
    event_type: str,
    event: dict[str, object],
    header_event_time: object,
) -> FeishuEventReference:
    message = _mapping(event.get("message"), field_name="event.message")
    sender = _optional_mapping(event.get("sender"))
    sender_id = _optional_mapping(sender.get("sender_id"))
    raw_message_type = message.get("message_type")
    message_type = (
        "text"
        if raw_message_type is None
        else _reference(raw_message_type, field_name="message_type")
    )
    common: dict[str, Any] = {
        "event_id": event_id,
        "event_type": event_type,
        "chat_id": _optional_reference(message.get("chat_id")),
        "user_id": _optional_reference(sender_id.get("open_id")),
        "message_id": _reference(message.get("message_id"), field_name="message_id"),
        "receive_id_type": "chat_id",
        "event_time": _event_time(header_value=header_event_time, message=message),
    }
    if message_type == "text":
        return FeishuEventReference(kind=FeishuInboundEventKind.MESSAGE, **common)
    if message_type != "file":
        raise FeishuEventReferenceError("Feishu message type is not supported")
    content = message.get("content")
    if not isinstance(content, str) or len(content) > 10_000:
        raise FeishuEventReferenceError("Feishu file content envelope is invalid")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise FeishuEventReferenceError("Feishu file content envelope is invalid") from exc
    file_content = _mapping(decoded, field_name="file content")
    return FeishuEventReference(
        kind=FeishuInboundEventKind.FILE,
        file_key=_reference(file_content.get("file_key"), field_name="file_key"),
        file_name=_filename(file_content.get("file_name")),
        **common,
    )


def _parse_card_action(
    *,
    event_id: str,
    event_type: str,
    event: dict[str, object],
    header_event_time: object,
) -> FeishuEventReference:
    operator = _mapping(event.get("operator"), field_name="event.operator")
    context = _mapping(event.get("context"), field_name="event.context")
    action = _mapping(event.get("action"), field_name="event.action")
    raw_value = _mapping(action.get("value"), field_name="event.action.value")
    if len(raw_value) > 16 or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_value.items()
    ):
        raise FeishuEventReferenceError("Feishu card action value must be string references")
    action_value = {str(key): str(value) for key, value in raw_value.items()}
    return FeishuEventReference(
        event_id=event_id,
        event_type=event_type,
        kind=FeishuInboundEventKind.CARD_ACTION,
        chat_id=_reference(context.get("open_chat_id"), field_name="chat_id"),
        user_id=_reference(operator.get("open_id"), field_name="user_id"),
        message_id=_reference(
            context.get("open_message_id"), field_name="message_id"
        ),
        action_value=action_value,
        receive_id_type="chat_id",
        event_time=_event_time(header_value=header_event_time, message={}),
    )


def _event_time(
    *,
    header_value: object,
    message: dict[str, object],
) -> datetime | None:
    raw = message.get("create_time", header_value)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.isdigit():
        raise FeishuEventReferenceError("Feishu event time is invalid")
    milliseconds = int(raw)
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise FeishuEventReferenceError("Feishu event time is invalid") from exc


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeishuEventReferenceError(f"Feishu {field_name} must be an object")
    return value


def _optional_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _reference(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FeishuEventReferenceError(f"Feishu {field_name} must be a string")
    normalized = value.strip()
    if _REFERENCE.fullmatch(normalized) is None:
        raise FeishuEventReferenceError(f"Feishu {field_name} is not a safe reference")
    return normalized


def _optional_reference(value: object) -> str | None:
    if value is None:
        return None
    return _reference(value, field_name="optional reference")


def _filename(value: object) -> str:
    if not isinstance(value, str):
        raise FeishuEventReferenceError("Feishu file_name must be a string")
    normalized = value.strip()
    if (
        not normalized
        or normalized != value
        or len(normalized) > 255
        or PurePath(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
        or any(ord(character) < 32 for character in normalized)
    ):
        raise FeishuEventReferenceError("Feishu file_name is not a safe basename")
    return normalized


def _optional_filename(value: str | None) -> str | None:
    return _filename(value) if value is not None else None


__all__ = [
    "FeishuEventReference",
    "FeishuEventReferenceError",
    "FeishuInboundEventKind",
    "parse_feishu_event_reference",
]
