from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from quanxin_life.integrations.feishu.routing import (
    FeishuEventReferenceError,
    FeishuInboundEventKind,
    parse_feishu_event_reference,
)


def test_text_message_retains_only_machine_references() -> None:
    payload = {
        "header": {"event_id": "evt_text", "event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_sender"}},
            "message": {
                "message_id": "om_text",
                "chat_id": "oc_chat",
                "message_type": "text",
                "content": json.dumps({"text": "private full user prompt"}),
            },
        },
    }

    reference = parse_feishu_event_reference(payload)

    assert reference.kind is FeishuInboundEventKind.MESSAGE
    assert reference.event_id == "evt_text"
    assert reference.chat_id == "oc_chat"
    assert reference.user_id == "ou_sender"
    assert reference.message_id == "om_text"
    assert reference.file_key is None
    assert "private full user prompt" not in repr(reference)
    assert "content" not in reference.model_dump(mode="json")


def test_file_message_extracts_file_key_and_safe_filename_only() -> None:
    payload = {
        "header": {
            "event_id": "evt_file",
            "event_type": "im.message.receive_v1",
            "create_time": "1786424400000",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_sender"}},
            "message": {
                "message_id": "om_file",
                "chat_id": "oc_chat",
                "message_type": "file",
                "content": json.dumps(
                    {"file_key": "file_source", "file_name": "observed.csv"}
                ),
            },
        },
    }

    reference = parse_feishu_event_reference(payload)

    assert reference.kind is FeishuInboundEventKind.FILE
    assert reference.file_key == "file_source"
    assert reference.file_name == "observed.csv"
    assert reference.receive_id_type == "chat_id"
    assert reference.event_time == datetime.fromtimestamp(1786424400, tz=UTC)


def test_card_action_extracts_only_allowlisted_identity_and_action_value() -> None:
    payload = {
        "header": {
            "event_id": "evt_action",
            "event_type": "card.action.trigger",
        },
        "event": {
            "operator": {"open_id": "ou_operator"},
            "context": {
                "open_chat_id": "oc_chat",
                "open_message_id": "om_card",
            },
            "action": {
                "tag": "button",
                "value": {"command": "approve", "run_id": "run-safe"},
            },
        },
    }

    reference = parse_feishu_event_reference(payload)

    assert reference.kind is FeishuInboundEventKind.CARD_ACTION
    assert reference.action_value == {"command": "approve", "run_id": "run-safe"}
    assert reference.user_id == "ou_operator"


@pytest.mark.parametrize(
    "payload",
    [
        {"header": {"event_id": "evt", "event_type": "im.message.receive_v1"}},
        {
            "header": {"event_id": "evt", "event_type": "im.message.receive_v1"},
            "event": {"message": {"message_type": "file", "content": "not-json"}},
        },
        {
            "header": {"event_id": "evt", "event_type": "unknown.event"},
            "event": {},
        },
    ],
)
def test_unsupported_or_incomplete_event_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(FeishuEventReferenceError):
        parse_feishu_event_reference(payload)
