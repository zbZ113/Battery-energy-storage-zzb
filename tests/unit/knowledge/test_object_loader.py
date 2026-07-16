from __future__ import annotations

import hashlib

import pytest

from quanxin_life.infrastructure.object_store import StoredObjectRef
from quanxin_life.knowledge.object_loader import MinioKnowledgeTextLoader


class _RecordingStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.request: StoredObjectRef | None = None

    def get_bytes(self, stored: StoredObjectRef) -> bytes:
        self.request = stored
        return self.payload


def test_loader_reconstructs_verified_reference_and_decodes_utf8() -> None:
    payload = "磷酸铁锂电芯老化证据".encode()
    digest = hashlib.sha256(payload).hexdigest()
    store = _RecordingStore(payload)

    text = MinioKnowledgeTextLoader(store).load_verified_text(
        object_uri=f"minio://quanxin-artifacts/knowledge/chunks/{digest}",
        expected_sha256=digest,
        size_bytes=len(payload),
        content_type="text/plain; charset=utf-8",
    )

    assert text == "磷酸铁锂电芯老化证据"
    assert store.request == StoredObjectRef(
        uri=f"minio://quanxin-artifacts/knowledge/chunks/{digest}",
        sha256=digest,
        size_bytes=len(payload),
        content_type="text/plain; charset=utf-8",
    )


def test_loader_rejects_non_utf8_chunk() -> None:
    payload = b"\xff\xfe"
    digest = hashlib.sha256(payload).hexdigest()

    with pytest.raises(ValueError, match="UTF-8"):
        MinioKnowledgeTextLoader(_RecordingStore(payload)).load_verified_text(
            object_uri=f"minio://quanxin-artifacts/knowledge/chunks/{digest}",
            expected_sha256=digest,
            size_bytes=len(payload),
            content_type="text/plain; charset=utf-8",
        )
