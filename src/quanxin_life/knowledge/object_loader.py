"""Verified UTF-8 knowledge chunk loading from immutable object storage."""

from __future__ import annotations

from typing import Protocol

from quanxin_life.infrastructure.object_store import StoredObjectRef


class VerifiedObjectReader(Protocol):
    """Read bytes only after validating an immutable object reference."""

    def get_bytes(self, stored: StoredObjectRef) -> bytes: ...


class MinioKnowledgeTextLoader:
    """Adapt the verified object store to the retrieval backend text contract."""

    def __init__(self, store: VerifiedObjectReader) -> None:
        self._store = store

    def load_verified_text(
        self,
        *,
        object_uri: str,
        expected_sha256: str,
        size_bytes: int,
        content_type: str,
    ) -> str:
        stored = StoredObjectRef(
            uri=object_uri,
            sha256=expected_sha256,
            size_bytes=size_bytes,
            content_type=content_type,
        )
        payload = self._store.get_bytes(stored)
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("knowledge chunk is not valid UTF-8 text") from exc


__all__ = ["MinioKnowledgeTextLoader", "VerifiedObjectReader"]
