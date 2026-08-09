"""Fail-closed attachment policy for Feishu battery-data uploads."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePath

from quanxin_life.application.ingestion import MAX_CANONICAL_CSV_BYTES

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_CONTENT_TYPES = frozenset(
    {"application/csv", "application/octet-stream", "text/csv"}
)
_DANGEROUS_PREFIXES = (
    b"PK\x03\x04",
    b"MZ",
    b"\x7fELF",
    b"PAR1",
    b"\x80\x02",
    b"\x80\x03",
    b"\x80\x04",
    b"\x80\x05",
)


class FeishuAttachmentError(ValueError):
    """Raised before untrusted attachment bytes reach the data layer."""


@dataclass(frozen=True, slots=True)
class VerifiedFeishuAttachment:
    filename: str
    content_type: str
    payload: bytes = field(repr=False)
    sha256: str
    size_bytes: int


class FeishuAttachmentPolicy:
    def __init__(self, *, max_bytes: int = MAX_CANONICAL_CSV_BYTES) -> None:
        if max_bytes < 1 or max_bytes > MAX_CANONICAL_CSV_BYTES:
            raise ValueError(
                "attachment max_bytes must be between 1 and the canonical CSV limit"
            )
        self._max_bytes = max_bytes

    def verify(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        expected_sha256: str | None = None,
    ) -> VerifiedFeishuAttachment:
        checked_filename = _csv_filename(filename)
        checked_content_type = _csv_content_type(content_type)
        if not isinstance(payload, bytes):
            raise FeishuAttachmentError("attachment payload must be exact bytes")
        if not payload:
            raise FeishuAttachmentError("attachment content must not be empty")
        if len(payload) > self._max_bytes:
            raise FeishuAttachmentError("attachment exceeds the configured size limit")
        if payload.startswith(_DANGEROUS_PREFIXES) or b"\x00" in payload:
            raise FeishuAttachmentError("attachment content is not canonical CSV text")
        try:
            payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FeishuAttachmentError("attachment must be UTF-8 canonical CSV") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            checked_expected = expected_sha256.strip().lower()
            if _SHA256.fullmatch(checked_expected) is None:
                raise FeishuAttachmentError("expected SHA-256 is invalid")
            if digest != checked_expected:
                raise FeishuAttachmentError("attachment SHA-256 does not match")
        return VerifiedFeishuAttachment(
            filename=checked_filename,
            content_type=checked_content_type,
            payload=payload,
            sha256=digest,
            size_bytes=len(payload),
        )


def _csv_filename(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or normalized != value
        or len(normalized) > 255
        or PurePath(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
        or any(ord(character) < 32 for character in normalized)
    ):
        raise FeishuAttachmentError("attachment filename must be a safe basename")
    if PurePath(normalized).suffix.lower() != ".csv":
        raise FeishuAttachmentError("first-phase attachments must be canonical CSV")
    return normalized


def _csv_content_type(value: str) -> str:
    normalized = value.split(";", maxsplit=1)[0].strip().lower()
    if normalized not in _ALLOWED_CONTENT_TYPES:
        raise FeishuAttachmentError("attachment content type is not allowed for CSV")
    return normalized


__all__ = [
    "FeishuAttachmentError",
    "FeishuAttachmentPolicy",
    "VerifiedFeishuAttachment",
]
