"""Fail-closed attachment policy for Feishu battery-data uploads."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath

from quanxin_life.application.battery_csv_mapping import (
    BATTERY_CSV_METADATA_ENVELOPE_V1,
)
from quanxin_life.application.ingestion import (
    CANONICAL_CYCLE_CSV_FIELDS,
    MAX_CANONICAL_CSV_BYTES,
)

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
    source_sha256: str | None = None
    mapping_evidence: dict[str, object] | None = field(default=None, repr=False)


class FeishuAttachmentPolicy:
    def __init__(
        self,
        *,
        max_bytes: int = MAX_CANONICAL_CSV_BYTES,
        max_rows: int = 500_000,
        max_cell_chars: int = 10_000,
    ) -> None:
        if max_bytes < 1 or max_bytes > MAX_CANONICAL_CSV_BYTES:
            raise ValueError(
                "attachment max_bytes must be between 1 and the canonical CSV limit"
            )
        self._max_bytes = max_bytes
        if max_rows < 1 or max_rows > 1_000_000:
            raise ValueError("attachment max_rows must be between 1 and 1000000")
        if max_cell_chars < 1 or max_cell_chars > 100_000:
            raise ValueError(
                "attachment max_cell_chars must be between 1 and 100000"
            )
        self._max_rows = max_rows
        self._max_cell_chars = max_cell_chars

    def verify(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        expected_sha256: str | None = None,
    ) -> VerifiedFeishuAttachment:
        verified = self.verify_csv_envelope(
            filename=filename,
            content_type=content_type,
            payload=payload,
            expected_sha256=expected_sha256,
        )
        text = verified.payload.decode("utf-8-sig")
        _verify_canonical_csv_shape(
            text,
            max_rows=self._max_rows,
            max_cell_chars=self._max_cell_chars,
        )
        return verified

    def verify_csv_envelope(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        expected_sha256: str | None = None,
    ) -> VerifiedFeishuAttachment:
        """Verify a safe UTF-8 CSV carrier before reviewed field mapping."""

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
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FeishuAttachmentError("attachment must be UTF-8 canonical CSV") from exc
        _verify_csv_envelope_shape(
            _shape_validation_csv_text(text),
            max_rows=self._max_rows,
            max_cell_chars=self._max_cell_chars,
        )
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


def _verify_canonical_csv_shape(
    text: str,
    *,
    max_rows: int,
    max_cell_chars: int,
) -> None:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise FeishuAttachmentError("attachment has no canonical CSV header") from exc
    if tuple(header) != CANONICAL_CYCLE_CSV_FIELDS:
        raise FeishuAttachmentError("attachment canonical CSV header is invalid")
    identifier_indexes = (
        CANONICAL_CYCLE_CSV_FIELDS.index("dataset_id"),
        CANONICAL_CYCLE_CSV_FIELDS.index("cell_id"),
    )
    row_count = 0
    try:
        for row_count, row in enumerate(reader, start=1):
            if row_count > max_rows:
                raise FeishuAttachmentError("attachment exceeds the CSV row limit")
            if len(row) != len(CANONICAL_CYCLE_CSV_FIELDS):
                raise FeishuAttachmentError(
                    "attachment CSV row does not match the canonical column count"
                )
            if any(len(cell) > max_cell_chars for cell in row):
                raise FeishuAttachmentError("attachment exceeds the CSV cell length limit")
            for index in identifier_indexes:
                if _looks_like_spreadsheet_formula(row[index]):
                    raise FeishuAttachmentError(
                        "attachment identifier contains spreadsheet formula content"
                    )
    except csv.Error as exc:
        raise FeishuAttachmentError("attachment CSV structure is invalid") from exc
    if row_count == 0:
        raise FeishuAttachmentError("attachment canonical CSV has no data rows")


def _verify_csv_envelope_shape(
    text: str,
    *,
    max_rows: int,
    max_cell_chars: int,
) -> None:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise FeishuAttachmentError("attachment has no CSV header") from exc
    if not header or any(not value.strip() or value != value.strip() for value in header):
        raise FeishuAttachmentError("attachment CSV header is invalid")
    if len(header) != len(set(header)):
        raise FeishuAttachmentError("attachment CSV header contains duplicate columns")
    if any(len(value) > max_cell_chars for value in header):
        raise FeishuAttachmentError("attachment exceeds the CSV cell length limit")
    row_count = 0
    try:
        for row_count, row in enumerate(reader, start=1):
            if row_count > max_rows:
                raise FeishuAttachmentError("attachment exceeds the CSV row limit")
            if len(row) != len(header):
                raise FeishuAttachmentError(
                    "attachment CSV row does not match the header column count"
                )
            if any(len(cell) > max_cell_chars for cell in row):
                raise FeishuAttachmentError("attachment exceeds the CSV cell length limit")
            if any(_looks_like_spreadsheet_formula(cell) for cell in row):
                raise FeishuAttachmentError(
                    "attachment contains spreadsheet formula content"
                )
    except csv.Error as exc:
        raise FeishuAttachmentError("attachment CSV structure is invalid") from exc
    if row_count == 0:
        raise FeishuAttachmentError("attachment CSV has no data rows")


def _shape_validation_csv_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != BATTERY_CSV_METADATA_ENVELOPE_V1:
        return text
    if len(lines) < 3:
        raise FeishuAttachmentError("self-described CSV envelope is incomplete")
    return "".join(lines[2:])


def _looks_like_spreadsheet_formula(value: str) -> bool:
    normalized = value.lstrip()
    if not normalized:
        return False
    if normalized[0] in {"=", "+", "@"}:
        return True
    if not normalized.startswith("-"):
        return False
    try:
        float(normalized)
    except ValueError:
        return True
    return False


__all__ = [
    "FeishuAttachmentError",
    "FeishuAttachmentPolicy",
    "VerifiedFeishuAttachment",
]
