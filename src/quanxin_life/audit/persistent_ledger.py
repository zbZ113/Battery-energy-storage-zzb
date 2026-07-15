"""Append-only JSONL persistence for validated :class:`ToolResult` evidence.

The journal is intentionally a non-executable, human-inspectable format.  Every
record is reconstructed through the public Pydantic contract during startup;
any malformed, truncated, duplicated, or conflicting entry makes startup fail
closed instead of silently dropping evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from quanxin_life.audit.numeric_firewall import (
    AuditLedger,
    DuplicateAuditResultError,
    InvalidAuditResultError,
)
from quanxin_life.core import ToolResult


class JsonlAuditLedger(AuditLedger):
    """Single-process append-only audit ledger backed by strict JSON Lines."""

    def __init__(self, journal_path: str | Path) -> None:
        self._journal_path = Path(journal_path).expanduser()
        self._validate_journal_path()
        super().__init__()
        self._restore_journal()

    @property
    def journal_path(self) -> Path:
        """Return the configured journal path without exposing mutable state."""

        return self._journal_path

    def register_result(self, result: ToolResult) -> ToolResult:
        """Validate, durably append, then expose one immutable tool result."""

        validated = self._validated_result(result)
        with self._lock:
            if validated.result_id in self._results:
                raise DuplicateAuditResultError(
                    f"duplicate ToolResult result_id: {validated.result_id}"
                )
            self._append_result(validated)
            self._results[validated.result_id] = validated
        return ToolResult.model_validate(validated.model_dump(mode="json"))

    def ensure_result(self, result: ToolResult) -> ToolResult:
        """Restore identical evidence idempotently and append only new evidence."""

        validated = self._validated_result(result)
        with self._lock:
            existing = self._results.get(validated.result_id)
            if existing is None:
                self._append_result(validated)
                self._results[validated.result_id] = validated
            elif existing != validated:
                raise DuplicateAuditResultError(
                    f"conflicting ToolResult content for result_id: {validated.result_id}"
                )
        return ToolResult.model_validate(validated.model_dump(mode="json"))

    def _validate_journal_path(self) -> None:
        if self._journal_path.exists() and self._journal_path.is_symlink():
            raise InvalidAuditResultError("audit journal must not be a symbolic link")
        if self._journal_path.exists() and not self._journal_path.is_file():
            raise InvalidAuditResultError("audit journal path must be a regular file")
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _restore_journal(self) -> None:
        if not self._journal_path.exists():
            return
        try:
            with self._journal_path.open("r", encoding="utf-8", newline="") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        raise InvalidAuditResultError(
                            f"audit journal contains a blank record at line {line_number}"
                        )
                    try:
                        payload = json.loads(raw_line)
                        result = ToolResult.model_validate(payload)
                        super().register_result(result)
                    except DuplicateAuditResultError as exc:
                        raise InvalidAuditResultError(
                            f"audit journal contains a duplicate result at line {line_number}"
                        ) from exc
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise InvalidAuditResultError(
                            f"audit journal contains an invalid record at line {line_number}"
                        ) from exc
        except UnicodeError as exc:
            raise InvalidAuditResultError("audit journal must be valid UTF-8") from exc

    def _append_result(self, result: ToolResult) -> None:
        payload = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(self._journal_path, flags, 0o600)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("audit journal append was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
