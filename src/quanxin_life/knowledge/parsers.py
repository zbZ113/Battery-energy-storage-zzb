"""Safe PDF and UTF-8 text parsing with page and section preservation."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass

MAX_CHUNK_CHARACTERS = 2_800
CHUNK_OVERLAP_CHARACTERS = 400
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")


class KnowledgeParsingError(ValueError):
    """Raised when a source cannot be safely converted to searchable text."""


@dataclass(frozen=True)
class ParsedKnowledgeChunk:
    text: str
    page_start: int | None
    page_end: int | None
    section_label: str | None


def parse_knowledge_document(
    payload: bytes,
    *,
    content_type: str,
) -> tuple[ParsedKnowledgeChunk, ...]:
    """Parse an approved source without OCR or executable deserialization."""

    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type == "application/pdf":
        return _parse_pdf(payload)
    if media_type in {"text/markdown", "text/plain"}:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeParsingError("knowledge text must be valid UTF-8") from exc
        if media_type == "text/markdown":
            return _parse_markdown(text)
        return tuple(
            ParsedKnowledgeChunk(chunk, None, None, None)
            for chunk in _split_text(text)
        )
    raise KnowledgeParsingError("only PDF, Markdown and plain text are supported")


def _parse_markdown(text: str) -> tuple[ParsedKnowledgeChunk, ...]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))
        current_lines.clear()

    for line in text.splitlines():
        heading = _MARKDOWN_HEADING.match(line)
        if heading is not None:
            flush()
            current_heading = heading.group("title").strip()
            continue
        current_lines.append(line)
    flush()
    chunks = [
        ParsedKnowledgeChunk(chunk, None, None, heading)
        for heading, body in sections
        for chunk in _split_text(body)
    ]
    if not chunks:
        raise KnowledgeParsingError("knowledge document contains no searchable text")
    return tuple(chunks)


def _parse_pdf(payload: bytes) -> tuple[ParsedKnowledgeChunk, ...]:
    try:
        pymupdf = importlib.import_module("pymupdf")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise KnowledgeParsingError(
            "PDF parsing requires installing the knowledge dependency group"
        ) from exc
    try:
        document = pymupdf.open(stream=payload, filetype="pdf")
    except Exception as exc:
        raise KnowledgeParsingError("PDF source is invalid or unreadable") from exc
    chunks: list[ParsedKnowledgeChunk] = []
    try:
        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            for chunk in _split_text(page_text):
                chunks.append(
                    ParsedKnowledgeChunk(chunk, page_index, page_index, None)
                )
    finally:
        document.close()
    if not chunks:
        raise KnowledgeParsingError("OCR_REQUIRED: PDF contains no extractable text")
    return tuple(chunks)


def _split_text(text: str) -> tuple[str, ...]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return ()
    if len(normalized) <= MAX_CHUNK_CHARACTERS:
        return (normalized,)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + MAX_CHUNK_CHARACTERS, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - CHUNK_OVERLAP_CHARACTERS
    return tuple(chunks)


__all__ = [
    "CHUNK_OVERLAP_CHARACTERS",
    "MAX_CHUNK_CHARACTERS",
    "KnowledgeParsingError",
    "ParsedKnowledgeChunk",
    "parse_knowledge_document",
]
