from __future__ import annotations

import pymupdf
import pytest

from quanxin_life.knowledge.parsers import (
    KnowledgeParsingError,
    parse_knowledge_document,
)


def _pdf_bytes(*page_texts: str) -> bytes:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def test_pdf_parser_preserves_page_number_for_each_chunk() -> None:
    chunks = parse_knowledge_document(
        _pdf_bytes("LFP aging evidence", "Conformal limits"),
        content_type="application/pdf",
    )

    assert [(chunk.text, chunk.page_start, chunk.page_end) for chunk in chunks] == [
        ("LFP aging evidence", 1, 1),
        ("Conformal limits", 2, 2),
    ]


def test_pdf_without_extractable_text_requires_ocr_instead_of_guessing() -> None:
    with pytest.raises(KnowledgeParsingError, match="OCR_REQUIRED"):
        parse_knowledge_document(
            _pdf_bytes(""),
            content_type="application/pdf",
        )


def test_invalid_utf8_text_is_rejected() -> None:
    with pytest.raises(KnowledgeParsingError, match="UTF-8"):
        parse_knowledge_document(b"\xff\xfe", content_type="text/plain")
