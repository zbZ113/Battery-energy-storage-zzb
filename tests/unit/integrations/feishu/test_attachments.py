from __future__ import annotations

import hashlib

import pytest

from quanxin_life.integrations.feishu.attachments import (
    FeishuAttachmentError,
    FeishuAttachmentPolicy,
)


def _csv_payload() -> bytes:
    return b"dataset_id,cell_id,cycle_index\nsource,cell,1\n"


def test_csv_attachment_preserves_exact_bytes_and_sha256() -> None:
    payload = _csv_payload()

    verified = FeishuAttachmentPolicy(max_bytes=1024).verify(
        filename="observed-cycles.csv",
        content_type="text/csv; charset=utf-8",
        payload=payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert verified.filename == "observed-cycles.csv"
    assert verified.content_type == "text/csv"
    assert verified.payload == payload
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()
    assert verified.size_bytes == len(payload)


def test_attachment_over_the_configured_limit_is_rejected() -> None:
    with pytest.raises(FeishuAttachmentError, match="size limit"):
        FeishuAttachmentPolicy(max_bytes=4).verify(
            filename="observed.csv",
            content_type="text/csv",
            payload=_csv_payload(),
        )

@pytest.mark.parametrize(
    "filename",
    [
        "payload.parquet",
        "payload.xlsx",
        "payload.zip",
        "payload.pkl",
        "payload.pickle",
        "payload.joblib",
        "payload.pt",
        "payload.pth",
        "payload.exe",
        "payload.csv.exe",
    ],
)
def test_unsupported_or_executable_attachment_suffix_is_rejected(filename: str) -> None:
    with pytest.raises(FeishuAttachmentError, match="canonical CSV"):
        FeishuAttachmentPolicy().verify(
            filename=filename,
            content_type="application/octet-stream",
            payload=_csv_payload(),
        )


@pytest.mark.parametrize(
    "filename",
    ["../observed.csv", "folder/observed.csv", "folder\\observed.csv", " observed.csv"],
)
def test_attachment_filename_cannot_escape_or_be_ambiguous(filename: str) -> None:
    with pytest.raises(FeishuAttachmentError, match="filename"):
        FeishuAttachmentPolicy().verify(
            filename=filename,
            content_type="text/csv",
            payload=_csv_payload(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"PK\x03\x04archive",
        b"MZexecutable",
        b"\x7fELFbinary",
        b"PAR1parquet",
        b"\x80\x04pickle",
        b"header\x00binary",
    ],
)
def test_dangerous_or_disguised_magic_bytes_are_rejected(payload: bytes) -> None:
    with pytest.raises(FeishuAttachmentError, match="content"):
        FeishuAttachmentPolicy().verify(
            filename="observed.csv",
            content_type="text/csv",
            payload=payload,
        )


def test_attachment_rejects_mismatched_sha256() -> None:
    with pytest.raises(FeishuAttachmentError, match="SHA-256"):
        FeishuAttachmentPolicy().verify(
            filename="observed.csv",
            content_type="text/csv",
            payload=_csv_payload(),
            expected_sha256="0" * 64,
        )


def test_attachment_rejects_non_utf8_or_mismatched_content_type() -> None:
    policy = FeishuAttachmentPolicy()

    with pytest.raises(FeishuAttachmentError, match="UTF-8"):
        policy.verify(
            filename="observed.csv",
            content_type="text/csv",
            payload=b"\xff\xfe",
        )
    with pytest.raises(FeishuAttachmentError, match="content type"):
        policy.verify(
            filename="observed.csv",
            content_type="application/pdf",
            payload=_csv_payload(),
        )
