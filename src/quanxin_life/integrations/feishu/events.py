"""Authenticated Feishu event parsing with injected replay protection."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import SecretStr

from .decryptor import FeishuDecryptorError
from .routing import FeishuEventReference, parse_feishu_event_reference
from .security import FeishuSignatureError, FeishuWebhookVerifier


class FeishuEventError(ValueError):
    """Raised when an authenticated callback payload violates the event contract."""


class FeishuPayloadDecryptionUnavailable(FeishuEventError):
    """Raised when Feishu sends ciphertext but no reviewed decryptor is configured."""


class FeishuPayloadDecryptor(Protocol):
    """Injected decryption port; this module deliberately implements no cipher."""

    def decrypt(self, encrypted: str) -> bytes:
        """Return authenticated plaintext bytes or raise a domain error."""
        ...


class FeishuReceiptClaimStatus(StrEnum):
    """Atomic receipt-claim result returned by the persistence adapter."""

    NEW = "NEW"
    PROCESSED = "PROCESSED"
    IN_PROGRESS = "IN_PROGRESS"
    RETRYABLE = "RETRYABLE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class FeishuReceiptClaim:
    """Claim decision plus the opaque ownership token for executable claims."""

    status: FeishuReceiptClaimStatus
    claim_token: str | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        executable = self.status in {
            FeishuReceiptClaimStatus.NEW,
            FeishuReceiptClaimStatus.RETRYABLE,
        }
        if executable and (self.claim_token is None or not self.claim_token.strip()):
            raise ValueError("an executable Feishu claim requires a claim_token")
        if not executable and self.claim_token is not None:
            raise ValueError("a non-executable Feishu claim cannot expose a claim_token")
        if executable and (self.attempt is None or self.attempt < 1):
            raise ValueError("an executable Feishu claim requires a positive attempt")


class FeishuReceiptStore(Protocol):
    """Persistence port matching the existing ``FeishuEventReceipt`` row fields."""

    def claim(
        self,
        *,
        event_id: str,
        event_type: str,
        payload_sha256: str,
        received_at: datetime,
    ) -> FeishuReceiptClaim:
        """Atomically claim an event and report its durable processing state."""
        ...

    def mark_processed(
        self, *, event_id: str, claim_token: str, processed_at: datetime
    ) -> None:
        """Mark a previously claimed event as processed."""
        ...

    def mark_failed(
        self, *, event_id: str, claim_token: str, failed_at: datetime
    ) -> None:
        """Release a failed claim for a later retry without changing its identity."""
        ...

    def renew_claim(
        self, *, event_id: str, claim_token: str, renewed_at: datetime
    ) -> datetime:
        """Extend a live owned lease and return its new UTC expiry."""
        ...


@dataclass(frozen=True, slots=True)
class FeishuEventOutcome:
    """Secret-free routing result; it intentionally contains no business values."""

    kind: str
    event_id: str | None
    event_type: str | None
    duplicate: bool
    receipt_status: FeishuReceiptClaimStatus | None
    claim_token: str | None
    reference: FeishuEventReference | None
    response: dict[str, str]


class FeishuEventProcessor:
    """Authenticate, validate and deduplicate Feishu callback requests."""

    def __init__(
        self,
        *,
        verifier: FeishuWebhookVerifier,
        verification_token: SecretStr,
        receipts: FeishuReceiptStore,
        decryptor: FeishuPayloadDecryptor | None = None,
    ) -> None:
        if not verification_token.get_secret_value().strip():
            raise ValueError("Feishu verification token must not be blank")
        self._verifier = verifier
        self._verification_token = verification_token
        self._receipts = receipts
        self._decryptor = decryptor

    @property
    def decryptor_configured(self) -> bool:
        """Expose only whether an injected decryptor exists, never its details."""

        return self._decryptor is not None

    def handle(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
        now: datetime,
    ) -> FeishuEventOutcome:
        envelope = _json_object(body)
        if envelope.get("type") == "url_verification":
            return self._url_verification_outcome(envelope)

        normalized_headers = {key.lower(): value for key, value in headers.items()}
        if envelope.get("encrypt") is not None and not _has_signature_header(
            normalized_headers
        ):
            payload = self._decrypt_payload_if_needed(envelope)
            if payload.get("type") == "url_verification":
                return self._url_verification_outcome(payload)
            raise FeishuSignatureError(
                "encrypted Feishu events require signature headers"
            )
        self._verifier.verify(
            timestamp=_required_header(normalized_headers, "x-lark-request-timestamp"),
            nonce=_required_header(normalized_headers, "x-lark-request-nonce"),
            signature=_required_header(normalized_headers, "x-lark-signature"),
            body=body,
            now=now,
        )
        payload = self._decrypt_payload_if_needed(envelope)

        if payload.get("type") == "url_verification":
            return self._url_verification_outcome(payload)

        header = payload.get("header")
        if not isinstance(header, dict):
            raise FeishuEventError("Feishu v2 event requires an object header")
        self._verify_token(header.get("token"))
        event_id = _required_text(header.get("event_id"), field_name="header.event_id")
        event_type = _required_text(header.get("event_type"), field_name="header.event_type")
        reference = parse_feishu_event_reference(payload)
        payload_sha256 = hashlib.sha256(body).hexdigest()
        claim = self._receipts.claim(
            event_id=event_id,
            event_type=event_type,
            payload_sha256=payload_sha256,
            received_at=_utc(now),
        )
        if claim.status is FeishuReceiptClaimStatus.CONFLICT:
            raise FeishuEventError(
                "Feishu event ID conflicts with an earlier payload or event type"
            )
        return FeishuEventOutcome(
            kind="event",
            event_id=event_id,
            event_type=event_type,
            duplicate=claim.status
            in {
                FeishuReceiptClaimStatus.PROCESSED,
                FeishuReceiptClaimStatus.IN_PROGRESS,
            },
            receipt_status=claim.status,
            claim_token=claim.claim_token,
            reference=reference,
            response={},
        )

    def mark_processed(
        self, *, event_id: str, claim_token: str, processed_at: datetime
    ) -> None:
        self._receipts.mark_processed(
            event_id=_required_text(event_id, field_name="event_id"),
            claim_token=_required_text(claim_token, field_name="claim_token"),
            processed_at=_utc(processed_at),
        )

    def mark_failed(
        self, *, event_id: str, claim_token: str, failed_at: datetime
    ) -> None:
        self._receipts.mark_failed(
            event_id=_required_text(event_id, field_name="event_id"),
            claim_token=_required_text(claim_token, field_name="claim_token"),
            failed_at=_utc(failed_at),
        )

    def renew_claim(
        self, *, event_id: str, claim_token: str, renewed_at: datetime
    ) -> datetime:
        return self._receipts.renew_claim(
            event_id=_required_text(event_id, field_name="event_id"),
            claim_token=_required_text(claim_token, field_name="claim_token"),
            renewed_at=_utc(renewed_at),
        )

    def _url_verification_outcome(
        self, payload: Mapping[str, object]
    ) -> FeishuEventOutcome:
        self._verify_token(payload.get("token"))
        challenge = _required_text(payload.get("challenge"), field_name="challenge")
        return FeishuEventOutcome(
            kind="url_verification",
            event_id=None,
            event_type=None,
            duplicate=False,
            receipt_status=None,
            claim_token=None,
            reference=None,
            response={"challenge": challenge},
        )

    def _decrypt_payload_if_needed(
        self, envelope: dict[str, object]
    ) -> dict[str, object]:
        encrypted = envelope.get("encrypt")
        if encrypted is None:
            return envelope
        ciphertext = _required_text(encrypted, field_name="encrypt")
        if self._decryptor is None:
            raise FeishuPayloadDecryptionUnavailable(
                "Feishu payload decryptor is not configured"
            )
        try:
            plaintext = self._decryptor.decrypt(ciphertext)
        except FeishuEventError:
            raise
        except FeishuDecryptorError as exc:
            raise FeishuEventError(str(exc)) from exc
        except Exception as exc:
            raise FeishuEventError("Feishu payload decryption failed") from exc
        return _json_object(plaintext)

    def _verify_token(self, candidate: object) -> None:
        token = _required_text(candidate, field_name="verification token")
        expected = self._verification_token.get_secret_value()
        if not hmac.compare_digest(expected, token):
            raise FeishuEventError("Feishu verification token mismatch")


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name)
    if value is None or not value.strip():
        raise FeishuEventError(f"missing required Feishu header: {name}")
    return value


def _has_signature_header(headers: Mapping[str, str]) -> bool:
    return any(
        headers.get(name, "").strip()
        for name in (
            "x-lark-request-timestamp",
            "x-lark-request-nonce",
            "x-lark-signature",
        )
    )


def _json_object(body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeishuEventError("Feishu callback body must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise FeishuEventError("Feishu callback body must be a JSON object")
    return payload


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeishuEventError(f"Feishu {field_name} must be a nonblank string")
    return value.strip()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Feishu timestamps must include a timezone")
    return value.astimezone(UTC)
