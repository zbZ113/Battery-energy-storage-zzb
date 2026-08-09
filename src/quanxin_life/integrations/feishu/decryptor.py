"""Optional Feishu AES-CBC event decryptor with fail-closed validation."""

from __future__ import annotations

import base64
import binascii
import hashlib

from pydantic import SecretStr

MAX_FEISHU_CIPHERTEXT_BYTES = 4 * 1024 * 1024


class FeishuDecryptorError(ValueError):
    """Raised when an encrypted Feishu envelope cannot be safely decrypted."""


class FeishuAesCbcDecryptor:
    """Decrypt Feishu's base64 AES-256-CBC envelope.

    Feishu derives a 32-byte AES key from SHA-256 of the configured Encrypt Key,
    prefixes the ciphertext with a random 16-byte IV, and PKCS#7-pads the JSON
    plaintext before AES-CBC encryption.
    """

    def __init__(self, encrypt_key: SecretStr) -> None:
        value = encrypt_key.get_secret_value().strip()
        if not value:
            raise ValueError("Feishu Encrypt Key must not be blank")
        self._key = value

    def decrypt(self, encrypted: str) -> bytes:
        if not isinstance(encrypted, str) or not encrypted.strip():
            raise FeishuDecryptorError("Feishu encrypted payload is blank")
        encoded = encrypted.strip()
        if len(encoded) > MAX_FEISHU_CIPHERTEXT_BYTES * 2:
            raise FeishuDecryptorError("Feishu encrypted payload is too large")
        try:
            ciphertext = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FeishuDecryptorError("Feishu encrypted payload is not valid base64") from exc
        if (
            len(ciphertext) < 32
            or len(ciphertext) > MAX_FEISHU_CIPHERTEXT_BYTES
            or len(ciphertext) % 16 != 0
        ):
            raise FeishuDecryptorError("Feishu encrypted payload has an invalid length")

        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise FeishuDecryptorError(
                "Feishu encrypted callbacks require the cryptography dependency"
            ) from exc

        key = hashlib.sha256(self._key.encode("utf-8")).digest()
        try:
            iv = ciphertext[:16]
            decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
            padded = decryptor.update(ciphertext[16:]) + decryptor.finalize()
            plaintext = _unpad(padded)
            if not plaintext:
                raise FeishuDecryptorError("Feishu plaintext is empty")
            plaintext.decode("utf-8")
            return plaintext
        except FeishuDecryptorError:
            raise
        except Exception as exc:
            raise FeishuDecryptorError("Feishu encrypted payload decrypt failed") from exc


def _unpad(value: bytes) -> bytes:
    if not value:
        raise FeishuDecryptorError("Feishu encrypted payload decrypt failed")
    padding = value[-1]
    if padding < 1 or padding > 16 or value[-padding:] != bytes([padding]) * padding:
        raise FeishuDecryptorError("Feishu encrypted payload decrypt failed")
    return value[:-padding]


__all__ = [
    "MAX_FEISHU_CIPHERTEXT_BYTES",
    "FeishuAesCbcDecryptor",
    "FeishuDecryptorError",
]
