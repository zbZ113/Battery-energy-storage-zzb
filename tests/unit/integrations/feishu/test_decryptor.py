from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pydantic import SecretStr

from quanxin_life.integrations.feishu.decryptor import (
    FeishuAesCbcDecryptor,
    FeishuDecryptorError,
)

ENCRYPT_KEY = "reviewed-encrypt-key"
PLAINTEXT = b'{"type":"url_verification","token":"token","challenge":"challenge"}'


def _encrypted_payload(*, plaintext: bytes = PLAINTEXT, key: str = ENCRYPT_KEY) -> str:
    key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    iv = b"i" * 16
    padding = 16 - (len(plaintext) % 16)
    padded = plaintext + bytes([padding]) * padding
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted = iv + encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(encrypted).decode("ascii")


def test_decryptor_recovers_feishu_plaintext_envelope() -> None:
    decryptor = FeishuAesCbcDecryptor(SecretStr(ENCRYPT_KEY))

    plaintext = decryptor.decrypt(_encrypted_payload())

    assert json.loads(plaintext) == json.loads(PLAINTEXT)


def test_decryptor_rejects_wrong_key_without_leaking_plaintext() -> None:
    decryptor = FeishuAesCbcDecryptor(SecretStr("wrong-key"))

    with pytest.raises(FeishuDecryptorError, match="decrypt"):
        decryptor.decrypt(_encrypted_payload())

    assert "challenge" not in repr(decryptor)


@pytest.mark.parametrize("value", ["", "%%%", base64.b64encode(b"short").decode()])
def test_decryptor_rejects_malformed_ciphertext(value: str) -> None:
    decryptor = FeishuAesCbcDecryptor(SecretStr(ENCRYPT_KEY))

    with pytest.raises(FeishuDecryptorError):
        decryptor.decrypt(value)
