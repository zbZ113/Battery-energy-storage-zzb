import hashlib

import pytest

from quanxin_life.auth.tokens import SessionTokenFactory, hash_session_token


def test_generated_session_token_is_opaque_redacted_and_hashable() -> None:
    token = SessionTokenFactory().issue()
    raw_token = token.get_secret_value()

    assert len(raw_token) >= 43
    assert raw_token not in repr(token)
    assert hash_session_token(token) == hashlib.sha256(raw_token.encode("ascii")).hexdigest()
    assert raw_token not in hash_session_token(token)


@pytest.mark.parametrize("invalid_token", ["", "short", "contains spaces" + "x" * 40])
def test_hash_session_token_rejects_invalid_external_token(invalid_token: str) -> None:
    with pytest.raises(ValueError, match="session token"):
        hash_session_token(invalid_token)


def test_token_factory_fails_closed_when_generator_returns_unsafe_value() -> None:
    factory = SessionTokenFactory(generator=lambda _size: "not-random-enough")

    with pytest.raises(RuntimeError, match="generator"):
        factory.issue()
