import pytest

from quanxin_life.auth.passwords import (
    Argon2idConfig,
    Argon2idPasswordHasher,
    PasswordPolicy,
)


@pytest.fixture
def hasher() -> Argon2idPasswordHasher:
    return Argon2idPasswordHasher(
        config=Argon2idConfig(
            time_cost=1,
            memory_cost_kib=8 * 1024,
            parallelism=1,
            hash_len=16,
            salt_len=16,
        )
    )


def test_argon2id_hash_never_contains_plaintext_and_verifies(
    hasher: Argon2idPasswordHasher,
) -> None:
    password = "correct horse battery staple"

    credential_hash = hasher.hash_password(password)

    assert credential_hash.startswith("$argon2id$")
    assert password not in credential_hash
    assert hasher.verify_password(credential_hash, password) is True
    assert hasher.verify_password(credential_hash, "wrong password") is False


def test_default_argon2id_parameters_match_interactive_server_budget() -> None:
    config = Argon2idConfig()

    assert config.time_cost == 2
    assert config.memory_cost_kib == 19 * 1024
    assert config.parallelism == 1
    assert config.hash_len == 32
    assert config.salt_len == 16


def test_invalid_or_non_argon2id_hash_fails_closed(hasher: Argon2idPasswordHasher) -> None:
    assert hasher.verify_password("not-a-password-hash", "any password") is False
    assert hasher.verify_password("$argon2i$v=19$m=8,t=1,p=1$bad$bad", "password") is False


def test_password_policy_rejects_short_or_excessively_long_passwords() -> None:
    policy = PasswordPolicy(min_length=12, max_length=64)

    with pytest.raises(ValueError, match="at least 12"):
        policy.validate_password("short")
    with pytest.raises(ValueError, match="at most 64"):
        policy.validate_password("x" * 65)


def test_password_policy_rejects_blank_edges_and_control_characters() -> None:
    policy = PasswordPolicy()

    with pytest.raises(ValueError, match="leading or trailing"):
        policy.validate_password(" valid-password-123 ")
    with pytest.raises(ValueError, match="control"):
        policy.validate_password("valid-password\n123")


def test_password_policy_accepts_long_passphrase_without_composition_tricks() -> None:
    policy = PasswordPolicy()

    assert policy.max_length == 128
    assert policy.validate_password("泉芯智寿 这是一个足够长的口令") == (
        "泉芯智寿 这是一个足够长的口令"
    )


def test_hasher_reports_when_parameters_need_upgrade(
    hasher: Argon2idPasswordHasher,
) -> None:
    old_hash = hasher.hash_password("a sufficiently long password")
    stronger = Argon2idPasswordHasher(
        config=Argon2idConfig(
            time_cost=2,
            memory_cost_kib=8 * 1024,
            parallelism=1,
            hash_len=16,
            salt_len=16,
        )
    )

    assert hasher.needs_rehash(old_hash) is False
    assert stronger.needs_rehash(old_hash) is True
