from quanxin_life.core.enums import SessionStatus, UserStatus


def test_authentication_status_values_are_stable() -> None:
    assert UserStatus.ACTIVE.value == "ACTIVE"
    assert UserStatus.DISABLED.value == "DISABLED"
    assert SessionStatus.ACTIVE.value == "ACTIVE"
    assert SessionStatus.REVOKED.value == "REVOKED"
