from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import User

ORIGIN = "https://app.example.test"
USERNAME = "student@example.test"
TEMPORARY_PASSWORD = "temporary passphrase 2026"
NEW_PASSWORD = "new private passphrase 2026"


def _build_auth_client(
    tmp_path: Path,
    *,
    role: UserRole,
    must_change_credential: bool,
    database_name: str,
) -> TestClient:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / database_name}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    hasher = Argon2idPasswordHasher()
    now = datetime.now(UTC)
    with session_factory.begin() as session:
        session.add(
            User(
                id=str(uuid4()),
                username=USERNAME,
                credential_hash=hasher.hash_password(TEMPORARY_PASSWORD),
                must_change_credential=must_change_credential,
                role=role.value,
                status=UserStatus.ACTIVE.value,
                created_at=now,
                updated_at=now,
            )
        )
    service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    adapter = create_auth_http_adapter(
        service,
        AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,)),
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(), auth_adapter=adapter
    )
    return TestClient(app, base_url="https://api.example.test")


@pytest.fixture
def auth_client(tmp_path: Path) -> TestClient:
    return _build_auth_client(
        tmp_path,
        role=UserRole.MEMBER,
        must_change_credential=True,
        database_name="api-auth.sqlite3",
    )


@pytest.fixture
def judge_client(tmp_path: Path) -> TestClient:
    return _build_auth_client(
        tmp_path,
        role=UserRole.JUDGE,
        must_change_credential=False,
        database_name="judge-auth.sqlite3",
    )


def test_login_sets_a_host_only_secure_cookie_without_returning_the_token(
    auth_client: TestClient,
) -> None:
    response = auth_client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": TEMPORARY_PASSWORD},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert set(response.json()) == {
        "user_id",
        "username",
        "role",
        "must_change_password",
        "expires_at",
    }
    assert response.json()["must_change_password"] is True
    assert "token" not in response.text.casefold()
    assert "hash" not in response.text.casefold()
    cookie = response.headers["set-cookie"]
    assert "__Host-quanxin_session=" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/" in cookie
    assert "Domain=" not in cookie


def test_login_rejects_untrusted_or_missing_origin(auth_client: TestClient) -> None:
    for headers in ({}, {"Origin": "https://attacker.example"}):
        response = auth_client.post(
            "/v1/auth/login",
            headers=headers,
            json={"username": USERNAME, "password": TEMPORARY_PASSWORD},
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "origin_not_allowed"


def test_browser_cors_preflight_allows_only_the_configured_frontend(
    auth_client: TestClient,
) -> None:
    allowed = auth_client.options(
        "/v1/auth/login",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert allowed.headers["access-control-allow-credentials"] == "true"

    blocked = auth_client.options(
        "/v1/auth/login",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert blocked.status_code == 400
    assert "access-control-allow-origin" not in blocked.headers


def test_me_change_password_and_logout_use_the_server_side_session(
    auth_client: TestClient,
) -> None:
    login = auth_client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": TEMPORARY_PASSWORD},
    )
    assert login.status_code == 200

    me = auth_client.get("/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == USERNAME
    assert me.json()["must_change_password"] is True

    changed = auth_client.post(
        "/v1/auth/change-password",
        headers={"Origin": ORIGIN},
        json={
            "current_password": TEMPORARY_PASSWORD,
            "new_password": NEW_PASSWORD,
        },
    )
    assert changed.status_code == 200
    assert changed.json()["must_change_password"] is False
    assert auth_client.get("/v1/auth/me").status_code == 200

    logout = auth_client.post("/v1/auth/logout", headers={"Origin": ORIGIN})
    assert logout.status_code == 204
    assert logout.headers["cache-control"] == "no-store"
    assert auth_client.get("/v1/auth/me").status_code == 401


def test_business_routes_require_login_and_completed_first_password_change(
    auth_client: TestClient,
) -> None:
    assert auth_client.get("/v1/tools").status_code == 401
    login = auth_client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": TEMPORARY_PASSWORD},
    )
    assert login.status_code == 200
    blocked = auth_client.get("/v1/tools")
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "credential_change_required"

    changed = auth_client.post(
        "/v1/auth/change-password",
        headers={"Origin": ORIGIN},
        json={
            "current_password": TEMPORARY_PASSWORD,
            "new_password": NEW_PASSWORD,
        },
    )
    assert changed.status_code == 200
    assert auth_client.get("/v1/tools").status_code == 200
    cross_site_write = auth_client.post("/v1/tools/validate_battery_data", json={})
    assert cross_site_write.status_code == 403
    assert cross_site_write.json()["detail"] == "origin_not_allowed"


def test_judge_can_inspect_tools_but_cannot_invoke_arbitrary_domain_tools(
    judge_client: TestClient,
) -> None:
    login = judge_client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": TEMPORARY_PASSWORD},
    )
    assert login.status_code == 200
    assert judge_client.get("/v1/tools").status_code == 200

    blocked = judge_client.post(
        "/v1/tools/validate_battery_data",
        headers={"Origin": ORIGIN},
        json={},
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "role_not_allowed"


def test_auth_failures_are_generic_and_never_echo_credentials(
    auth_client: TestClient,
) -> None:
    response = auth_client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": USERNAME, "password": "wrong passphrase 2026"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "authentication_failed"
    assert "wrong passphrase" not in response.text


def test_production_cookie_configuration_cannot_be_downgraded() -> None:
    config = AuthCookieConfig(environment="production", allowed_origins=(ORIGIN,))
    assert config.cookie_name == "__Host-quanxin_session"
    assert config.secure is True

    development = AuthCookieConfig(
        environment="development",
        allowed_origins=("http://localhost:3000",),
    )
    assert development.cookie_name == "quanxin_dev_session"
    assert development.secure is False


@pytest.mark.parametrize(
    "origin",
    ["*", "https://app.example.test/path", "file:///tmp/app", "javascript:alert(1)"],
)
def test_cookie_configuration_rejects_unsafe_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="origin"):
        AuthCookieConfig(environment="production", allowed_origins=(origin,))
