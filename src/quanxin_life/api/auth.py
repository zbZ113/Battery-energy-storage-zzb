"""Secure FastAPI cookie adapter for the transactional authentication service."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import Field, SecretStr, field_validator

from quanxin_life.auth import (
    AccountUnavailableError,
    AuthenticationRequiredError,
    AuthPrincipal,
    AuthService,
    InvalidCredentialsError,
)
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel


class AuthCookieConfig(ContractModel):
    """Explicit browser security policy; production cannot disable HTTPS cookies."""

    environment: Literal["production", "development"]
    allowed_origins: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, origins: tuple[str, ...], info: Any) -> tuple[str, ...]:
        environment = info.data.get("environment")
        normalized: list[str] = []
        for origin in origins:
            if origin == "*":
                raise ValueError("origin wildcard is forbidden")
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                raise ValueError("origin must be an HTTP(S) scheme and authority only")
            if environment == "production" and parsed.scheme != "https":
                raise ValueError("production origin must use HTTPS")
            normalized.append(origin.rstrip("/"))
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed origins must be unique")
        return tuple(normalized)

    @property
    def secure(self) -> bool:
        return self.environment == "production"

    @property
    def cookie_name(self) -> str:
        return "__Host-quanxin_session" if self.secure else "quanxin_dev_session"


class LoginRequest(ContractModel):
    username: str = Field(min_length=1, max_length=320)
    password: SecretStr


class ChangePasswordRequest(ContractModel):
    current_password: SecretStr
    new_password: SecretStr


class AuthSessionResponse(ContractModel):
    user_id: str
    username: str
    role: UserRole
    must_change_password: bool
    expires_at: datetime


class AuthPrincipalResponse(ContractModel):
    user_id: str
    username: str
    role: UserRole
    must_change_password: bool


@dataclass(frozen=True, slots=True)
class AuthHttpAdapter:
    """Router plus reusable dependencies for protecting business endpoints."""

    router: APIRouter
    allowed_origins: tuple[str, ...]
    get_principal: Any
    require_ready_user: Any
    require_roles: Any


def _principal_response(principal: AuthPrincipal) -> AuthPrincipalResponse:
    return AuthPrincipalResponse(
        user_id=principal.user_id,
        username=principal.username,
        role=principal.role,
        must_change_password=principal.must_change_password,
    )


def create_auth_http_adapter(
    service: AuthService,
    config: AuthCookieConfig,
) -> AuthHttpAdapter:
    """Build auth routes without exposing raw tokens outside HttpOnly cookies."""

    router = APIRouter(prefix="/v1/auth", tags=["authentication"])

    def verify_origin(request: Request) -> None:
        if request.headers.get("origin") not in config.allowed_origins:
            raise HTTPException(status_code=403, detail="origin_not_allowed")

    def get_principal(request: Request) -> AuthPrincipal:
        token = request.cookies.get(config.cookie_name)
        if token is None:
            raise HTTPException(status_code=401, detail="authentication_failed")
        try:
            return service.authenticate(token, now=datetime.now(UTC))
        except (AuthenticationRequiredError, AccountUnavailableError) as exc:
            raise HTTPException(status_code=401, detail="authentication_failed") from exc

    def require_ready_user(
        principal: Annotated[AuthPrincipal, Depends(get_principal)],
    ) -> AuthPrincipal:
        if principal.must_change_password:
            raise HTTPException(status_code=403, detail="credential_change_required")
        return principal

    def require_roles(allowed_roles: set[UserRole]) -> Any:
        frozen_roles = frozenset(allowed_roles)
        if not frozen_roles:
            raise ValueError("at least one allowed role is required")

        def dependency(
            principal: Annotated[AuthPrincipal, Depends(require_ready_user)],
        ) -> AuthPrincipal:
            if principal.role not in frozen_roles:
                raise HTTPException(status_code=403, detail="role_not_allowed")
            return principal

        return dependency

    def set_session_cookie(response: Response, *, grant: Any, now: datetime) -> None:
        max_age = max(0, int((grant.expires_at - now).total_seconds()))
        response.set_cookie(
            key=config.cookie_name,
            value=grant.raw_token.get_secret_value(),
            max_age=max_age,
            expires=grant.expires_at,
            path="/",
            secure=config.secure,
            httponly=True,
            samesite="strict",
        )
        response.headers["Cache-Control"] = "no-store"

    def clear_session_cookie(response: Response) -> None:
        response.delete_cookie(
            key=config.cookie_name,
            path="/",
            secure=config.secure,
            httponly=True,
            samesite="strict",
        )
        response.headers["Cache-Control"] = "no-store"

    @router.post("/login", response_model=AuthSessionResponse)
    def login(payload: LoginRequest, request: Request, response: Response) -> Any:
        verify_origin(request)
        now = datetime.now(UTC)
        try:
            grant = service.login(
                username=payload.username,
                password=payload.password.get_secret_value(),
                now=now,
            )
        except (InvalidCredentialsError, AccountUnavailableError) as exc:
            raise HTTPException(status_code=401, detail="authentication_failed") from exc
        set_session_cookie(response, grant=grant, now=now)
        return AuthSessionResponse(
            **_principal_response(grant.principal).model_dump(),
            expires_at=grant.expires_at,
        )

    @router.get("/me", response_model=AuthPrincipalResponse)
    def me(
        response: Response,
        principal: Annotated[AuthPrincipal, Depends(get_principal)],
    ) -> Any:
        response.headers["Cache-Control"] = "no-store"
        return _principal_response(principal)

    @router.post("/change-password", response_model=AuthSessionResponse)
    def change_password(
        payload: ChangePasswordRequest,
        request: Request,
        response: Response,
    ) -> Any:
        verify_origin(request)
        raw_token = request.cookies.get(config.cookie_name)
        if raw_token is None:
            raise HTTPException(status_code=401, detail="authentication_failed")
        now = datetime.now(UTC)
        try:
            grant = service.change_password(
                raw_token,
                current_password=payload.current_password.get_secret_value(),
                new_password=payload.new_password.get_secret_value(),
                now=now,
            )
        except (
            InvalidCredentialsError,
            AuthenticationRequiredError,
            AccountUnavailableError,
        ) as exc:
            raise HTTPException(status_code=401, detail="authentication_failed") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_new_password") from exc
        set_session_cookie(response, grant=grant, now=now)
        return AuthSessionResponse(
            **_principal_response(grant.principal).model_dump(),
            expires_at=grant.expires_at,
        )

    @router.post("/logout", status_code=204)
    def logout(request: Request, response: Response) -> Response:
        verify_origin(request)
        service.logout(request.cookies.get(config.cookie_name), now=datetime.now(UTC))
        clear_session_cookie(response)
        response.status_code = 204
        return response

    return AuthHttpAdapter(
        router=router,
        allowed_origins=config.allowed_origins,
        get_principal=get_principal,
        require_ready_user=require_ready_user,
        require_roles=require_roles,
    )


__all__ = ["AuthCookieConfig", "AuthHttpAdapter", "create_auth_http_adapter"]
