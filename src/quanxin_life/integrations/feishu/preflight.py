"""Secret-free configuration completeness checks for Feishu/Aily deployment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

FEISHU_REQUIRED_ENVIRONMENT = (
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_BITABLE_APP_TOKEN",
    "FEISHU_BITABLE_TABLE_ID",
    "FEISHU_CONNECTOR_API_KEY",
    "FEISHU_ENCRYPT_KEY",
    "FEISHU_TEST_CHAT_ID",
    "FEISHU_VERIFICATION_TOKEN",
    "QUANXIN_EXTERNAL_HTTPS_BASE_URL",
)


@dataclass(frozen=True, slots=True)
class FeishuPreflightReport:
    mode: str
    ready: bool
    configured: tuple[str, ...]
    missing: tuple[str, ...]
    invalid: tuple[str, ...]


def run_feishu_preflight(
    environment: Mapping[str, str],
    *,
    production: bool,
) -> FeishuPreflightReport:
    """Check names and value shapes without retaining or returning secrets."""

    if not production:
        return FeishuPreflightReport(
            mode="sandbox",
            ready=True,
            configured=(),
            missing=(),
            invalid=(),
        )
    configured = tuple(
        sorted(
            name
            for name in FEISHU_REQUIRED_ENVIRONMENT
            if _configured(environment.get(name))
        )
    )
    missing = tuple(
        sorted(set(FEISHU_REQUIRED_ENVIRONMENT) - set(configured))
    )
    invalid: list[str] = []
    external_url = environment.get("QUANXIN_EXTERNAL_HTTPS_BASE_URL")
    if (
        isinstance(external_url, str)
        and external_url.strip()
        and not _is_external_https_url(external_url)
    ):
        invalid.append("QUANXIN_EXTERNAL_HTTPS_BASE_URL_REQUIRES_HTTPS")
    return FeishuPreflightReport(
        mode="production",
        ready=not missing and not invalid,
        configured=configured,
        missing=missing,
        invalid=tuple(invalid),
    )


def _configured(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_external_https_url(value: str) -> bool:
    parsed = urlsplit(value.strip())
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )


__all__ = [
    "FEISHU_REQUIRED_ENVIRONMENT",
    "FeishuPreflightReport",
    "run_feishu_preflight",
]
