from __future__ import annotations

from quanxin_life.integrations.feishu.preflight import (
    FEISHU_REQUIRED_ENVIRONMENT,
    run_feishu_preflight,
)


def _production_environment() -> dict[str, str]:
    values = {name: f"configured-{name.lower()}" for name in FEISHU_REQUIRED_ENVIRONMENT}
    values["QUANXIN_EXTERNAL_HTTPS_BASE_URL"] = "https://battery.example.invalid"
    return values


def test_production_preflight_reports_missing_names_without_secret_values() -> None:
    environment = _production_environment()
    secret = environment.pop("FEISHU_APP_SECRET")

    report = run_feishu_preflight(environment, production=True)

    assert report.ready is False
    assert report.missing == ("FEISHU_APP_SECRET",)
    assert secret not in repr(report)


def test_production_preflight_requires_an_https_external_base_url() -> None:
    environment = _production_environment()
    environment["QUANXIN_EXTERNAL_HTTPS_BASE_URL"] = "http://127.0.0.1:8000"

    report = run_feishu_preflight(environment, production=True)

    assert report.ready is False
    assert report.invalid == ("QUANXIN_EXTERNAL_HTTPS_BASE_URL_REQUIRES_HTTPS",)


def test_complete_production_preflight_returns_names_only() -> None:
    environment = _production_environment()

    report = run_feishu_preflight(environment, production=True)

    assert report.ready is True
    assert report.missing == ()
    assert report.invalid == ()
    assert report.configured == tuple(sorted(FEISHU_REQUIRED_ENVIRONMENT))
    assert not any(value in repr(report) for value in environment.values())


def test_sandbox_preflight_needs_no_real_feishu_credentials() -> None:
    report = run_feishu_preflight({}, production=False)

    assert report.ready is True
    assert report.mode == "sandbox"
    assert report.configured == ()
