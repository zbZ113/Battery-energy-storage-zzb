from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    try:
        return (ROOT / relative_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AssertionError(f"missing local deployment file: {relative_path}") from None


def test_local_frontend_build_requires_explicit_loopback_profile() -> None:
    local_dockerfile = _read("frontend/Dockerfile.local")
    production_dockerfile = _read("frontend/Dockerfile")

    assert "NEXT_PUBLIC_RUNTIME_PROFILE=local" in local_dockerfile
    assert "NEXT_PUBLIC_API_BASE_URL=http://localhost:8080" in local_dockerfile
    assert "pnpm build" in local_dockerfile
    assert "NEXT_PUBLIC_RUNTIME_PROFILE" in production_dockerfile
    assert "local profile is forbidden" in production_dockerfile


def test_frontend_docker_context_excludes_host_build_artifacts() -> None:
    dockerignore = _read("frontend/.dockerignore")

    for excluded in (
        "node_modules",
        ".next",
        "tests",
        "*.tsbuildinfo",
        "coverage",
        "test-results",
    ):
        assert excluded in dockerignore.splitlines()


def test_local_gateway_has_no_tls_redirect_or_certificate_dependency() -> None:
    dockerfile = _read("deploy/Dockerfile.gateway.local")
    nginx = _read("deploy/nginx/local.conf")

    assert "COPY deploy/nginx/local.conf" in dockerfile
    assert "listen 80;" in nginx
    assert "location /v1/" in nginx
    assert "proxy_pass http://api:8000;" in nginx
    assert "proxy_buffering off;" in nginx
    assert "proxy_pass http://frontend:3000;" in nginx
    assert "X-Forwarded-Proto http" in nginx
    for forbidden in ("listen 443", "ssl_certificate", "return 308", "letsencrypt"):
        assert forbidden not in nginx


def test_local_compose_exposes_only_the_loopback_gateway() -> None:
    compose = _read("deploy/local.compose.yaml")

    for service in (
        "postgres",
        "redis",
        "migrate",
        "api",
        "worker",
        "frontend",
        "gateway",
    ):
        assert re.search(rf"(?m)^  {service}:\s*$", compose)

    assert '"127.0.0.1:8080:80"' in compose
    assert compose.count("ports:") == 1
    for forbidden in ('"5432:', '"6379:', '"8000:', '"3000:'):
        assert forbidden not in compose

    assert "pgvector/pgvector:0.8.1-pg16@${POSTGRES_IMAGE_DIGEST}" in compose
    assert "redis:7.4.2-alpine@${REDIS_IMAGE_DIGEST}" in compose
    assert "--appendonly" in compose
    assert '"yes"' in compose
    assert "redis-cli --user quanxin ping" in compose
    assert "deploy.local_api:app" in compose
    assert "deploy.local_worker:app" in compose
    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose
    assert "deploy.local_migrate" in compose
    assert "agent-runs,advanced-calibration,report-exports" in compose


def test_local_compose_keeps_runtime_inputs_and_secrets_outside_the_repo() -> None:
    compose = _read("deploy/local.compose.yaml")
    template = _read("deploy/local.env.example")

    for name in (
        "LOCAL_RUNTIME_ROOT",
        "QUANXIN_SECRETS_ROOT",
        "POSTGRES_IMAGE_DIGEST",
        "REDIS_IMAGE_DIGEST",
        "DEPLOYMENT_REGISTRY_ID",
        "PUBLIC_ORIGIN",
    ):
        assert re.search(rf"(?m)^{name}=\s*$", template)
        assert f"${{{name}}}" in compose

    for forbidden in ("postgresql://", "redis://:", "password=", "BEGIN PRIVATE KEY"):
        assert forbidden not in template


def test_production_deployment_does_not_reference_local_profiles() -> None:
    production_compose = _read("deploy/competition.compose.yaml")
    production_gateway = _read("deploy/Dockerfile.gateway")

    assert "local_api" not in production_compose
    assert "local_worker" not in production_compose
    assert "Dockerfile.gateway.local" not in production_compose
    assert "local.conf" not in production_gateway


def test_local_migration_uses_loopback_runtime_settings() -> None:
    local_migrate = _read("deploy/local_migrate.py")

    assert "LocalCompetitionRuntimeSettings.from_environment(os.environ)" in (
        local_migrate
    )
    assert "run_database_migrations" in local_migrate
