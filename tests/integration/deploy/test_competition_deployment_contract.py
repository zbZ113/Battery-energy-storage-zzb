from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_competition_compose_uses_only_immutable_private_acr_images() -> None:
    compose = _read("deploy/competition.compose.yaml")

    for service, repository in (
        ("postgres", "quanxin-postgres"),
        ("redis", "quanxin-redis"),
        ("api", "quanxin-backend"),
        ("worker", "quanxin-backend"),
        ("frontend", "quanxin-frontend"),
        ("gateway", "quanxin-nginx"),
    ):
        assert re.search(rf"(?m)^  {service}:\s*$", compose)
        assert (
            "${ACR_REGISTRY}/${ACR_NAMESPACE}/"
            f"{repository}:${{RELEASE_TAG}}"
        ) in compose

    assert ":latest" not in compose
    assert "build:" not in compose


def test_only_https_gateway_ports_are_published_to_the_host() -> None:
    compose = _read("deploy/competition.compose.yaml")

    assert '"80:80"' in compose
    assert '"443:443"' in compose
    for forbidden in ('"3000:', '"5432:', '"6379:', '"8000:'):
        assert forbidden not in compose
    assert compose.count("ports:") == 1
    assert "internal: true" in compose


def test_runtime_services_use_secret_files_and_fail_closed_dependencies() -> None:
    compose = _read("deploy/competition.compose.yaml")

    for secret in (
        "postgres_password",
        "database_url",
        "redis_acl",
        "redis_url",
    ):
        assert f"- {secret}" in compose
        assert re.search(rf"(?m)^  {secret}:\s*$", compose)

    for setting in (
        "QUANXIN_DATABASE_URL_FILE=/run/secrets/database_url",
        "QUANXIN_REDIS_URL_FILE=/run/secrets/redis_url",
        "QUANXIN_TRUSTED_ORIGIN=${PUBLIC_ORIGIN}",
        "QUANXIN_DEPLOYMENT_REGISTRY_ID=${DEPLOYMENT_REGISTRY_ID}",
        "QUANXIN_CALIBRATION_EVIDENCE_ROOT=/srv/quanxin/calibration-evidence",
        "QUANXIN_AGENT_POLICY_FILE=/srv/quanxin/policies/advanced-agent.json",
    ):
        assert setting in compose

    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose
    assert "deploy.migrate" in compose
    assert "deploy.competition_api:app" in compose
    assert "deploy.competition_worker:app" in compose
    assert "agent-runs,advanced-calibration" in compose


def test_competition_compose_mounts_certificates_and_evidence_with_safe_modes() -> None:
    compose = _read("deploy/competition.compose.yaml")

    for mount in (
        "${LETSENCRYPT_ROOT:-/etc/letsencrypt}:/etc/letsencrypt:ro",
        "${ACME_WEBROOT:-/srv/quanxin/acme}:/var/www/certbot:ro",
        "${DEPLOYMENT_REGISTRY_ROOT}:/srv/quanxin/deployment-registry:ro",
        "${CALIBRATION_EVIDENCE_ROOT}:/srv/quanxin/calibration-evidence:ro",
    ):
        assert mount in compose

    assert "no-new-privileges:true" in compose
    assert "read_only: true" in compose
    assert "tmpfs:" in compose
    assert "mem_limit:" in compose


def test_gateway_configuration_comes_from_the_immutable_image() -> None:
    compose = _read("deploy/competition.compose.yaml")

    assert "./nginx/competition.conf:/etc/nginx/conf.d/default.conf:ro" not in compose
    assert "${LETSENCRYPT_ROOT:-/etc/letsencrypt}:/etc/letsencrypt:ro" in compose
    assert "${ACME_WEBROOT:-/srv/quanxin/acme}:/var/www/certbot:ro" in compose


def test_nginx_terminates_tls_serves_acme_and_proxies_same_origin_routes() -> None:
    nginx = _read("deploy/nginx/competition.conf")

    assert "listen 80;" in nginx
    assert "listen 443 ssl;" in nginx
    assert "location ^~ /.well-known/acme-challenge/" in nginx
    assert "root /var/www/certbot;" in nginx
    assert "return 308 https://$host$request_uri;" in nginx
    assert (
        "ssl_certificate /etc/letsencrypt/live/quanxin-ecs-ip/fullchain.pem;"
        in nginx
    )
    assert (
        "ssl_certificate_key /etc/letsencrypt/live/quanxin-ecs-ip/privkey.pem;"
        in nginx
    )
    assert "location /v1/" in nginx
    assert "proxy_pass http://api:8000;" in nginx
    assert "location / {" in nginx
    assert "proxy_pass http://frontend:3000;" in nginx
    assert "proxy_buffering off;" in nginx


def test_deployment_environment_template_contains_identities_not_secrets() -> None:
    template = _read("deploy/competition.env.example")

    for name in (
        "ACR_REGISTRY",
        "ACR_NAMESPACE",
        "RELEASE_TAG",
        "PUBLIC_ORIGIN",
        "DEPLOYMENT_REGISTRY_ID",
        "DEPLOYMENT_REGISTRY_ROOT",
        "CALIBRATION_EVIDENCE_ROOT",
        "QUANXIN_SECRETS_ROOT",
    ):
        assert re.search(rf"(?m)^{name}=\s*$", template)

    for forbidden in (
        "password=",
        "postgresql://",
        "redis://:",
        "BEGIN PRIVATE KEY",
    ):
        assert forbidden not in template


def test_production_entrypoints_use_the_strict_competition_runtime() -> None:
    api = _read("deploy/competition_api.py")
    worker = _read("deploy/competition_worker.py")

    for source in (api, worker):
        assert "CompetitionRuntimeSettings.from_environment(os.environ)" in source
        assert "create_competition_runtime(settings)" in source
        assert "foundation_api" not in source
    assert "app = runtime.http_app" in api
    assert "app = runtime.celery_app" in worker
