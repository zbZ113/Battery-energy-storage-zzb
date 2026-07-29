from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_acr_release_is_manual_private_and_versioned() -> None:
    workflow = _read(".github/workflows/publish-acr.yml")

    assert "workflow_dispatch:" in workflow
    assert "release_tag:" in workflow
    assert "public_origin:" in workflow
    assert re.search(r"(?m)^  push:", workflow) is None
    assert re.search(r"(?m)^  pull_request:", workflow) is None
    assert "permissions:" in workflow
    assert "contents: read" in workflow
    assert "docker/login-action@" in workflow
    assert "secrets.ACR_REGISTRY" in workflow
    assert "secrets.ACR_USERNAME" in workflow
    assert "secrets.ACR_PASSWORD" in workflow
    assert "vars.ACR_NAMESPACE" in workflow
    assert ":latest" not in workflow


def test_acr_release_publishes_application_and_owned_infrastructure_images() -> None:
    workflow = _read(".github/workflows/publish-acr.yml")

    for repository in (
        "quanxin-backend",
        "quanxin-frontend",
        "quanxin-postgres",
        "quanxin-redis",
        "quanxin-nginx",
    ):
        assert repository in workflow

    assert "deploy/Dockerfile.backend" in workflow
    assert "frontend/Dockerfile" in workflow
    assert "pgvector/pgvector:0.8.1-pg16" in workflow
    assert "redis:7.4.2-alpine" in workflow
    assert "nginx:1.27.4-alpine" in workflow
    assert "NEXT_PUBLIC_API_BASE_URL" in workflow
    assert "https://" in workflow


def test_backend_image_contains_runtime_code_but_not_local_results() -> None:
    dockerfile = _read("deploy/Dockerfile.backend")
    dockerignore = _read(".dockerignore")

    assert "FROM python:3.11-slim" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY migrations ./migrations" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert "COPY deploy ./deploy" in dockerfile
    assert "alembic.ini" in dockerfile
    assert "server-results/" in dockerignore
    assert "runs/" in dockerignore
    assert "artifacts/" in dockerignore
    assert "*.pt" in dockerignore
    assert "*.pth" in dockerignore


def test_backend_image_installs_the_llm_runtime_extra() -> None:
    dockerfile = _read("deploy/Dockerfile.backend")
    editable_install = re.search(r'"\.\[([^\]]+)\]"', dockerfile)

    assert editable_install is not None
    installed_extras = {
        value.strip() for value in editable_install.group(1).split(",")
    }
    assert "llm" in installed_extras


def test_frontend_image_uses_locked_standalone_next_build() -> None:
    dockerfile = _read("frontend/Dockerfile")
    next_config = _read("frontend/next.config.ts")

    assert "pnpm install --frozen-lockfile" in dockerfile
    assert "pnpm build" in dockerfile
    assert "NEXT_PUBLIC_API_BASE_URL" in dockerfile
    assert ".next/standalone" in dockerfile
    assert ".next/static" in dockerfile
    assert 'NEXT_OUTPUT === "standalone"' in next_config
    assert "ENV NEXT_OUTPUT=standalone" in dockerfile
