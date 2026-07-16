from pathlib import Path


def test_docker_build_context_excludes_secrets_and_untrusted_data() -> None:
    path = Path(".dockerignore")
    assert path.is_file(), "repository Docker builds require a root .dockerignore"
    rules = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    required = {
        ".env*",
        ".git/",
        ".venv/",
        "secrets/",
        "deploy/secrets/",
        "data/",
        "artifacts/",
        "models/",
        "reports/generated/",
        "*.pkl",
        "*.pickle",
        "*.joblib",
        "*.pt",
        "*.pth",
        "*.secret",
    }
    assert required <= rules
