from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    try:
        return (ROOT / relative_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AssertionError(f"missing local runtime script: {relative_path}") from None


def test_prepare_runtime_generates_external_secrets_without_printing_values() -> None:
    script = _read("scripts/local/prepare_runtime.ps1")

    for required in (
        "D:\\QuanxinRuntime",
        "RandomNumberGenerator",
        "postgres_password",
        "database_url",
        "redis_password",
        "redis_acl",
        "redis_url",
        "user default off",
        "user quanxin on",
        "pgvector/pgvector:0.8.1-pg16",
        "redis:7.4.2-alpine",
        "RepoDigests",
        "GetDirectoryName($dockerCli)",
        "$env:PATH",
        "local.env",
        "http://localhost:8080",
    ):
        assert required in script

    assert "Get-Content" not in script
    assert "Write-Host $" not in script
    assert "Write-Output $postgres" not in script
    assert "Write-Output $redis" not in script


def test_bootstrap_admin_uses_hidden_input_and_always_removes_temporary_file() -> None:
    script = _read("scripts/local/bootstrap_admin.ps1")

    assert "Read-Host -AsSecureString" in script
    assert "QUANXIN_BOOTSTRAP_ADMIN_PASSWORD_FILE" in script
    assert "python -m deploy.bootstrap_admin" in script
    assert "finally" in script
    assert "Remove-Item" in script
    assert "SecureStringToBSTR" in script
    assert "Write-Output $plain" not in script
