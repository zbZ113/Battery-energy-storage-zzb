from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_windows_installer_keeps_movable_docker_data_on_d_drive() -> None:
    script = (ROOT / "scripts/local/install_docker_desktop.ps1").read_text(
        encoding="utf-8"
    )

    for required in (
        "D:\\QuanxinRuntime",
        "docker-desktop",
        "docker-wsl",
        "docker-windows",
        "Microsoft-Windows-Subsystem-Linux",
        "VirtualMachinePlatform",
        "--update",
        "--web-download",
        "--installation-dir",
        "--wsl-default-data-root",
        "--windows-containers-default-data-root",
        "--backend=wsl-2",
        "--accept-license",
        "docker compose version",
    ):
        assert required in script

    assert "100GB" in script
    assert "REBOOT_REQUIRED" in script
    assert "--password" not in script.lower()
    assert "--install --no-distribution" not in script


def test_windows_installer_uses_only_the_official_docker_download() -> None:
    script = (ROOT / "scripts/local/install_docker_desktop.ps1").read_text(
        encoding="utf-8"
    )

    assert "https://desktop.docker.com/win/main/amd64/" in script
    assert "Docker%20Desktop%20Installer.exe" in script
    assert "Invoke-WebRequest" in script


def test_windows_installer_falls_back_to_signed_microsoft_wsl2_kernel() -> None:
    script = (ROOT / "scripts/local/install_docker_desktop.ps1").read_text(
        encoding="utf-8"
    )

    for required in (
        "https://wslstorestorage.blob.core.windows.net/wslblob/",
        "wsl_update_x64.msi",
        "Get-AuthenticodeSignature",
        "Microsoft Corporation",
        "msiexec.exe",
        "/qn",
        "/norestart",
    ):
        assert required in script

    assert "WSL update failed with exit code" not in script


def test_windows_installer_finishes_wsl_upgrade_before_docker_download() -> None:
    script = (ROOT / "scripts/local/install_docker_desktop.ps1").read_text(
        encoding="utf-8"
    )

    assert script.count("& wsl.exe --update --web-download") == 2
    assert "if ($kernelInstallation.ExitCode -eq 3010)" in script
    assert "& wsl.exe --status" in script

    status_index = script.index("& wsl.exe --status")
    default_version_index = script.index("& wsl.exe --set-default-version 2")
    docker_download_index = script.index("Docker Desktop Installer.exe")
    assert status_index < default_version_index < docker_download_index
