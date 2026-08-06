from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/local/prepare_assets.py"


def _module():
    if not SCRIPT.is_file():
        pytest.fail("local asset preparation script is not implemented")
    spec = importlib.util.spec_from_file_location("prepare_local_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_verification_rejects_changed_or_escaping_files(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.sha256"
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest.write_text(f"{digest}  payload.json\n", encoding="utf-8")

    module.verify_manifest(tmp_path, manifest)

    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        module.verify_manifest(tmp_path, manifest)

    manifest.write_text(f"{digest}  ../payload.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bounded relative path"):
        module.verify_manifest(tmp_path, manifest)


def test_selected_tar_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    module = _module()
    archive = tmp_path / "unsafe.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("release/demo-target/../../escape.txt")
        content = b"escape"
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))

    with pytest.raises(ValueError, match="unsafe archive member"):
        module.install_demo_package(
            archive,
            target=tmp_path / "target",
            release_root="release",
        )


def test_demo_registration_must_match_observed_csv_sha(tmp_path: Path) -> None:
    module = _module()
    csv_path = tmp_path / "MATR_b3c34-cutoff-20.csv"
    csv_path.write_text("cell_id,cycle_index\nMATR_b3c34,1\n", encoding="utf-8")
    registration = tmp_path / "MATR_b3c34-cutoff-20.registration.json"
    registration.write_text(
        '{"metadata":{"source_sha256":"' + "0" * 64 + '"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registration source SHA-256"):
        module.verify_demo_registration(csv_path, registration)


def test_script_pins_the_runtime_v2_registry_and_four_cutoffs() -> None:
    source = SCRIPT.read_text(encoding="utf-8") if SCRIPT.is_file() else ""

    assert "c31f62e68faa66e56b16d21ebdd3067d5dea0c8408bb3ad6baa73e05a42824be" in source
    assert "deployment-registry-provenance-v2" in source
    assert "deployment-registry-runtime-v2-delta" in source
    assert "quanxin-competition-demo-ready-2026.07.28-1.tgz" in source
    assert "model.safetensors" in source
    assert "os.link" in source
    assert "(20, 50, 100, 150)" in source
