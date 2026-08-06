from pathlib import Path


def test_hust_quarantine_compose_enforces_runtime_isolation() -> None:
    path = Path("quarantine/hust/compose.yaml")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")

    required_fragments = (
        "network_mode: none",
        "read_only: true",
        'user: "65532:65532"',
        "cap_drop:",
        "- ALL",
        "no-new-privileges:true",
        "pids_limit: 32",
        "mem_limit: 2g",
        'cpus: "1.0"',
        "restart: \"no\"",
        "${HUST_QUARANTINE_INPUT:?set a workspace-external read-only input directory}",
        "${HUST_QUARANTINE_JOB:?set a workspace-external read-only job directory}",
        "${HUST_QUARANTINE_OUTPUT:?set a workspace-external output directory}",
        ":/input:ro",
        ":/job:ro",
        ":/output:rw",
        "/tmp:size=256m,noexec,nosuid,nodev",
    )
    for fragment in required_fragments:
        assert fragment in text
    assert "env_file:" not in text
    assert "secrets:" not in text


def test_hust_quarantine_build_context_is_narrow_and_versioned() -> None:
    dockerfile = Path("quarantine/hust/Dockerfile").read_text(encoding="utf-8")
    compose = Path("quarantine/hust/compose.yaml").read_text(encoding="utf-8")

    assert "context: ." in compose
    assert "COPY discover.py /app/discover.py" in dockerfile
    assert "COPY convert.py /app/convert.py" in dockerfile
    assert "COPY requirements.lock /app/requirements.lock" in dockerfile
    assert "COPY ." not in dockerfile
    assert "python:3.11.13-slim-bookworm" in dockerfile
    assert "USER 65532:65532" in dockerfile


def test_hust_quarantine_discovery_has_explicit_gates_and_bounded_output() -> None:
    text = Path("quarantine/hust/discover.py").read_text(encoding="utf-8")

    for marker in (
        "ready_for_conversion",
        "review_status",
        '"APPROVED"',
        "confirm_inventory_sha256",
        "member_sha256",
        "MAX_MEMBER_BYTES",
        "MAX_GLOBAL_REFERENCES",
        "MAX_OUTPUT_BYTES",
        "pickletools.genops",
    ):
        assert marker in text
    assert "pickle.load(" not in text
    assert "pickle.loads(" not in text
    assert "extractall" not in text
    assert ".extract(" not in text


def test_main_process_and_a100_paths_do_not_deserialize_unsafe_artifacts() -> None:
    roots = [Path("src/quanxin_life"), Path("scripts/a100")]
    forbidden = ("pickle.load(", "joblib.load(", "torch.load(")
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not any(fragment in text for fragment in forbidden), path
