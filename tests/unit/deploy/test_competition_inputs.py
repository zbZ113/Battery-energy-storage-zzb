from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deploy.competition_inputs import (
    load_advanced_agent_policy,
    load_calibration_source_registrations,
)


def test_agent_policy_is_strict_versioned_and_hash_bound(tmp_path: Path) -> None:
    policy_file = tmp_path / "advanced-agent.json"
    payload = (
        b'{"schema_version":"advanced-agent-policy-v1",'
        b'"conformal_alpha":0.1}\n'
    )
    policy_file.write_bytes(payload)

    loaded = load_advanced_agent_policy(policy_file)

    assert loaded.policy.schema_version == "advanced-agent-policy-v1"
    assert loaded.policy.conformal_alpha == 0.1
    assert loaded.file_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":0.1,'
        '"conformal_alpha":0.2}',
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":NaN}',
        '{"schema_version":"unknown","conformal_alpha":0.1}',
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":1.0}',
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":0.1,'
        '"caller_value":42}',
    ),
)
def test_agent_policy_rejects_ambiguous_or_unsupported_json(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "advanced-agent.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        load_advanced_agent_policy(path)


def test_calibration_registry_resolves_only_relative_roots_below_evidence(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    source_root = evidence_root / "matr-three-batch"
    source_root.mkdir(parents=True)
    registry = tmp_path / "calibration-sources.json"
    registry.write_text(
        (
            '{"schema_version":"advanced-calibration-source-registry-v1",'
            '"sources":[{"registration_id":"matr-three-batch-final-v1",'
            '"evidence_relative_root":"matr-three-batch",'
            f'"three_batch_manifest_sha256":"{"a" * 64}"}}]}}'
        ),
        encoding="utf-8",
    )

    registrations = load_calibration_source_registrations(
        registry,
        evidence_root=evidence_root,
    )

    assert len(registrations) == 1
    assert registrations[0].registration_id == "matr-three-batch-final-v1"
    assert registrations[0].evidence_root == source_root.resolve(strict=True)
    assert registrations[0].three_batch_manifest_sha256 == "a" * 64


@pytest.mark.parametrize(
    "relative_root",
    ("../escape", "/absolute", "C:/absolute", ".", ""),
)
def test_calibration_registry_rejects_path_escape_or_unbounded_root(
    tmp_path: Path,
    relative_root: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    registry = tmp_path / "calibration-sources.json"
    registry.write_text(
        (
            '{"schema_version":"advanced-calibration-source-registry-v1",'
            '"sources":[{"registration_id":"matr-three-batch-final-v1",'
            f'"evidence_relative_root":"{relative_root}",'
            f'"three_batch_manifest_sha256":"{"a" * 64}"}}]}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_calibration_source_registrations(
            registry,
            evidence_root=evidence_root,
        )


def test_calibration_registry_rejects_duplicate_keys_and_sources(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    (evidence_root / "source").mkdir(parents=True)
    duplicate_key = tmp_path / "duplicate-key.json"
    duplicate_key.write_text(
        '{"schema_version":"advanced-calibration-source-registry-v1",'
        '"schema_version":"advanced-calibration-source-registry-v1",'
        '"sources":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_calibration_source_registrations(
            duplicate_key,
            evidence_root=evidence_root,
        )

    duplicate_source = tmp_path / "duplicate-source.json"
    entry = (
        '{"registration_id":"same","evidence_relative_root":"source",'
        f'"three_batch_manifest_sha256":"{"a" * 64}"}}'
    )
    duplicate_source.write_text(
        '{"schema_version":"advanced-calibration-source-registry-v1",'
        f'"sources":[{entry},{entry}]}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_calibration_source_registrations(
            duplicate_source,
            evidence_root=evidence_root,
        )
