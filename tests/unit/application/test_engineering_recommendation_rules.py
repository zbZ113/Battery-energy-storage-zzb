from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quanxin_life.application.engineering_recommendation_rules import (
    load_engineering_recommendation_ruleset_registry,
)
from quanxin_life.core import SourceKind, sha256_canonical


def _registry_payload(*, manifest_sha256: str | None = None) -> bytes:
    ruleset = {
        "ruleset_id": "reviewed-battery-release-gate",
        "ruleset_version": "reviewed-battery-release-gate-v1",
        "review_status": "APPROVED",
        "reviewed_at": "2026-08-14T00:00:00Z",
        "unresolved_reason_code": "RECOMMENDATION_EVIDENCE_UNRESOLVED",
        "rules": [
            {
                "rule_id": "minimum-predicted-cycle",
                "result_tool_name": "predict_cycle_life",
                "result_tool_version": "advanced-rul-prediction-tool-v1",
                "value_path": "values.artifact.cycle_life_prediction.predicted_cycle",
                "comparator": "GTE",
                "threshold": 900.0,
                "recheck_reason_code": "PREDICTED_CYCLE_BELOW_REVIEWED_GATE",
            }
        ],
    }
    ruleset["ruleset_manifest_sha256"] = (
        manifest_sha256 or sha256_canonical(ruleset)
    )
    return (
        json.dumps(
            {
                "schema_version": "engineering-recommendation-ruleset-registry-v1",
                "default_ruleset_id": ruleset["ruleset_id"],
                "rulesets": [ruleset],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def test_loads_exact_sha_verified_reviewed_ruleset(tmp_path: Path) -> None:
    payload = _registry_payload()
    path = tmp_path / "engineering-recommendation-rules.json"
    path.write_bytes(payload)
    file_sha256 = hashlib.sha256(payload).hexdigest()

    registry = load_engineering_recommendation_ruleset_registry(
        path,
        expected_file_sha256=file_sha256,
    )
    ruleset = registry.resolve_verified_engineering_recommendation_ruleset(
        registry.default_ruleset_id
    )

    assert registry.file_sha256 == file_sha256
    assert registry.default_ruleset_id == "reviewed-battery-release-gate"
    assert ruleset.ruleset_version == "reviewed-battery-release-gate-v1"
    assert ruleset.ruleset_manifest_sha256 == sha256_canonical(
        {
            "ruleset_id": "reviewed-battery-release-gate",
            "ruleset_version": "reviewed-battery-release-gate-v1",
            "review_status": "APPROVED",
            "reviewed_at": "2026-08-14T00:00:00Z",
            "unresolved_reason_code": "RECOMMENDATION_EVIDENCE_UNRESOLVED",
            "rules": [ruleset.rules[0].model_dump(mode="json")],
        }
    )
    assert len(ruleset.provenance) == 1
    assert ruleset.provenance[0].source_kind is SourceKind.OBSERVED
    assert ruleset.provenance[0].sha256 == file_sha256
    assert "reviewed-battery-release-gate-v1" in ruleset.provenance[0].uri


def test_ruleset_loader_rejects_file_or_semantic_sha_substitution(
    tmp_path: Path,
) -> None:
    payload = _registry_payload()
    path = tmp_path / "engineering-recommendation-rules.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="file SHA-256"):
        load_engineering_recommendation_ruleset_registry(
            path,
            expected_file_sha256="0" * 64,
        )

    substituted = _registry_payload(manifest_sha256="1" * 64)
    path.write_bytes(substituted)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_engineering_recommendation_ruleset_registry(
            path,
            expected_file_sha256=hashlib.sha256(substituted).hexdigest(),
        )


def test_ruleset_loader_rejects_duplicate_json_keys_and_symlinks(tmp_path: Path) -> None:
    duplicate = (
        b'{"schema_version":"engineering-recommendation-ruleset-registry-v1",'
        b'"schema_version":"engineering-recommendation-ruleset-registry-v1",'
        b'"default_ruleset_id":"x","rulesets":[]}\n'
    )
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_engineering_recommendation_ruleset_registry(
            path,
            expected_file_sha256=hashlib.sha256(duplicate).hexdigest(),
        )

    target = tmp_path / "target.json"
    payload = _registry_payload()
    target.write_bytes(payload)
    link = tmp_path / "rules-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    with pytest.raises(ValueError, match="regular file"):
        load_engineering_recommendation_ruleset_registry(
            link,
            expected_file_sha256=hashlib.sha256(payload).hexdigest(),
        )
