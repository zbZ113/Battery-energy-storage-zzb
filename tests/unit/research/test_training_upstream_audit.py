from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.audit_training_upstreams import audit_training_upstreams


def test_training_upstream_audit_binds_commit_and_license_status() -> None:
    root = Path(__file__).parents[3]

    records = {record["model_family"]: record for record in audit_training_upstreams(root)}

    assert records["pbt"]["commit"] == "a2df9d36db3f57ab2c5686638952ad7715ea0646"
    assert records["pbt"]["license_status"] == "VERIFIED_LICENSE_PRESENT"
    assert records["pbt"]["license_sha256"] is not None
    assert records["diting"]["commit"] == "b67f48373c591ca62c030fca257ee94910c974ce"
    assert records["diting"]["license_status"] == "RESEARCH_ONLY_LICENSE_UNVERIFIED"
    assert records["diting"]["license_sha256"] is None
    assert records["batterymformer"]["commit"] == "febe174032ad4861fa057b9af23f5bcee8a8fb77"
    assert records["batterymformer"]["license_status"] == "RESEARCH_ONLY_LICENSE_UNVERIFIED"
    assert records["magnet"]["commit"] == "aafb90c551d20748251a35fd51a34eae2539aaca"
    assert records["magnet"]["license_status"] == "VERIFIED_LICENSE_PRESENT"
    assert records["battgp"]["commit"] == "6d5e1db3337f0de5f3a533acbc09eddccc178e9d"
    assert records["blast"]["commit"] == "b093495b47dc40dd96dba865d91f553619501e94"
    assert records["blast"]["license_status"] == "VERIFIED_LICENSE_PRESENT"
    assert records["smart_feature"]["commit"] == "dc6beea547960cf44d1721734dba93bdcab19a1f"
    assert records["smart_feature"]["license_status"] == "RESEARCH_ONLY_LICENSE_UNVERIFIED"


def test_training_upstream_audit_rejects_commit_drift() -> None:
    root = Path(__file__).parents[3]

    with pytest.raises(ValueError, match="commit"):
        audit_training_upstreams(
            root,
            expected_commits={
                "pbt": "0" * 40,
                "diting": "1" * 40,
                "batterymformer": "2" * 40,
                "magnet": "3" * 40,
                "battgp": "4" * 40,
                "blast": "5" * 40,
                "smart_feature": "6" * 40,
            },
        )
