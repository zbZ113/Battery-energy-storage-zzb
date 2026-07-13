from __future__ import annotations

from datetime import UTC, datetime

from quanxin_life.core import ProvenanceRecord, SourceKind
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools import ToolRegistry


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="matr-split-v1",
            source_kind=SourceKind.OBSERVED,
            uri="file:///manifests/matr-split-v1.json",
            sha256="c" * 64,
            description="Cell-disjoint split manifest for audit fixture",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _manifest(*, calibration: tuple[str, ...] = ("cell-c",)) -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("cell-a", "cell-b"),
        validation=("cell-v",),
        calibration=calibration,
        test=("cell-t",),
    )


def test_registered_split_audit_tool_emits_cell_disjoint_audit_result() -> None:
    from quanxin_life.tools.split_audit import (
        AuditDatasetSplitToolInput,
        register_audit_dataset_split_tool,
    )

    registry = ToolRegistry()
    register_audit_dataset_split_tool(registry)
    tool_input = AuditDatasetSplitToolInput(
        manifest=_manifest(),
        split_version="matr-split-v1",
        data_version="matr-v1",
        feature_version="early-cycle-v1",
        provenance=_provenance(),
        audited_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    result = registry.execute("audit_dataset_split", tool_input)

    assert result.tool_name == "audit_dataset_split"
    assert result.tool_version == "split-audit-tool-v1"
    assert result.model_version == "split-manifest-auditor-v1"
    assert result.values["cell_overlap_detected"] is False
    assert result.values["blocked"] is False
    assert result.values["split_version"] == "matr-split-v1"
    assert result.values["partition_counts"] == {
        "train": 2,
        "validation": 1,
        "calibration": 1,
        "test": 1,
    }


def test_split_audit_tool_blocks_missing_required_calibration_partition() -> None:
    from quanxin_life.tools.split_audit import (
        AuditDatasetSplitToolInput,
        execute_audit_dataset_split_tool,
    )

    result = execute_audit_dataset_split_tool(
        AuditDatasetSplitToolInput(
            manifest=_manifest(calibration=()),
            split_version="matr-split-v1",
            data_version="matr-v1",
            feature_version="early-cycle-v1",
            provenance=_provenance(),
            audited_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
    )

    assert result.values["blocked"] is True
    assert result.values["reason_codes"] == ["EMPTY_REQUIRED_SPLIT:calibration"]
    assert "EMPTY_REQUIRED_SPLIT:calibration" in result.warnings
