"""Stable, service-independent domain contracts."""

from quanxin_life.core.enums import Decision, EvidenceLevel, PredictionTarget, SourceKind
from quanxin_life.core.hashing import canonical_json_bytes, sha256_canonical
from quanxin_life.core.schemas import (
    AnalysisState,
    CellMetadata,
    ConformalCalibration,
    LifePrediction,
    NormalizedConformalCalibration,
    NormalizedPredictionInterval,
    LifetimeMetrics,
    PredictionInterval,
    ProvenanceRecord,
    ToolResult,
)

__all__ = [
    "AnalysisState",
    "CellMetadata",
    "ConformalCalibration",
    "Decision",
    "EvidenceLevel",
    "LifePrediction",
    "NormalizedConformalCalibration",
    "NormalizedPredictionInterval",
    "LifetimeMetrics",
    "PredictionInterval",
    "PredictionTarget",
    "ProvenanceRecord",
    "SourceKind",
    "ToolResult",
    "canonical_json_bytes",
    "sha256_canonical",
]
