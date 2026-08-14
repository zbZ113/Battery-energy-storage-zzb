"""Project audited analysis results into scalar engineer-facing Bitable fields."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from quanxin_life.core import CellMetadata, ToolResult
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
)
from quanxin_life.tools.cell_metadata_evidence import (
    validate_versioned_cell_metadata_evidence,
)


class AnalysisBitableProjectionError(ValueError):
    """Raised when audited values cannot form a scalar Bitable projection."""


def build_audited_analysis_bitable_fields(
    result: ToolResult,
) -> dict[str, object]:
    """Return scalar business fields without accepting caller-authored numbers."""

    checked = ToolResult.model_validate(result.model_dump(mode="json"))
    if checked.tool_name == "predict_cycle_life":
        return _cycle_life_fields(checked)
    if checked.tool_name == "predict_soh_trajectory":
        return _soh_fields(checked)
    if checked.tool_name in {
        "compare_operation_scenarios",
        "project_storage_lifetime",
    }:
        return _scenario_fields(checked)
    raise AnalysisBitableProjectionError(
        "ToolResult has no reviewed Bitable business projection"
    )


def _cycle_life_fields(result: ToolResult) -> dict[str, object]:
    if (
        result.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
        or result.values.get("artifact_type")
        not in ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES
    ):
        raise AnalysisBitableProjectionError("cycle-life result version is unsupported")
    artifact = _artifact(result)
    metadata = _metadata(
        result,
        artifact,
        legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
        metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    )
    prediction = artifact.get("cycle_life_prediction")
    if not isinstance(prediction, Mapping):
        raise AnalysisBitableProjectionError("cycle-life prediction is invalid")
    observed = _finite_nonnegative(artifact.get("cutoff_cycle"), label="cutoff")
    predicted_total = _finite_nonnegative(
        prediction.get("predicted_cycle"),
        label="predicted total cycles",
    )
    predicted_remaining = _finite_nonnegative(
        artifact.get("derived_remaining_cycles"),
        label="predicted remaining cycles",
    )
    if predicted_total < observed or not math.isclose(
        predicted_total - observed,
        predicted_remaining,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise AnalysisBitableProjectionError(
            "cycle-life observed and remaining values are inconsistent"
        )
    fields: dict[str, object] = {
        "analysis_summary": "已完成个体早期循环寿命预测",
        "applicability": "结果为循环次数预测。不表示自然年寿命",
        "observed_cycle_count": _format_cycle_count(observed),
        "predicted_total_cycles": _format_cycle_count(predicted_total),
        "predicted_remaining_cycles": _format_cycle_count(predicted_remaining),
    }
    fields.update(_metadata_fields(metadata))
    return fields


def _soh_fields(result: ToolResult) -> dict[str, object]:
    if (
        result.tool_version != ADVANCED_SOH_PREDICTION_TOOL_VERSION
        or result.values.get("artifact_type")
        not in ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES
    ):
        raise AnalysisBitableProjectionError("SOH result version is unsupported")
    artifact = _artifact(result)
    metadata = _metadata(
        result,
        artifact,
        legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
        metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    )
    observed = _finite_nonnegative(artifact.get("cutoff_cycle"), label="cutoff")
    horizon = _finite_nonnegative(
        artifact.get("horizon_end_cycle"),
        label="SOH horizon",
    )
    cycles = artifact.get("prediction_cycles")
    soh = artifact.get("predicted_soh")
    if (
        not isinstance(cycles, list)
        or not isinstance(soh, list)
        or not cycles
        or len(cycles) != len(soh)
    ):
        raise AnalysisBitableProjectionError("SOH trajectory is invalid")
    last_cycle = _finite_nonnegative(cycles[-1], label="SOH final cycle")
    last_soh = _finite_fraction(soh[-1], label="SOH horizon value")
    if not math.isclose(last_cycle, horizon, rel_tol=0.0, abs_tol=1e-9):
        raise AnalysisBitableProjectionError(
            "SOH horizon does not match the final trajectory cycle"
        )
    fields: dict[str, object] = {
        "analysis_summary": "已完成有限循环 SOH 轨迹预测",
        "applicability": (
            f"仅覆盖至第 {_format_number(horizon)} 个循环。不表示自然年寿命"
        ),
        "observed_cycle_count": _format_number(observed),
        "soh_horizon_cycle": _format_number(horizon),
        "soh_horizon_value": _format_number(last_soh),
    }
    fields.update(_metadata_fields(metadata))
    return fields


def _scenario_fields(result: ToolResult) -> dict[str, object]:
    expected_version = {
        "compare_operation_scenarios": COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        "project_storage_lifetime": PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
    }[result.tool_name]
    if result.tool_version != expected_version:
        raise AnalysisBitableProjectionError("scenario result version is unsupported")
    artifact = _artifact(result)
    if artifact.get("status") != "COMPLETED":
        raise AnalysisBitableProjectionError("scenario result is not completed")
    projection = (
        artifact.get("baseline")
        if result.tool_name == "compare_operation_scenarios"
        else artifact.get("projection")
    )
    if not isinstance(projection, Mapping):
        raise AnalysisBitableProjectionError("scenario projection is invalid")
    return {
        "analysis_summary": (
            "已完成参考工况退化对比"
            if result.tool_name == "compare_operation_scenarios"
            else "已完成项目储能寿命参考推演"
        ),
        "applicability": "结果为物理参考工况。不是目标电芯个体寿命结论",
        "scenario_id": _safe_text(
            projection.get("scenario_id"),
            label="scenario_id",
        ),
        "scenario_version": _safe_text(
            projection.get("scenario_version"),
            label="scenario_version",
        ),
    }


def _artifact(result: ToolResult) -> Mapping[str, object]:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AnalysisBitableProjectionError("analysis artifact is invalid")
    return artifact


def _metadata(
    result: ToolResult,
    artifact: Mapping[str, object],
    *,
    legacy_artifact_type: str,
    metadata_artifact_type: str,
) -> CellMetadata | None:
    try:
        return validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=result.values.get("artifact_type"),
            legacy_artifact_type=legacy_artifact_type,
            metadata_artifact_type=metadata_artifact_type,
        )
    except ValueError as exc:
        raise AnalysisBitableProjectionError(
            "analysis cell metadata evidence is invalid"
        ) from exc


def _metadata_fields(metadata: CellMetadata | None) -> dict[str, object]:
    if metadata is None:
        return {}
    return {
        "cell_name": metadata.cell_id,
        "chemistry": metadata.chemistry,
        "nominal_capacity_ah": f"{_format_number(metadata.nominal_capacity_ah)} Ah",
        "data_source": metadata.dataset_id,
        "test_specification": metadata.protocol_description,
    }


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisBitableProjectionError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AnalysisBitableProjectionError(
            f"{label} must be a finite nonnegative number"
        )
    return number


def _finite_fraction(value: object, *, label: str) -> float:
    number = _finite_nonnegative(value, label=label)
    if number > 1.0:
        raise AnalysisBitableProjectionError(f"{label} must not exceed one")
    return number


def _safe_text(value: object, *, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 200
        or any(ord(character) < 32 for character in normalized)
    ):
        raise AnalysisBitableProjectionError(f"{label} is invalid")
    return normalized


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_cycle_count(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


__all__ = [
    "AnalysisBitableProjectionError",
    "build_audited_analysis_bitable_fields",
]
