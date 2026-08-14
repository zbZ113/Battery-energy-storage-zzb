"""Render an audited cycle-life composition summary from one ToolResult."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from hashlib import sha256
from importlib.resources import as_file, files
from numbers import Real

from quanxin_life.core import ToolResult
from quanxin_life.integrations.feishu.scenario_plot import (
    FeishuScenarioPlotArtifact,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.cell_metadata_evidence import (
    validate_versioned_cell_metadata_evidence,
)

RUL_SUMMARY_PLOT_VERSION = "feishu-rul-summary-plot-v1"
RUL_SUMMARY_PLOT_FONT_SHA256 = (
    "940ccb354f0d125ad6970449553b1fcf12641ee294514c2569ec05ee35ecb57e"
)
_RUL_PLOT_FONT_RESOURCE = "assets/QuanxinRulSans-Regular.ttf"


class FeishuRulSummaryPlotError(ValueError):
    """Raised when a cycle-life result cannot be safely rendered."""


class FeishuRulSummaryPlotter:
    """Create one deterministic observed-versus-remaining lifetime summary."""

    renderer_version = RUL_SUMMARY_PLOT_VERSION

    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        cell_id, observed, predicted_total, predicted_remaining = _values(checked)
        payload = _render_png(
            cell_id=cell_id,
            observed=observed,
            predicted_total=predicted_total,
            predicted_remaining=predicted_remaining,
        )
        return FeishuScenarioPlotArtifact(
            source_result_id=checked.result_id,
            filename=f"cycle-life-summary-{checked.result_id}.png",
            media_type="image/png",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


def _values(result: ToolResult) -> tuple[str, float, float, float]:
    if (
        result.tool_name != "predict_cycle_life"
        or result.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
        or result.values.get("artifact_type")
        not in ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES
    ):
        raise FeishuRulSummaryPlotError(
            "ToolResult is not a supported cycle-life result"
        )
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise FeishuRulSummaryPlotError("cycle-life artifact is invalid")
    try:
        validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=result.values.get("artifact_type"),
            legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
        )
    except ValueError as exc:
        raise FeishuRulSummaryPlotError(
            "cycle-life cell metadata evidence is invalid"
        ) from exc
    cell_id = artifact.get("cell_id")
    prediction = artifact.get("cycle_life_prediction")
    if (
        not isinstance(cell_id, str)
        or not cell_id.strip()
        or len(cell_id) > 100
        or not isinstance(prediction, Mapping)
    ):
        raise FeishuRulSummaryPlotError("cycle-life identity or prediction is invalid")
    observed = _finite_nonnegative(artifact.get("cutoff_cycle"), label="cutoff_cycle")
    predicted_total = _finite_nonnegative(
        prediction.get("predicted_cycle"),
        label="predicted_cycle",
    )
    predicted_remaining = _finite_nonnegative(
        artifact.get("derived_remaining_cycles"),
        label="derived_remaining_cycles",
    )
    if predicted_total < observed or not math.isclose(
        predicted_total - observed,
        predicted_remaining,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise FeishuRulSummaryPlotError(
            "cycle-life observed and remaining values are inconsistent"
        )
    return cell_id, observed, predicted_total, predicted_remaining


def _render_png(
    *,
    cell_id: str,
    observed: float,
    predicted_total: float,
    predicted_remaining: float,
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("cycle-life summary rendering requires matplotlib") from exc

    font_resource = files("quanxin_life.integrations.feishu").joinpath(
        _RUL_PLOT_FONT_RESOURCE
    )
    if sha256(font_resource.read_bytes()).hexdigest() != RUL_SUMMARY_PLOT_FONT_SHA256:
        raise FeishuRulSummaryPlotError(
            "reviewed cycle-life plot font SHA-256 does not match"
        )
    with as_file(font_resource) as font_path:
        font = FontProperties(fname=str(font_path))
        figure, axis = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
        axis.barh(
            [0],
            [observed],
            color="#176B87",
            height=0.44,
            label="已观测循环",
        )
        axis.barh(
            [0],
            [predicted_remaining],
            left=[observed],
            color="#69A88D",
            height=0.44,
            label="预计剩余循环",
        )
        axis.set_xlim(0.0, max(predicted_total * 1.08, 1.0))
        axis.set_yticks([])
        axis.set_xlabel("循环次数", fontproperties=font)
        axis.set_title(f"{cell_id} | 早期循环寿命概览", fontproperties=font)
        axis.grid(axis="x", alpha=0.2)
        axis.legend(prop=font, loc="upper center", ncols=2)
        axis.text(
            observed / 2.0,
            0,
            f"已观测 {observed:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontproperties=font,
            fontsize=9,
        )
        axis.text(
            observed + predicted_remaining / 2.0,
            0,
            f"预计剩余 {predicted_remaining:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontproperties=font,
            fontsize=9,
        )
        figure.text(
            0.01,
            0.01,
            f"预计总循环寿命 {predicted_total:,.1f}; 该图不表示自然年寿命.",
            fontsize=7,
            color="#555555",
            fontproperties=font,
        )
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=160,
            metadata={"Software": RUL_SUMMARY_PLOT_VERSION},
        )
        plt.close(figure)
    return buffer.getvalue()


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FeishuRulSummaryPlotError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise FeishuRulSummaryPlotError(f"{label} must be a finite nonnegative number")
    return number


__all__ = [
    "RUL_SUMMARY_PLOT_FONT_SHA256",
    "RUL_SUMMARY_PLOT_VERSION",
    "FeishuRulSummaryPlotError",
    "FeishuRulSummaryPlotter",
]
