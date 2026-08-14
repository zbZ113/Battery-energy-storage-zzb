"""Render an audited finite-cycle SOH trajectory from one ToolResult."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from hashlib import sha256
from importlib.resources import as_file, files
from itertools import pairwise
from numbers import Real

from quanxin_life.core import ToolResult
from quanxin_life.integrations.feishu.scenario_plot import (
    FeishuScenarioPlotArtifact,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.cell_metadata_evidence import (
    validate_versioned_cell_metadata_evidence,
)

SOH_PLOT_VERSION = "feishu-soh-plot-v1"
SOH_PLOT_FONT_SHA256 = (
    "cd42dca9abc49fc97b6e5426afd8bcf87b6b002ace89bb1d03e9b2f4ecfa32d5"
)
_SOH_PLOT_FONT_RESOURCE = "assets/QuanxinSohSans-Regular.ttf"


class FeishuSohPlotError(ValueError):
    """Raised when a SOH result cannot be safely rendered."""


class FeishuSohPlotter:
    """Create one deterministic finite-cycle plot without caller numbers."""

    renderer_version = SOH_PLOT_VERSION

    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        cycles, soh, cutoff_cycle = _series(checked)
        payload = _render_png(
            cycles=cycles,
            soh=soh,
            cutoff_cycle=cutoff_cycle,
        )
        return FeishuScenarioPlotArtifact(
            source_result_id=checked.result_id,
            filename=f"soh-trajectory-{checked.result_id}.png",
            media_type="image/png",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


def _series(result: ToolResult) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if (
        result.tool_name != "predict_soh_trajectory"
        or result.tool_version != ADVANCED_SOH_PREDICTION_TOOL_VERSION
        or result.values.get("artifact_type")
        not in ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES
    ):
        raise FeishuSohPlotError("ToolResult is not a supported SOH result")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise FeishuSohPlotError("SOH artifact is invalid")
    try:
        validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=result.values.get("artifact_type"),
            legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
        )
    except ValueError as exc:
        raise FeishuSohPlotError("SOH cell metadata evidence is invalid") from exc
    cycles = _finite_axis(artifact.get("prediction_cycles"), label="prediction_cycles")
    soh = _finite_axis(artifact.get("predicted_soh"), label="predicted_soh")
    if len(cycles) < 2 or len(cycles) != len(soh):
        raise FeishuSohPlotError("SOH trajectory axes must align")
    if any(later <= earlier for earlier, later in pairwise(cycles)):
        raise FeishuSohPlotError("SOH cycle axis must be strictly increasing")
    if any(value < 0.0 or value > 1.5 for value in soh):
        raise FeishuSohPlotError("SOH values must be within the formal contract")
    if any(later > earlier for earlier, later in pairwise(soh)):
        raise FeishuSohPlotError("SOH trajectory must be non-increasing")
    cutoff_cycle = _finite_number(artifact.get("cutoff_cycle"), label="cutoff_cycle")
    horizon = _finite_number(
        artifact.get("horizon_end_cycle"),
        label="horizon_end_cycle",
    )
    if cycles[0] <= cutoff_cycle or cycles[-1] != horizon or horizon != 500.0:
        raise FeishuSohPlotError("SOH finite-cycle boundary is invalid")
    cell_id = artifact.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id.strip() or len(cell_id) > 100:
        raise FeishuSohPlotError("SOH cell identity is invalid")
    return cycles, soh, cutoff_cycle


def _render_png(
    *,
    cycles: tuple[float, ...],
    soh: tuple[float, ...],
    cutoff_cycle: float,
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("SOH plot rendering requires matplotlib") from exc

    font_resource = files("quanxin_life.integrations.feishu").joinpath(
        _SOH_PLOT_FONT_RESOURCE
    )
    if sha256(font_resource.read_bytes()).hexdigest() != SOH_PLOT_FONT_SHA256:
        raise FeishuSohPlotError("reviewed SOH plot font SHA-256 does not match")
    with as_file(font_resource) as font_path:
        font = FontProperties(fname=str(font_path))
        figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        axis.plot(cycles, soh, color="#176B87", linewidth=2.2, label="预测轨迹")
        axis.axvline(
            cutoff_cycle,
            color="#767676",
            linestyle="--",
            linewidth=1.0,
            label="已观测数据截止",
        )
        axis.set_xlabel("循环次数", fontproperties=font)
        axis.set_ylabel("健康状态 SOH", fontproperties=font)
        axis.set_title("有限时域 SOH 预测轨迹", fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
        figure.text(
            0.01,
            0.01,
            "预测边界为第 500 个循环; 当前结果不包含统计区间",
            fontsize=7,
            color="#555555",
            fontproperties=font,
        )
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=160,
            metadata={"Software": SOH_PLOT_VERSION},
        )
        plt.close(figure)
    return buffer.getvalue()


def _finite_axis(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise FeishuSohPlotError(f"{label} must be a list")
    return tuple(_finite_number(item, label=label) for item in value)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FeishuSohPlotError(f"{label} must contain finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise FeishuSohPlotError(f"{label} must contain finite numbers")
    return number


__all__ = [
    "SOH_PLOT_FONT_SHA256",
    "SOH_PLOT_VERSION",
    "FeishuSohPlotError",
    "FeishuSohPlotter",
]
