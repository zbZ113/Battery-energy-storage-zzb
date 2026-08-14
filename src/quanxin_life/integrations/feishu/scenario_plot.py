"""Render audited BLAST scenario curves directly from a ToolResult."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import as_file, files
from itertools import pairwise
from numbers import Real
from typing import Any

from quanxin_life.core import ToolResult
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
)

SCENARIO_PLOT_VERSION = "feishu-scenario-plot-v2"
SCENARIO_PLOT_FONT_SHA256 = (
    "b981c60b7f35d4730109fd6d0baee61d7b5bf29ea488cd2ddec87554b50fe457"
)
_SCENARIO_PLOT_FONT_RESOURCE = "assets/QuanxinScenarioSans-Regular.ttf"


class FeishuScenarioPlotError(ValueError):
    """Raised when a scenario result cannot be safely rendered."""


@dataclass(frozen=True, slots=True)
class FeishuScenarioPlotArtifact:
    source_result_id: str
    filename: str
    media_type: str
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _ProjectionSeries:
    scenario_id: str
    natural_years: tuple[float, ...]
    equivalent_full_cycles: tuple[float, ...]
    soh: tuple[float, ...]
    eol_threshold: float
    eol_year: float | None


class FeishuScenarioPlotter:
    """Create one deterministic comparison PNG without accepting caller numbers."""

    renderer_version = SCENARIO_PLOT_VERSION

    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        series = _series_from_result(checked)
        payload = _render_png(series)
        return FeishuScenarioPlotArtifact(
            source_result_id=checked.result_id,
            filename=f"scenario-{checked.result_id}.png",
            media_type="image/png",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


def _series_from_result(result: ToolResult) -> tuple[_ProjectionSeries, ...]:
    artifact = _mapping(result.values.get("artifact"), label="scenario artifact")
    if artifact.get("status") != "COMPLETED":
        raise FeishuScenarioPlotError("scenario plot requires a completed ToolResult")
    if (
        result.tool_name == StandardToolName.COMPARE_OPERATION_SCENARIOS.value
        and result.tool_version == COMPARE_OPERATION_SCENARIOS_TOOL_VERSION
    ):
        raw = [artifact.get("baseline")]
        comparisons = artifact.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise FeishuScenarioPlotError("scenario comparison has no comparison curves")
        raw.extend(comparisons)
    elif (
        result.tool_name == StandardToolName.PROJECT_STORAGE_LIFETIME.value
        and result.tool_version == PROJECT_STORAGE_LIFETIME_TOOL_VERSION
    ):
        raw = [artifact.get("projection")]
    else:
        raise FeishuScenarioPlotError("ToolResult is not a supported scenario result")
    return tuple(_projection(item) for item in raw)


def _projection(value: object) -> _ProjectionSeries:
    mapping = _mapping(value, label="scenario projection")
    scenario_id = _label(mapping.get("scenario_id"))
    years = _finite_axis(mapping.get("natural_years"), label="natural_years")
    efc = _finite_axis(
        mapping.get("equivalent_full_cycles"),
        label="equivalent_full_cycles",
    )
    soh = _finite_axis(mapping.get("soh"), label="soh")
    if len(years) < 2 or len(years) != len(efc) or len(years) != len(soh):
        raise FeishuScenarioPlotError("scenario curve axes must be equal nontrivial lengths")
    if any(later < earlier for earlier, later in pairwise(years)):
        raise FeishuScenarioPlotError("natural year axis must be nondecreasing")
    if any(later < earlier for earlier, later in pairwise(efc)):
        raise FeishuScenarioPlotError("EFC axis must be nondecreasing")
    if any(value < 0.0 for value in soh):
        raise FeishuScenarioPlotError("scenario SOH values must be nonnegative")
    threshold = _finite_number(mapping.get("eol_threshold"), label="eol_threshold")
    if not 0.0 < threshold < 1.0:
        raise FeishuScenarioPlotError("scenario EOL threshold must be between zero and one")
    eol = _mapping(mapping.get("eol"), label="scenario EOL outcome")
    status = eol.get("status")
    if status == "REACHED":
        eol_year = _finite_number(eol.get("natural_year"), label="eol.natural_year")
    elif status == "NOT_REACHED":
        eol_year = None
    else:
        raise FeishuScenarioPlotError("scenario EOL status is invalid")
    return _ProjectionSeries(
        scenario_id=scenario_id,
        natural_years=years,
        equivalent_full_cycles=efc,
        soh=soh,
        eol_threshold=threshold,
        eol_year=eol_year,
    )


def _render_png(
    series: tuple[_ProjectionSeries, ...],
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("scenario plot rendering requires matplotlib") from exc

    font_resource = files("quanxin_life.integrations.feishu").joinpath(
        _SCENARIO_PLOT_FONT_RESOURCE
    )
    if sha256(font_resource.read_bytes()).hexdigest() != SCENARIO_PLOT_FONT_SHA256:
        raise FeishuScenarioPlotError(
            "reviewed scenario plot font SHA-256 does not match"
        )
    with as_file(font_resource) as font_path:
        font = FontProperties(fname=str(font_path))
        figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        for index, item in enumerate(series):
            if len(series) == 1:
                label = "参考工况"
            elif index == 0:
                label = "基准工况"
            else:
                label = f"对比工况 {index}"
            line = axis.plot(
                item.natural_years,
                item.soh,
                linewidth=2.0,
                label=label,
            )[0]
            if item.eol_year is not None:
                axis.scatter(
                    [item.eol_year],
                    [item.eol_threshold],
                    color=line.get_color(),
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
        for threshold in dict.fromkeys(item.eol_threshold for item in series):
            axis.axhline(
                threshold,
                color="#A61B1B",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
        axis.set_xlabel("自然年", fontproperties=font)
        axis.set_ylabel("健康状态 SOH", fontproperties=font)
        axis.set_title("BLAST-Lite 参考工况退化对比", fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
        baseline = series[0]
        tick_indices = _tick_indices(len(baseline.natural_years), maximum=6)
        top = axis.twiny()
        top.set_xlim(axis.get_xlim())
        top.set_xticks([baseline.natural_years[index] for index in tick_indices])
        top.set_xticklabels(
            [
                format(baseline.equivalent_full_cycles[index], ".0f")
                for index in tick_indices
            ],
            fontproperties=font,
        )
        top.set_xlabel("基准工况计划等效全循环 EFC", fontproperties=font)
        figure.text(
            0.01,
            0.01,
            "受控工具结果绘制; 不含统计置信区间, 不代表目标电芯寿命承诺",
            fontsize=7,
            color="#555555",
            fontproperties=font,
        )
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=160,
            metadata={"Software": SCENARIO_PLOT_VERSION},
        )
        plt.close(figure)
    return buffer.getvalue()


def _tick_indices(length: int, *, maximum: int) -> tuple[int, ...]:
    if length <= maximum:
        return tuple(range(length))
    return tuple(
        dict.fromkeys(round(index * (length - 1) / (maximum - 1)) for index in range(maximum))
    )


def _finite_axis(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise FeishuScenarioPlotError(f"{label} must be a list")
    return tuple(_finite_number(item, label=label) for item in value)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FeishuScenarioPlotError(f"{label} must contain finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise FeishuScenarioPlotError(f"{label} must contain finite numbers")
    return number


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeishuScenarioPlotError(f"{label} is invalid")
    return value


def _label(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 100 or any(ord(char) < 32 for char in normalized):
        raise FeishuScenarioPlotError("scenario label is invalid")
    return normalized


__all__ = [
    "SCENARIO_PLOT_VERSION",
    "FeishuScenarioPlotArtifact",
    "FeishuScenarioPlotError",
    "FeishuScenarioPlotter",
]
