"""Select controlled Feishu plot templates from audited ToolResult identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from quanxin_life.core import ToolResult
from quanxin_life.tools import StandardToolName
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

from .scenario_plot import (
    FeishuScenarioPlotArtifact,
    FeishuScenarioPlotter,
)
from .soh_plot import SOH_PLOT_VERSION, FeishuSohPlotter

ANALYSIS_PLOT_DISPATCHER_VERSION = "feishu-analysis-plot-dispatcher-v1"


class AnalysisPlotTemplate(StrEnum):
    """Reviewed presentation templates selected from result type only."""

    CYCLE_LIFE_SUMMARY = "CYCLE_LIFE_SUMMARY"
    FINITE_SOH_CURVE = "FINITE_SOH_CURVE"
    SCENARIO_COMPARISON = "SCENARIO_COMPARISON"


class FeishuAnalysisPlotError(ValueError):
    """Raised when a result cannot use the requested controlled template."""


@dataclass(frozen=True, slots=True)
class AnalysisPlotPlan:
    template: AnalysisPlotTemplate
    image_required: bool


@dataclass(frozen=True, slots=True)
class FeishuAnalysisPlotArtifact(FeishuScenarioPlotArtifact):
    template: AnalysisPlotTemplate
    renderer_version: str


class FeishuAnalysisPlotter:
    """Dispatch audited results to versioned renderers without caller numbers."""

    dispatcher_version = ANALYSIS_PLOT_DISPATCHER_VERSION
    # The durable SOH delivery checkpoint stores this attribute. Preserve the
    # existing renderer identity so in-flight jobs remain replayable.
    renderer_version = SOH_PLOT_VERSION

    def __init__(
        self,
        *,
        soh_plotter: FeishuSohPlotter | None = None,
        scenario_plotter: FeishuScenarioPlotter | None = None,
    ) -> None:
        self._soh_plotter = soh_plotter or FeishuSohPlotter()
        self._scenario_plotter = scenario_plotter or FeishuScenarioPlotter()

    def plan(self, result: ToolResult) -> AnalysisPlotPlan:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        if (
            checked.tool_name == StandardToolName.PREDICT_SOH_TRAJECTORY.value
            and checked.tool_version == ADVANCED_SOH_PREDICTION_TOOL_VERSION
            and checked.values.get("artifact_type")
            in ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES
        ):
            self._validate_metadata(
                checked,
                legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
                metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
            )
            return AnalysisPlotPlan(
                template=AnalysisPlotTemplate.FINITE_SOH_CURVE,
                image_required=True,
            )
        if (
            checked.tool_name
            == StandardToolName.COMPARE_OPERATION_SCENARIOS.value
            and checked.tool_version == COMPARE_OPERATION_SCENARIOS_TOOL_VERSION
        ) or (
            checked.tool_name == StandardToolName.PROJECT_STORAGE_LIFETIME.value
            and checked.tool_version == PROJECT_STORAGE_LIFETIME_TOOL_VERSION
        ):
            return AnalysisPlotPlan(
                template=AnalysisPlotTemplate.SCENARIO_COMPARISON,
                image_required=True,
            )
        if (
            checked.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value
            and checked.tool_version == ADVANCED_RUL_PREDICTION_TOOL_VERSION
            and checked.values.get("artifact_type")
            in ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES
        ):
            self._validate_metadata(
                checked,
                legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
                metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
            )
            return AnalysisPlotPlan(
                template=AnalysisPlotTemplate.CYCLE_LIFE_SUMMARY,
                image_required=False,
            )
        raise FeishuAnalysisPlotError(
            "ToolResult does not have a reviewed analysis plot template"
        )

    @staticmethod
    def _validate_metadata(
        result: ToolResult,
        *,
        legacy_artifact_type: str,
        metadata_artifact_type: str,
    ) -> None:
        try:
            validate_versioned_cell_metadata_evidence(
                result.values.get("artifact"),
                artifact_type=result.values.get("artifact_type"),
                legacy_artifact_type=legacy_artifact_type,
                metadata_artifact_type=metadata_artifact_type,
            )
        except ValueError as exc:
            raise FeishuAnalysisPlotError(
                "ToolResult cell metadata evidence is invalid"
            ) from exc

    def render(
        self,
        result: ToolResult,
        *,
        template: AnalysisPlotTemplate | None = None,
    ) -> FeishuAnalysisPlotArtifact:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        plan = self.plan(checked)
        if not plan.image_required:
            raise FeishuAnalysisPlotError(
                f"{plan.template.value} does not require an image"
            )
        if template is not None and template is not plan.template:
            raise FeishuAnalysisPlotError(
                f"template {template.value} is not allowed for this ToolResult"
            )
        if plan.template is AnalysisPlotTemplate.FINITE_SOH_CURVE:
            rendered = self._soh_plotter.render(checked)
            renderer_version = self._soh_plotter.renderer_version
        else:
            rendered = self._scenario_plotter.render(checked)
            renderer_version = self._scenario_plotter.renderer_version
        return FeishuAnalysisPlotArtifact(
            source_result_id=rendered.source_result_id,
            filename=rendered.filename,
            media_type=rendered.media_type,
            payload=rendered.payload,
            sha256=rendered.sha256,
            template=plan.template,
            renderer_version=renderer_version,
        )


__all__ = [
    "ANALYSIS_PLOT_DISPATCHER_VERSION",
    "AnalysisPlotPlan",
    "AnalysisPlotTemplate",
    "FeishuAnalysisPlotArtifact",
    "FeishuAnalysisPlotError",
    "FeishuAnalysisPlotter",
]
