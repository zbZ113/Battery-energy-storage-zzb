"""Shared versions for audited report generation and downstream rendering."""

AUDITED_REPORT_TOOL_NAME = "generate_audited_report"
AUDITED_REPORT_TOOL_VERSION = "audited-report-tool-v1"
RECOMMENDATION_REPORT_RENDERER_VERSION = (
    "engineering-recommendation-markdown-report-v1"
)
ADVANCED_CELL_REPORT_TOOL_VERSION = "advanced-cell-report-tool-v1"
ADVANCED_CELL_REPORT_EVIDENCE_TYPE = "quanxin_life.advanced_cell_report.v1"
ADVANCED_CELL_REPORT_MODEL_VERSION = "advanced-cell-report-template-v1"

__all__ = [
    "ADVANCED_CELL_REPORT_EVIDENCE_TYPE",
    "ADVANCED_CELL_REPORT_MODEL_VERSION",
    "ADVANCED_CELL_REPORT_TOOL_VERSION",
    "AUDITED_REPORT_TOOL_NAME",
    "AUDITED_REPORT_TOOL_VERSION",
    "RECOMMENDATION_REPORT_RENDERER_VERSION",
]
