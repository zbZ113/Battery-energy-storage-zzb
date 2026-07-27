from __future__ import annotations

import copy
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import (
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE,
    ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
    ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
)
from quanxin_life.tools.advanced_project_report import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    ADVANCED_CELL_REPORT_TOOL_VERSION,
    GenerateAdvancedCellReportToolInput,
    execute_generate_advanced_cell_report_tool,
    register_project_generate_advanced_cell_report_tool,
)
from quanxin_life.tools.registry import (
    StandardToolName,
    ToolExecutionScope,
    ToolRegistry,
)
from tests.unit.tools.test_project_advanced_input_tool import _context


def _input_payload() -> dict[str, str]:
    return {
        "rul_result_id": str(uuid4()),
        "soh_result_id": str(uuid4()),
        "rul_conformal_result_id": str(uuid4()),
        "soh_conformal_result_id": str(uuid4()),
    }


def test_project_advanced_report_input_accepts_only_four_result_ids() -> None:
    payload = _input_payload()

    validated = GenerateAdvancedCellReportToolInput.model_validate(payload)

    assert validated.model_dump(mode="json") == payload

    for forbidden_name, value in (
        ("title", "自定义标题"),
        ("narrative", "模型表现很好"),
        ("predicted_cycle", 812.5),
        ("coverage_target", 0.9),
    ):
        with pytest.raises(ValueError, match="Extra inputs"):
            GenerateAdvancedCellReportToolInput.model_validate(
                {**payload, forbidden_name: value}
            )


@pytest.mark.parametrize(
    "field_name",
    (
        "rul_result_id",
        "soh_result_id",
        "rul_conformal_result_id",
        "soh_conformal_result_id",
    ),
)
def test_project_advanced_report_input_requires_distinct_uuid_ids(
    field_name: str,
) -> None:
    payload = _input_payload()
    payload[field_name] = "not-a-uuid"

    with pytest.raises(ValueError, match="UUID"):
        GenerateAdvancedCellReportToolInput.model_validate(payload)

    duplicate = _input_payload()
    duplicate["soh_result_id"] = duplicate["rul_result_id"]
    with pytest.raises(ValueError, match="distinct"):
        GenerateAdvancedCellReportToolInput.model_validate(duplicate)


def _provenance(source_id: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=SourceKind.PREDICTED,
        uri=f"artifact://test/{source_id}",
        sha256=sha256_canonical({"source_id": source_id}),
        description="Verified report fixture",
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )


def _result(
    *,
    tool_name: StandardToolName,
    tool_version: str,
    artifact_type: str,
    artifact: dict[str, object],
    source_id: str,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version=tool_version,
        model_version="advanced-model-v1",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="e" * 64,
        values={"artifact_type": artifact_type, "artifact": artifact},
        uncertainty=None,
        warnings=[],
        provenance=[_provenance(source_id)],
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )


def _results() -> tuple[ToolResult, ToolResult, ToolResult, ToolResult]:
    identity = {
        "record_batch_id": str(uuid4()),
        "dataset_id": "MATR",
        "cell_id": "cell-target",
        "cutoff_cycle": 20,
        "data_version": "matr-three-batch-v1",
        "feature_version": "cyclepatch-multichannel-v1",
        "split_version": "matr-cell-split-v1",
        "model_version": "advanced-model-v1",
        "raw_sequence_input_sha256": "1" * 64,
        "normalization_statistics_sha256": "2" * 64,
        "artifact_id": str(uuid4()),
        "artifact_manifest_sha256": "3" * 64,
        "decision_event_id": str(uuid4()),
        "ledger_sequence_number": 7,
        "ledger_head_sha256": "4" * 64,
    }
    rul = _result(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        tool_version="advanced-rul-prediction-tool-v1",
        artifact_type="quanxin_life.advanced_rul_prediction.v1",
        artifact={
            **identity,
            "task": "RUL",
            "route_role": "DEFAULT",
            "output_target": "matr_official_cycle_life",
            "artifact_kind": "cyclepatch_direct",
            "cycle_life_prediction": {
                "dataset_id": "MATR",
                "cell_id": "cell-target",
                "cutoff_cycle": 20,
                "target": "matr_official_cycle_life",
                "predicted_cycle": 812.5,
                "observed_cycle": None,
                "right_censored": True,
                "feature_version": "cyclepatch-multichannel-v1",
                "split_version": "matr-cell-split-v1",
                "model_version": "advanced-model-v1",
                "data_version": "matr-three-batch-v1",
            },
            "derived_remaining_cycles": 792.5,
            "upstream_result_id": str(uuid4()),
            "transform_config_sha256": "5" * 64,
            "source_manifest_hash": "6" * 64,
        },
        source_id="rul",
    )
    soh = _result(
        tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
        tool_version="advanced-soh-prediction-tool-v1",
        artifact_type="quanxin_life.advanced_soh_trajectory.v1",
        artifact={
            **identity,
            "task": "SOH",
            "route_role": "TAIL_EFFICIENCY",
            "output_target": "soh_trajectory",
            "artifact_kind": "current_hybrid",
            "prediction_cycles": [21, 22, 23],
            "predicted_soh": [0.99, 0.98, 0.97],
            "horizon_end_cycle": 23,
            "upstream_result_id": str(uuid4()),
            "transform_config_sha256": "5" * 64,
            "source_manifest_hash": "6" * 64,
        },
        source_id="soh",
    )
    rul_conformal = _result(
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        tool_version=ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
        artifact_type=ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE,
        artifact={
            **identity,
            "task": "RUL",
            "route_role": "DEFAULT",
            "output_target": "matr_official_cycle_life",
            "artifact_kind": "cyclepatch_direct",
            "prediction_result_id": rul.result_id,
            "calibration_result_id": str(uuid4()),
            "coverage_target": 0.9,
            "point_prediction_cycle": 812.5,
            "lower_cycle": 780.0,
            "upper_cycle": 845.0,
            "derived_rul_cycle": 792.5,
            "lower_rul_cycle": 760.0,
            "upper_rul_cycle": 825.0,
        },
        source_id="rul-conformal",
    )
    soh_conformal = _result(
        tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        tool_version=ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
        artifact_type=ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
        artifact={
            **identity,
            "task": "SOH",
            "route_role": "TAIL_EFFICIENCY",
            "output_target": "soh_trajectory",
            "artifact_kind": "current_hybrid",
            "prediction_result_id": soh.result_id,
            "calibration_result_id": str(uuid4()),
            "coverage_target": 0.9,
            "prediction_cycles": [21, 22, 23],
            "predicted_soh": [0.99, 0.98, 0.97],
            "lower_soh": [0.94, 0.93, 0.92],
            "upper_soh": [1.04, 1.03, 1.02],
            "finite_horizon_only": True,
            "coverage_scope": "simultaneous_finite_trajectory",
        },
        source_id="conformal",
    )
    return rul, soh, rul_conformal, soh_conformal


class _Ledger:
    def __init__(self, results: tuple[ToolResult, ...]) -> None:
        self.results = {result.result_id: result for result in results}
        self.calls: list[tuple[str, str]] = []

    def resolve_registered_result(
        self,
        context: object,
        result_id: str,
    ) -> ToolResult:
        self.calls.append((context.project_id, result_id))
        return self.results[result_id]


class _ContextValidator:
    def revalidate(self, context: object) -> object:
        return context


def test_project_advanced_report_reads_all_values_from_same_project_results() -> None:
    rul, soh, rul_conformal, soh_conformal = _results()
    ledger = _Ledger((rul, soh, rul_conformal, soh_conformal))
    tool_input = GenerateAdvancedCellReportToolInput(
        rul_result_id=rul.result_id,
        soh_result_id=soh.result_id,
        rul_conformal_result_id=rul_conformal.result_id,
        soh_conformal_result_id=soh_conformal.result_id,
    )

    result = execute_generate_advanced_cell_report_tool(
        tool_input,
        context=_context(),
        project_audit_ledger=ledger,
        clock=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
    )

    assert result.tool_version == ADVANCED_CELL_REPORT_TOOL_VERSION
    assert result.values["artifact_type"] == ADVANCED_CELL_REPORT_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert artifact["dataset_id"] == "MATR"
    assert artifact["cell_id"] == "cell-target"
    assert artifact["rul"]["predicted_cycle"] == 812.5
    assert artifact["soh"]["predicted_soh"] == [0.99, 0.98, 0.97]
    assert artifact["rul_conformal"]["coverage_target"] == 0.9
    assert artifact["soh_conformal"]["coverage_target"] == 0.9
    markdown = result.values["markdown"]
    assert "MATR official cycle life" in markdown
    assert "not a unified EOL80 definition" in markdown
    assert "finite horizon" in markdown
    assert "prediction interval, not a parameter confidence interval" in markdown
    assert "812.5" in markdown
    assert ledger.calls == [
        ("project-1", rul.result_id),
        ("project-1", soh.result_id),
        ("project-1", rul_conformal.result_id),
        ("project-1", soh_conformal.result_id),
    ]
    assert result.provenance == [
        rul.provenance[0],
        soh.provenance[0],
        rul_conformal.provenance[0],
        soh_conformal.provenance[0],
    ]


def test_project_advanced_report_keeps_point_and_coverage_routes_separate() -> None:
    rul, soh, rul_conformal, soh_conformal = _results()

    rul_values = copy.deepcopy(rul.values)
    rul_artifact = rul_values["artifact"]
    rul_artifact["cutoff_cycle"] = 50
    rul_artifact["route_role"] = "POINT_ACCURACY"
    rul_artifact["cycle_life_prediction"]["cutoff_cycle"] = 50
    rul_artifact["cycle_life_prediction"]["predicted_cycle"] = 825.0
    rul_artifact["derived_remaining_cycles"] = 775.0
    point_rul = rul.model_copy(update={"values": rul_values})

    soh_values = copy.deepcopy(soh.values)
    soh_values["artifact"]["cutoff_cycle"] = 50
    soh_values["artifact"]["prediction_cycles"] = [51, 52, 53]
    soh_values["artifact"]["horizon_end_cycle"] = 53
    point_soh = soh.model_copy(update={"values": soh_values})

    interval_values = copy.deepcopy(rul_conformal.values)
    interval_artifact = interval_values["artifact"]
    interval_artifact.update(
        {
            "cutoff_cycle": 50,
            "route_role": "COVERAGE",
            "model_version": "coverage-model-v1",
            "artifact_kind": "cyclepatch_batlinet",
            "artifact_id": str(uuid4()),
            "artifact_manifest_sha256": "7" * 64,
            "normalization_statistics_sha256": "8" * 64,
            "prediction_result_id": str(uuid4()),
            "point_prediction_cycle": 806.0,
            "lower_cycle": 770.0,
            "upper_cycle": 842.0,
            "derived_rul_cycle": 756.0,
            "lower_rul_cycle": 720.0,
            "upper_rul_cycle": 792.0,
        }
    )
    coverage_interval = rul_conformal.model_copy(
        update={
            "model_version": "coverage-model-v1",
            "values": interval_values,
        }
    )

    soh_interval_values = copy.deepcopy(soh_conformal.values)
    soh_interval_values["artifact"]["cutoff_cycle"] = 50
    soh_interval_values["artifact"]["prediction_cycles"] = [51, 52, 53]
    point_soh_interval = soh_conformal.model_copy(
        update={"values": soh_interval_values}
    )

    ledger = _Ledger(
        (point_rul, point_soh, coverage_interval, point_soh_interval)
    )
    tool_input = GenerateAdvancedCellReportToolInput(
        rul_result_id=point_rul.result_id,
        soh_result_id=point_soh.result_id,
        rul_conformal_result_id=coverage_interval.result_id,
        soh_conformal_result_id=point_soh_interval.result_id,
    )

    result = execute_generate_advanced_cell_report_tool(
        tool_input,
        context=_context(),
        project_audit_ledger=ledger,
    )

    artifact = result.values["artifact"]
    assert artifact["rul"]["route_role"] == "POINT_ACCURACY"
    assert artifact["rul"]["predicted_cycle"] == 825.0
    assert artifact["rul_conformal"]["route_role"] == "COVERAGE"
    assert artifact["rul_conformal"]["prediction_result_id"] != point_rul.result_id
    assert artifact["rul_conformal"]["point_prediction_cycle"] == 806.0
    markdown = result.values["markdown"]
    assert "Point-accuracy predicted cycle: `825.0`" in markdown
    assert "Coverage-route interval center: `806.0`" in markdown


def test_project_advanced_report_accepts_equivalent_default_route_invocations() -> None:
    rul, soh, rul_conformal, soh_conformal = _results()
    interval_values = copy.deepcopy(rul_conformal.values)
    interval_values["artifact"]["prediction_result_id"] = str(uuid4())
    independently_issued_interval = rul_conformal.model_copy(
        update={"values": interval_values}
    )
    ledger = _Ledger((rul, soh, independently_issued_interval, soh_conformal))

    result = execute_generate_advanced_cell_report_tool(
        GenerateAdvancedCellReportToolInput(
            rul_result_id=rul.result_id,
            soh_result_id=soh.result_id,
            rul_conformal_result_id=independently_issued_interval.result_id,
            soh_conformal_result_id=soh_conformal.result_id,
        ),
        context=_context(),
        project_audit_ledger=ledger,
    )

    assert result.values["artifact"]["rul"]["route_role"] == "DEFAULT"
    assert result.values["artifact"]["rul_conformal"]["route_role"] == "DEFAULT"
    assert (
        result.values["artifact"]["rul_conformal"]["prediction_result_id"]
        != rul.result_id
    )


def test_project_advanced_report_registration_is_project_only() -> None:
    registry = ToolRegistry(project_context_validator=_ContextValidator())
    register_project_generate_advanced_cell_report_tool(
        registry,
        project_audit_ledger=object(),
    )

    assert registry.list_schemas() == ()
    schemas = registry.list_schemas(
        execution_scope=ToolExecutionScope.PROJECT
    )
    assert len(schemas) == 1
    assert schemas[0].tool_name is StandardToolName.GENERATE_AUDITED_REPORT
    assert schemas[0].tool_version == ADVANCED_CELL_REPORT_TOOL_VERSION


def test_project_advanced_report_rejects_relabelled_rul_or_unbound_interval() -> None:
    rul, soh, rul_conformal, soh_conformal = _results()
    relabelled_values = copy.deepcopy(rul.values)
    relabelled_values["artifact"]["cycle_life_prediction"]["target"] = (
        "unified_eol80_cycle"
    )
    relabelled = rul.model_copy(update={"values": relabelled_values})
    ledger = _Ledger((relabelled, soh, rul_conformal, soh_conformal))
    tool_input = GenerateAdvancedCellReportToolInput(
        rul_result_id=relabelled.result_id,
        soh_result_id=soh.result_id,
        rul_conformal_result_id=rul_conformal.result_id,
        soh_conformal_result_id=soh_conformal.result_id,
    )

    with pytest.raises(ValueError, match=r"RUL|official|target"):
        execute_generate_advanced_cell_report_tool(
            tool_input,
            context=_context(),
            project_audit_ledger=ledger,
        )

    changed_values = copy.deepcopy(rul_conformal.values)
    changed_values["artifact"]["point_prediction_cycle"] = 810.0
    unbound = rul_conformal.model_copy(update={"values": changed_values})
    ledger = _Ledger((rul, soh, unbound, soh_conformal))
    with pytest.raises(ValueError, match="bind"):
        execute_generate_advanced_cell_report_tool(
            tool_input.model_copy(
                update={"rul_conformal_result_id": unbound.result_id}
            ),
            context=_context(),
            project_audit_ledger=ledger,
        )
