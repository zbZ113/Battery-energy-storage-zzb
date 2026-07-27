from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.application.invocation_context import (
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
    ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE,
    ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE,
    ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
    ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
    ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE,
    ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
    AdvancedSplitConformalToolInput,
    execute_advanced_split_conformal_tool,
    register_project_advanced_split_conformal_tool,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    StandardToolName,
    ToolAuthorizationError,
    ToolExecutionScope,
    ToolRegistry,
)


def _context() -> VerifiedProjectInvocationContext:
    return VerifiedProjectInvocationContext(
        project_id="project-1",
        actor_user_id="user-1",
        actor_session_id="session-1",
        actor_role=UserRole.ADMIN,
        invocation_source=ProjectInvocationSource.HTTP,
        _authorization_tag="d" * 64,
    )


def _provenance(kind: SourceKind = SourceKind.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=f"source-{kind.value}",
        source_kind=kind,
        uri=f"trusted://source/{kind.value}",
        sha256="f" * 64,
        description="Audited calibration evidence",
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )


def _runtime(*, task: AdvancedModelTask, role: AdvancedModelRouteRole) -> dict[str, object]:
    return {
        "task": task.value,
        "route_role": role.value,
        "output_target": (
            "matr_official_cycle_life"
            if task is AdvancedModelTask.RUL
            else "soh_trajectory"
        ),
        "artifact_kind": (
            "cyclepatch_direct"
            if task is AdvancedModelTask.RUL
            else "current_hybrid"
        ),
        "artifact_id": str(uuid4()),
        "artifact_manifest_sha256": "a" * 64,
        "normalization_statistics_sha256": "b" * 64,
        "decision_event_id": str(uuid4()),
        "ledger_sequence_number": 7,
        "ledger_head_sha256": "c" * 64,
    }


def _result(
    artifact_type: str,
    artifact: dict[str, object],
    *,
    tool_name: StandardToolName,
    result_id: str | None = None,
) -> ToolResult:
    return ToolResult(
        result_id=result_id or str(uuid4()),
        tool_name=tool_name.value,
        tool_version="trusted-fixture-v1",
        model_version="advanced-model-v1",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="e" * 64,
        values={"artifact_type": artifact_type, "artifact": artifact},
        uncertainty=None,
        warnings=[],
        provenance=[_provenance(), _provenance(SourceKind.PREDICTED)],
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )


class _Resolver:
    def __init__(self, results: tuple[ToolResult, ...]) -> None:
        self.results = {result.result_id: result for result in results}

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        return self.results[result_id]


class _ContextValidator:
    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        return context


def test_project_rul_conformal_calibrates_and_issues_coverage_interval() -> None:
    runtime = _runtime(
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.DEFAULT,
    )
    calibration_samples = tuple(
        _result(
            ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            {
                **runtime,
                "dataset_id": "MATR",
                "cell_id": f"cal-{index}",
                "cutoff_cycle": 20,
                "split_version": "matr-cell-split-v1",
                "split_partition": "calibration",
                "point_prediction_cycle": point,
                "observed_cycle": observed,
            },
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        )
        for index, (point, observed) in enumerate(
            ((100.0, 105),) * 8 + ((110.0, 120),)
        )
    )
    resolver = _Resolver(calibration_samples)
    calibration = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="calibrate",
            task=AdvancedModelTask.RUL,
            route_role=AdvancedModelRouteRole.DEFAULT,
            alpha=0.10,
            calibration_sample_result_ids=tuple(
                result.result_id for result in calibration_samples
            ),
        ),
        context=_context(),
        result_resolver=resolver,
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )
    assert calibration.tool_version == ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION
    assert (
        calibration.values["artifact_type"]
        == ADVANCED_RUL_SPLIT_CALIBRATION_EVIDENCE_TYPE
    )
    assert calibration.values["artifact"]["route_role"] == "DEFAULT"

    prediction = _result(
        ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
        {
            **runtime,
            "record_batch_id": str(uuid4()),
            "dataset_id": "MATR",
            "cell_id": "target",
            "cutoff_cycle": 20,
            "split_version": "matr-cell-split-v1",
            "cycle_life_prediction": {
                "dataset_id": "MATR",
                "cell_id": "target",
                "cutoff_cycle": 20,
                "target": "matr_official_cycle_life",
                "predicted_cycle": 120.0,
                "observed_cycle": None,
                "right_censored": True,
                "feature_version": "cyclepatch-multichannel-v1",
                "split_version": "matr-cell-split-v1",
                "model_version": "advanced-model-v1",
                "data_version": "matr-three-batch-v1",
            },
            "derived_remaining_cycles": 100.0,
            "upstream_result_id": str(uuid4()),
            "raw_sequence_input_sha256": "1" * 64,
            "transform_config_sha256": "2" * 64,
            "source_manifest_hash": "3" * 64,
        },
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
    )
    resolver.results[prediction.result_id] = prediction
    resolver.results[calibration.result_id] = calibration

    interval = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="issue",
            task=AdvancedModelTask.RUL,
            route_role=AdvancedModelRouteRole.DEFAULT,
            prediction_result_id=prediction.result_id,
            calibration_result_id=calibration.result_id,
        ),
        context=_context(),
        result_resolver=resolver,
        clock=lambda: datetime(2026, 7, 26, 11, 0, tzinfo=UTC),
    )

    assert interval.values["artifact_type"] == ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE
    artifact = interval.values["artifact"]
    assert {
        "point_prediction_cycle",
        "lower_cycle",
        "upper_cycle",
        "derived_rul_cycle",
        "lower_rul_cycle",
        "upper_rul_cycle",
    } <= set(artifact)
    assert interval.uncertainty == {
        "coverage_target": 0.9,
        "point_prediction_cycle": 120.0,
        "lower_cycle": 110.0,
        "upper_cycle": 130.0,
        "derived_rul_cycle": 100.0,
        "lower_rul_cycle": 90.0,
        "upper_rul_cycle": 110.0,
    }


def test_project_rul_conformal_rejects_point_accuracy_prediction() -> None:
    runtime = _runtime(
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.POINT_ACCURACY,
    )
    prediction = _result(
        ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
        {
            **runtime,
            "record_batch_id": str(uuid4()),
            "dataset_id": "MATR",
            "cell_id": "target",
            "cutoff_cycle": 20,
            "split_version": "matr-cell-split-v1",
            "cycle_life_prediction": {
                "dataset_id": "MATR",
                "cell_id": "target",
                "cutoff_cycle": 20,
                "target": "matr_official_cycle_life",
                "predicted_cycle": 120.0,
                "observed_cycle": None,
                "right_censored": True,
                "feature_version": "cyclepatch-multichannel-v1",
                "split_version": "matr-cell-split-v1",
                "model_version": "advanced-model-v1",
                "data_version": "matr-three-batch-v1",
            },
            "derived_remaining_cycles": 100.0,
            "upstream_result_id": str(uuid4()),
            "raw_sequence_input_sha256": "1" * 64,
            "transform_config_sha256": "2" * 64,
            "source_manifest_hash": "3" * 64,
        },
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
    )
    with pytest.raises(ValueError, match="COVERAGE"):
        execute_advanced_split_conformal_tool(
            AdvancedSplitConformalToolInput(
                operation="issue",
                task=AdvancedModelTask.RUL,
                route_role=AdvancedModelRouteRole.COVERAGE,
                prediction_result_id=prediction.result_id,
                calibration_result_id=str(uuid4()),
            ),
            context=_context(),
            result_resolver=_Resolver((prediction,)),
        )


def test_project_conformal_calibration_rejects_requested_route_mismatch() -> None:
    runtime = _runtime(
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.COVERAGE,
    )
    samples = tuple(
        _result(
            ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            {
                **runtime,
                "dataset_id": "MATR",
                "cell_id": f"cal-{index}",
                "cutoff_cycle": 50,
                "split_version": "matr-cell-split-v1",
                "split_partition": "calibration",
                "point_prediction_cycle": 100.0 + index,
                "observed_cycle": 105 + index,
            },
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        )
        for index in range(9)
    )

    with pytest.raises(ValueError, match="requested Conformal route"):
        execute_advanced_split_conformal_tool(
            AdvancedSplitConformalToolInput(
                operation="calibrate",
                task=AdvancedModelTask.RUL,
                route_role=AdvancedModelRouteRole.DEFAULT,
                alpha=0.10,
                calibration_sample_result_ids=tuple(
                    result.result_id for result in samples
                ),
            ),
            context=_context(),
            result_resolver=_Resolver(samples),
        )


def test_project_soh_conformal_calibrates_simultaneous_finite_band() -> None:
    runtime = _runtime(
        task=AdvancedModelTask.SOH,
        role=AdvancedModelRouteRole.TAIL_EFFICIENCY,
    )
    samples = tuple(
        _result(
            ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            {
                **runtime,
                "dataset_id": "MATR",
                "cell_id": f"cal-{index}",
                "cutoff_cycle": 20,
                "split_version": "matr-cell-split-v1",
                "split_partition": "calibration",
                "prediction_cycles": [21, 22],
                "predicted_soh": predicted,
                "observed_soh": observed,
            },
            tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
        )
            for index, (predicted, observed) in enumerate(
                (([1.0, 0.90], [0.98, 0.85]),) * 8
                + (([0.95, 0.80], [0.90, 0.70]),)
            )
    )
    resolver = _Resolver(samples)
    calibration = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="calibrate",
            task=AdvancedModelTask.SOH,
            route_role=AdvancedModelRouteRole.TAIL_EFFICIENCY,
            alpha=0.10,
            calibration_sample_result_ids=tuple(result.result_id for result in samples),
        ),
        context=_context(),
        result_resolver=resolver,
    )
    assert (
        calibration.values["artifact_type"]
        == ADVANCED_SOH_SPLIT_CALIBRATION_EVIDENCE_TYPE
    )

    prediction = _result(
        ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
        {
            **runtime,
            "record_batch_id": str(uuid4()),
            "dataset_id": "MATR",
            "cell_id": "target",
            "cutoff_cycle": 20,
            "split_version": "matr-cell-split-v1",
            "prediction_cycles": [21, 22],
            "predicted_soh": [0.95, 0.80],
            "horizon_end_cycle": 22,
            "upstream_result_id": str(uuid4()),
            "raw_sequence_input_sha256": "1" * 64,
            "transform_config_sha256": "2" * 64,
            "source_manifest_hash": "3" * 64,
        },
        tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
    )
    resolver.results[prediction.result_id] = prediction
    resolver.results[calibration.result_id] = calibration
    band = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="issue",
            task=AdvancedModelTask.SOH,
            route_role=AdvancedModelRouteRole.TAIL_EFFICIENCY,
            prediction_result_id=prediction.result_id,
            calibration_result_id=calibration.result_id,
        ),
        context=_context(),
        result_resolver=resolver,
    )

    assert band.values["artifact_type"] == ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE
    artifact = band.values["artifact"]
    assert artifact["prediction_cycles"] == [21, 22]
    assert artifact["predicted_soh"] == [0.95, 0.8]
    assert artifact["lower_soh"] == pytest.approx([0.85, 0.70])
    assert artifact["upper_soh"] == pytest.approx([1.05, 0.90])
    assert artifact["finite_horizon_only"] is True
    assert artifact["coverage_scope"] == "simultaneous_finite_trajectory"


def test_advanced_split_conformal_registration_is_project_only() -> None:
    registry = ToolRegistry(project_context_validator=_ContextValidator())
    register_project_advanced_split_conformal_tool(
        registry,
        project_audit_ledger=object(),
    )

    assert registry.list_schemas() == ()
    assert [
        item.tool_name
        for item in registry.list_schemas(
            execution_scope=ToolExecutionScope.PROJECT
        )
    ] == [StandardToolName.CALIBRATE_PREDICTION_INTERVAL]
    with pytest.raises(ToolAuthorizationError, match="project-scoped"):
        registry.execute(
            StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
            {
                "operation": "issue",
                "task": "RUL",
                "route_role": "COVERAGE",
                "prediction_result_id": str(uuid4()),
                "calibration_result_id": str(uuid4()),
            },
        )
