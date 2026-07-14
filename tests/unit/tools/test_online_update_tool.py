"""Contracts for the ledger-bound online individual-parameter update tool."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.online import (
    CalibrationConfig,
    FrozenGlobalTrajectory,
    IndividualTrajectoryCalibrator,
    NewlyObservedSOH,
)
from quanxin_life.tools.observed_soh_ingestion import (
    NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
    OBSERVED_SOH_INGESTION_MODEL_VERSION,
    OBSERVED_SOH_INGESTION_TOOL_VERSION,
)
from quanxin_life.tools.online_update import ONLINE_CALIBRATION_EVIDENCE_TYPE
from quanxin_life.tools.registry import StandardToolName, ToolRegistry
from quanxin_life.tools.trajectory_prediction import (
    PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE,
    TRAJECTORY_PREDICTION_TOOL_VERSION,
)


def _global_trajectory() -> FrozenGlobalTrajectory:
    return FrozenGlobalTrajectory(
        dataset_id="synthetic-lfp",
        cell_id="cell-online-tool-01",
        cutoff_cycle=20,
        cycles=(20, 50, 100, 150, 200),
        soh=(0.990, 0.960, 0.900, 0.830, 0.750),
        model_version="global-hybrid-v1",
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        data_version="synthetic-data-v1",
    )


def _observations() -> tuple[NewlyObservedSOH, ...]:
    return (
        NewlyObservedSOH(
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=20,
            soh=0.990,
            source_kind=SourceKind.NEWLY_OBSERVED,
        ),
        NewlyObservedSOH(
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=50,
            soh=0.940,
            source_kind=SourceKind.NEWLY_OBSERVED,
        ),
        NewlyObservedSOH(
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=100,
            soh=0.865,
            source_kind=SourceKind.NEWLY_OBSERVED,
        ),
        NewlyObservedSOH(
            dataset_id="synthetic-lfp",
            cell_id="cell-online-tool-01",
            cycle=150,
            soh=0.775,
            source_kind=SourceKind.NEWLY_OBSERVED,
        ),
    )


def _provenance() -> tuple[ProvenanceRecord, ProvenanceRecord]:
    return (
        ProvenanceRecord(
            source_id="registered-global-trajectory-source",
            source_kind=SourceKind.OBSERVED,
            uri="tool-result://trajectory/fixture",
            sha256="b" * 64,
            description="Synthetic registered trajectory source for tool-contract testing",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
        ProvenanceRecord(
            source_id="registered-new-observation-source",
            source_kind=SourceKind.NEWLY_OBSERVED,
            uri="tool-result://observations/fixture",
            sha256="c" * 64,
            description="Synthetic newly observed SOH source for tool-contract testing",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _trajectory_artifact(
    *,
    global_trajectory: FrozenGlobalTrajectory | None = None,
    model_artifact_status: str = "UNREGISTERED_IN_MEMORY",
) -> dict[str, object]:
    trajectory = global_trajectory or _global_trajectory()
    return {
        "record_batch_id": "trusted-trajectory-batch-01",
        "dataset_id": trajectory.dataset_id,
        "cell_id": trajectory.cell_id,
        "cutoff_cycle": trajectory.cutoff_cycle,
        "prediction_cycles": list(trajectory.cycles),
        "predicted_soh": list(trajectory.soh),
        "eol80_crossing": {"cutoff_cycle": 20, "eol80_cycle": 200},
        "derived_rul_cycle": 180,
        "model_version": trajectory.model_version,
        "feature_version": trajectory.feature_version,
        "split_version": trajectory.split_version,
        "data_version": trajectory.data_version,
        "upstream_result_id": str(uuid4()),
        "model_condition_feature_names": ["nominal_capacity_ah", "temperature_mean_c"],
        "model_artifact_status": model_artifact_status,
    }


def _observation_artifact(
    *,
    observations: tuple[NewlyObservedSOH, ...] | None = None,
    global_trajectory: FrozenGlobalTrajectory | None = None,
) -> dict[str, object]:
    trajectory = global_trajectory or _global_trajectory()
    source_observations = observations or _observations()
    raw_observations = [
        {
            "measurement_id": str(uuid4()),
            "dataset_id": observation.dataset_id,
            "cell_id": observation.cell_id,
            "cycle": observation.cycle,
            "soh": observation.soh,
            "source_kind": SourceKind.NEWLY_OBSERVED.value,
            "measured_at": datetime(2026, 7, 14, 8, observation.cycle % 60, tzinfo=UTC).isoformat(),
            "source_record_hash": sha256_canonical({"fixture": observation.cycle}),
        }
        for observation in source_observations
    ]
    return {
        "measurement_batch_id": "trusted-observation-batch-01",
        "dataset_id": trajectory.dataset_id,
        "cell_id": trajectory.cell_id,
        "reference_capacity_ah": 10.0,
        "reference_capacity_method": "trusted_diagnostic_reference",
        "observations": raw_observations,
        "observation_count": len(raw_observations),
        "data_version": trajectory.data_version,
        "feature_version": trajectory.feature_version,
        "split_version": trajectory.split_version,
    }


def _upstream_results(
    *,
    observations: tuple[NewlyObservedSOH, ...] | None = None,
    trajectory_artifact: dict[str, object] | None = None,
    observation_artifact: dict[str, object] | None = None,
) -> tuple[ToolResult, ToolResult]:
    global_trajectory = _global_trajectory()
    trajectory_payload = trajectory_artifact or _trajectory_artifact(
        global_trajectory=global_trajectory
    )
    observation_payload = observation_artifact or _observation_artifact(
        observations=observations,
        global_trajectory=global_trajectory,
    )
    provenance = _provenance()
    trajectory_result = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
        tool_version=TRAJECTORY_PREDICTION_TOOL_VERSION,
        model_version=global_trajectory.model_version,
        data_version=global_trajectory.data_version,
        feature_version=global_trajectory.feature_version,
        input_hash=sha256_canonical({"fixture": "global-trajectory"}),
        values={
            "artifact_type": PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE,
            "artifact": trajectory_payload,
        },
        provenance=[provenance[0]],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    observation_result = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.INGEST_NEWLY_OBSERVED_SOH.value,
        tool_version=OBSERVED_SOH_INGESTION_TOOL_VERSION,
        model_version=OBSERVED_SOH_INGESTION_MODEL_VERSION,
        data_version=global_trajectory.data_version,
        feature_version=global_trajectory.feature_version,
        input_hash=sha256_canonical({"fixture": "newly-observed-soh"}),
        values={
            "artifact_type": NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
            "artifact": observation_payload,
        },
        provenance=[provenance[1]],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    return trajectory_result, observation_result


def _update_version(calibrator: IndividualTrajectoryCalibrator) -> str:
    config = calibrator.config.model_dump(mode="json")
    return f"{calibrator.config.config_version}:{sha256_canonical(config)}"


def _input(
    *,
    trajectory_result_id: str,
    observation_result_id: str,
    calibrator: IndividualTrajectoryCalibrator | None = None,
):
    from quanxin_life.tools.online_update import UpdateCellParametersToolInput

    numerical_calibrator = calibrator or IndividualTrajectoryCalibrator()
    return UpdateCellParametersToolInput(
        trajectory_result_id=trajectory_result_id,
        observation_result_id=observation_result_id,
        update_version=_update_version(numerical_calibrator),
    )


def _online_artifact(result: ToolResult) -> dict[str, object]:
    assert result.values["artifact_type"] == ONLINE_CALIBRATION_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert isinstance(artifact, dict)
    return artifact


def test_registered_online_update_tool_derives_versions_and_audit_from_standard_ledger_evidence(
) -> None:
    from quanxin_life.tools.online_update import register_update_cell_parameters_tool

    upstream_results = _upstream_results()
    ledger = AuditLedger(upstream_results)
    registry = ToolRegistry()
    calibrator = IndividualTrajectoryCalibrator()
    register_update_cell_parameters_tool(registry, audit_ledger=ledger)
    tool_input = _input(
        calibrator=calibrator,
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    )

    result = registry.execute(StandardToolName.UPDATE_CELL_PARAMETERS, tool_input)
    artifact = _online_artifact(result)

    assert result.tool_name == StandardToolName.UPDATE_CELL_PARAMETERS.value
    assert result.tool_version == "online-update-tool-v1"
    assert result.model_version == "global-hybrid-v1"
    assert result.data_version == "synthetic-data-v1"
    assert result.feature_version == "early-cycle-v1"
    assert artifact["split_version"] == "synthetic-split-v1"
    assert artifact["status"] == "ADAPTED"
    assert artifact["trajectory"]["cycles"] == [20, 50, 100, 150, 200]
    assert artifact["audit"]["update_version"] == tool_input.update_version
    assert artifact["audit"]["calibration_config"] == artifact["calibration_config"]
    assert artifact["upstream_result_ids"] == [
        upstream_results[0].result_id,
        upstream_results[1].result_id,
    ]
    assert artifact["model_artifact_status"] == "UNREGISTERED_IN_MEMORY"
    assert "UPSTREAM_MODEL_ARTIFACT_UNREGISTERED" in result.warnings
    assert result.provenance == list(_provenance())
    assert result.uncertainty is None


def test_online_update_tool_returns_real_recheck_for_insufficient_registered_observations() -> None:
    from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

    upstream_results = _upstream_results(observations=_observations()[:2])
    tool_input = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    )

    result = execute_update_cell_parameters_tool(
        tool_input,
        audit_ledger=AuditLedger(upstream_results),
    )
    artifact = _online_artifact(result)

    assert artifact["status"] == "RECHECK"
    assert artifact["reason_code"] == "INSUFFICIENT_OBSERVATIONS"
    assert artifact["trajectory"]["adapted_soh"] == [0.990, 0.960, 0.900, 0.830, 0.750]
    assert artifact["audit"]["updated_parameters"] == {
        "bias_soh": 0.0,
        "rate_multiplier": 1.0,
        "knee_offset_fraction": 0.0,
    }
    assert "INSUFFICIENT_OBSERVATIONS" in result.warnings


def test_online_update_input_rejects_direct_numbers_invalid_ids_and_free_versions() -> None:
    from quanxin_life.tools.online_update import UpdateCellParametersToolInput

    upstream_results = _upstream_results()
    payload = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    ).model_dump(mode="python")
    payload["trajectory_result_id"] = "not-a-uuid"
    with pytest.raises(ValueError, match="trajectory_result_id"):
        UpdateCellParametersToolInput.model_validate(payload)

    payload = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    ).model_dump(mode="python")
    payload["global_trajectory"] = _global_trajectory().model_dump(mode="json")
    payload["observations"] = [item.model_dump(mode="json") for item in _observations()]
    payload["provenance"] = [item.model_dump(mode="json") for item in _provenance()]
    with pytest.raises(ValueError, match="Extra inputs"):
        UpdateCellParametersToolInput.model_validate(payload)

    payload = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    ).model_dump(mode="python")
    payload["update_version"] = "caller-chosen-update-v1"
    unbound_version_input = UpdateCellParametersToolInput.model_validate(payload)
    with pytest.raises(ValueError, match="update_version"):
        from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

        execute_update_cell_parameters_tool(
            unbound_version_input,
            audit_ledger=AuditLedger(upstream_results),
        )


def test_online_update_tool_rejects_legacy_malformed_or_unregistered_upstream_evidence() -> None:
    from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

    upstream_results = _upstream_results()
    unregistered_input = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=str(uuid4()),
    )
    with pytest.raises(ValueError, match="not registered"):
        execute_update_cell_parameters_tool(
            unregistered_input,
            audit_ledger=AuditLedger((upstream_results[0],)),
        )

    legacy_trajectory = upstream_results[0].model_copy(
        update={"values": {"prediction_cycles": [20, 50], "predicted_soh": [1.0, 0.9]}}
    )
    legacy_input = _input(
        trajectory_result_id=legacy_trajectory.result_id,
        observation_result_id=upstream_results[1].result_id,
    )
    with pytest.raises(ValueError, match="artifact_type"):
        execute_update_cell_parameters_tool(
            legacy_input,
            audit_ledger=AuditLedger((legacy_trajectory, upstream_results[1])),
        )

    malformed_observation = upstream_results[1].model_copy(
        update={
            "values": {
                "artifact_type": NEWLY_OBSERVED_SOH_EVIDENCE_TYPE,
                "artifact": {"measurement_batch_id": "missing-required-fields"},
            }
        }
    )
    malformed_input = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=malformed_observation.result_id,
    )
    with pytest.raises(ValueError, match="observation artifact"):
        execute_update_cell_parameters_tool(
            malformed_input,
            audit_ledger=AuditLedger((upstream_results[0], malformed_observation)),
        )

    no_observation_provenance = upstream_results[1].model_copy(
        update={"provenance": [upstream_results[0].provenance[0]]},
    )
    provenance_input = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=no_observation_provenance.result_id,
    )
    with pytest.raises(ValueError, match="NEWLY_OBSERVED"):
        execute_update_cell_parameters_tool(
            provenance_input,
            audit_ledger=AuditLedger((upstream_results[0], no_observation_provenance)),
        )


def test_online_update_rejects_outer_and_artifact_version_or_identity_mismatches() -> None:
    from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

    upstream_results = _upstream_results()
    outer_mismatch = upstream_results[0].model_copy(update={"data_version": "other-data-v1"})
    outer_mismatch_input = _input(
        trajectory_result_id=outer_mismatch.result_id,
        observation_result_id=upstream_results[1].result_id,
    )
    with pytest.raises(ValueError, match="data_version must match"):
        execute_update_cell_parameters_tool(
            outer_mismatch_input,
            audit_ledger=AuditLedger((outer_mismatch, upstream_results[1])),
        )

    different_cell_trajectory = _global_trajectory().model_copy(
        update={"cell_id": "different-cell"}
    )
    different_cell_observations = tuple(
        item.model_copy(update={"cell_id": "different-cell"}) for item in _observations()
    )
    mismatched_observation = _observation_artifact(
        global_trajectory=different_cell_trajectory,
        observations=different_cell_observations,
    )
    identity_results = _upstream_results(observation_artifact=mismatched_observation)
    identity_input = _input(
        trajectory_result_id=identity_results[0].result_id,
        observation_result_id=identity_results[1].result_id,
    )
    with pytest.raises(ValueError, match="identity"):
        execute_update_cell_parameters_tool(
            identity_input,
            audit_ledger=AuditLedger(identity_results),
        )


def test_online_update_config_evidence_and_hash_change_with_actual_config() -> None:
    from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

    upstream_results = _upstream_results()
    ledger = AuditLedger(upstream_results)
    default_calibrator = IndividualTrajectoryCalibrator()
    bounded_calibrator = IndividualTrajectoryCalibrator(
        config=CalibrationConfig(max_fit_rmse=0.025),
    )
    default_input = _input(
        calibrator=default_calibrator,
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    )
    bounded_input = _input(
        calibrator=bounded_calibrator,
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    )

    default_result = execute_update_cell_parameters_tool(
        default_input,
        calibrator=default_calibrator,
        audit_ledger=ledger,
    )
    bounded_result = execute_update_cell_parameters_tool(
        bounded_input,
        calibrator=bounded_calibrator,
        audit_ledger=ledger,
    )

    assert _online_artifact(default_result)["calibration_config"]["config_hash"] != (
        _online_artifact(bounded_result)["calibration_config"]["config_hash"]
    )
    assert default_result.input_hash != bounded_result.input_hash


def test_online_update_created_at_uses_execution_clock_not_upstream_timestamp() -> None:
    from quanxin_life.tools.online_update import execute_update_cell_parameters_tool

    upstream_results = _upstream_results()
    tool_input = _input(
        trajectory_result_id=upstream_results[0].result_id,
        observation_result_id=upstream_results[1].result_id,
    )
    executed_at = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)

    result = execute_update_cell_parameters_tool(
        tool_input,
        audit_ledger=AuditLedger(upstream_results),
        clock=lambda: executed_at,
    )

    assert result.created_at == executed_at
    assert result.created_at != upstream_results[0].created_at
