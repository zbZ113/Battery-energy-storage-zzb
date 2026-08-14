"""Project-bound single-cell report assembled only from registered ToolResults."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.audit.project_ledger import ProjectResultLedger
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProvenanceRecord,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.reporting.contracts import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    ADVANCED_CELL_REPORT_MODEL_VERSION,
    ADVANCED_CELL_REPORT_TOOL_VERSION,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE,
    ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
    ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
    AdvancedRULInference,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
    AdvancedSOHInference,
)
from quanxin_life.tools.cell_metadata_evidence import (
    validate_versioned_cell_metadata_evidence,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )

ADVANCED_CELL_REPORT_FEATURE_VERSION = "advanced-cell-report-evidence-v1"
Clock = Callable[[], datetime]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
SOHValue = Annotated[float, Field(ge=0.0, le=1.5, allow_inf_nan=False)]


class GenerateAdvancedCellReportToolInput(ContractModel):
    """Four same-project result references; callers cannot supply report content."""

    rul_result_id: str
    soh_result_id: str
    rul_conformal_result_id: str
    soh_conformal_result_id: str

    @field_validator(
        "rul_result_id",
        "soh_result_id",
        "rul_conformal_result_id",
        "soh_conformal_result_id",
    )
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("report result references must be UUID strings") from exc

    @model_validator(mode="after")
    def require_distinct_result_ids(self) -> GenerateAdvancedCellReportToolInput:
        result_ids = (
            self.rul_result_id,
            self.soh_result_id,
            self.rul_conformal_result_id,
            self.soh_conformal_result_id,
        )
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("report result references must be distinct")
        return self


class _RULIntervalEvidence(ContractModel):
    task: Literal[AdvancedModelTask.RUL]
    route_role: AdvancedModelRouteRole
    output_target: Literal["matr_official_cycle_life"]
    dataset_id: Literal["MATR"]
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    artifact_kind: Literal["cyclepatch_direct", "cyclepatch_batlinet"]
    artifact_id: str
    artifact_manifest_sha256: Sha256
    normalization_statistics_sha256: Sha256
    decision_event_id: str
    ledger_sequence_number: int = Field(ge=1)
    ledger_head_sha256: Sha256
    prediction_result_id: str
    calibration_result_id: str
    coverage_target: float = Field(gt=0.0, lt=1.0)
    point_prediction_cycle: FiniteFloat
    lower_cycle: FiniteFloat
    upper_cycle: FiniteFloat
    derived_rul_cycle: FiniteFloat
    lower_rul_cycle: FiniteFloat
    upper_rul_cycle: FiniteFloat

    @field_validator(
        "prediction_result_id",
        "calibration_result_id",
        "artifact_id",
        "decision_event_id",
    )
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Conformal result references must be UUID strings") from exc

    @model_validator(mode="after")
    def interval_is_ordered(self) -> _RULIntervalEvidence:
        expected_role = (
            AdvancedModelRouteRole.DEFAULT
            if self.cutoff_cycle == 20
            else AdvancedModelRouteRole.COVERAGE
        )
        if self.route_role is not expected_role:
            raise ValueError("RUL Conformal route role does not match cutoff")
        if not self.lower_cycle <= self.point_prediction_cycle <= self.upper_cycle:
            raise ValueError("RUL Conformal cycle interval is not ordered")
        if not self.lower_rul_cycle <= self.derived_rul_cycle <= self.upper_rul_cycle:
            raise ValueError("RUL Conformal remaining-cycle interval is not ordered")
        return self


class _SOHBandEvidence(ContractModel):
    task: Literal[AdvancedModelTask.SOH]
    route_role: AdvancedModelRouteRole
    output_target: Literal["soh_trajectory"]
    dataset_id: Literal["MATR"]
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    artifact_kind: Literal["hybridpatch_v2", "current_hybrid"]
    artifact_id: str
    artifact_manifest_sha256: Sha256
    normalization_statistics_sha256: Sha256
    decision_event_id: str
    ledger_sequence_number: int = Field(ge=1)
    ledger_head_sha256: Sha256
    prediction_result_id: str
    calibration_result_id: str
    coverage_target: float = Field(gt=0.0, lt=1.0)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    predicted_soh: tuple[SOHValue, ...] = Field(min_length=1)
    lower_soh: tuple[SOHValue, ...] = Field(min_length=1)
    upper_soh: tuple[SOHValue, ...] = Field(min_length=1)
    finite_horizon_only: Literal[True]
    coverage_scope: Literal["simultaneous_finite_trajectory"]

    @field_validator(
        "prediction_result_id",
        "calibration_result_id",
        "artifact_id",
        "decision_event_id",
    )
    @classmethod
    def require_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Conformal result references must be UUID strings") from exc

    @model_validator(mode="after")
    def band_is_aligned(self) -> _SOHBandEvidence:
        lengths = {
            len(self.prediction_cycles),
            len(self.predicted_soh),
            len(self.lower_soh),
            len(self.upper_soh),
        }
        if len(lengths) != 1:
            raise ValueError("SOH Conformal band axes do not align")
        if (
            self.prediction_cycles[0] <= self.cutoff_cycle
            or self.prediction_cycles[-1] > 500
            or any(
                following <= current
                for current, following in pairwise(self.prediction_cycles)
            )
        ):
            raise ValueError("SOH Conformal prediction axis is invalid")
        if any(
            not lower <= point <= upper
            for lower, point, upper in zip(
                self.lower_soh,
                self.predicted_soh,
                self.upper_soh,
                strict=True,
            )
        ):
            raise ValueError("SOH Conformal band is not ordered")
        return self

    @field_validator("route_role")
    @classmethod
    def require_soh_role(
        cls,
        value: AdvancedModelRouteRole,
    ) -> AdvancedModelRouteRole:
        if value not in {
            AdvancedModelRouteRole.MEAN_ACCURACY,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
        }:
            raise ValueError("SOH Conformal route role is invalid")
        return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _artifact(
    result: ToolResult,
    *,
    expected_type: str | Collection[str],
) -> Mapping[str, object]:
    expected_types = (
        frozenset({expected_type})
        if isinstance(expected_type, str)
        else frozenset(expected_type)
    )
    if result.values.get("artifact_type") not in expected_types:
        raise ValueError("upstream ToolResult has an unsupported evidence artifact")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("upstream ToolResult is missing structured evidence")
    return artifact


def _resolve_project_result(
    ledger: ProjectResultLedger,
    context: VerifiedProjectInvocationContext,
    result_id: str,
) -> ToolResult:
    resolved = ledger.resolve_registered_result(context, result_id)
    result = ToolResult.model_validate(resolved.model_dump(mode="json"))
    if result.result_id != result_id:
        raise ValueError("project result ledger returned a mismatched ToolResult")
    return result


def _decode_rul(result: ToolResult) -> tuple[AdvancedRULInference, Mapping[str, object]]:
    if (
        result.tool_name != StandardToolName.PREDICT_CYCLE_LIFE.value
        or result.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
    ):
        raise ValueError("RUL result is not a formal Advanced prediction")
    artifact = _artifact(
        result,
        expected_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    )
    validate_versioned_cell_metadata_evidence(
        artifact,
        artifact_type=result.values.get("artifact_type"),
        legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
        metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    )
    inference = AdvancedRULInference.model_validate(
        {
            "prediction": artifact.get("cycle_life_prediction"),
            "task": artifact.get("task"),
            "route_role": artifact.get("route_role"),
            "output_target": artifact.get("output_target"),
            "artifact_kind": artifact.get("artifact_kind"),
            "artifact_id": artifact.get("artifact_id"),
            "artifact_manifest_sha256": artifact.get(
                "artifact_manifest_sha256"
            ),
            "raw_sequence_input_sha256": artifact.get(
                "raw_sequence_input_sha256"
            ),
            "normalization_statistics_sha256": artifact.get(
                "normalization_statistics_sha256"
            ),
            "decision_event_id": artifact.get("decision_event_id"),
            "ledger_sequence_number": artifact.get(
                "ledger_sequence_number"
            ),
            "ledger_head_sha256": artifact.get("ledger_head_sha256"),
        }
    )
    prediction = inference.prediction
    if (
        result.model_version != prediction.model_version
        or result.data_version != prediction.data_version
        or result.feature_version != prediction.feature_version
    ):
        raise ValueError("RUL ToolResult versions do not match its evidence")
    return inference, artifact


def _decode_soh(result: ToolResult) -> tuple[AdvancedSOHInference, Mapping[str, object]]:
    if (
        result.tool_name != StandardToolName.PREDICT_SOH_TRAJECTORY.value
        or result.tool_version != ADVANCED_SOH_PREDICTION_TOOL_VERSION
    ):
        raise ValueError("SOH result is not a formal Advanced prediction")
    artifact = _artifact(
        result,
        expected_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPES,
    )
    validate_versioned_cell_metadata_evidence(
        artifact,
        artifact_type=result.values.get("artifact_type"),
        legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
        metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    )
    inference = AdvancedSOHInference.model_validate(
        {
            "dataset_id": artifact.get("dataset_id"),
            "cell_id": artifact.get("cell_id"),
            "cutoff_cycle": artifact.get("cutoff_cycle"),
            "data_version": result.data_version,
            "feature_version": result.feature_version,
            "split_version": artifact.get("split_version"),
            "model_version": result.model_version,
            "task": artifact.get("task"),
            "route_role": artifact.get("route_role"),
            "output_target": artifact.get("output_target"),
            "artifact_kind": artifact.get("artifact_kind"),
            "artifact_id": artifact.get("artifact_id"),
            "artifact_manifest_sha256": artifact.get(
                "artifact_manifest_sha256"
            ),
            "raw_sequence_input_sha256": artifact.get(
                "raw_sequence_input_sha256"
            ),
            "normalization_statistics_sha256": artifact.get(
                "normalization_statistics_sha256"
            ),
            "prediction_cycles": artifact.get("prediction_cycles"),
            "predicted_soh": artifact.get("predicted_soh"),
            "decision_event_id": artifact.get("decision_event_id"),
            "ledger_sequence_number": artifact.get(
                "ledger_sequence_number"
            ),
            "ledger_head_sha256": artifact.get("ledger_head_sha256"),
        }
    )
    return inference, artifact


def _decode_conformal(
    result: ToolResult,
) -> tuple[_RULIntervalEvidence | _SOHBandEvidence, str]:
    if (
        result.tool_name
        != StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value
        or result.tool_version != ADVANCED_SPLIT_CONFORMAL_TOOL_VERSION
    ):
        raise ValueError("Conformal result is not a route-specific Advanced result")
    artifact_type = result.values.get("artifact_type")
    if not isinstance(artifact_type, str):
        raise ValueError("Conformal result is missing artifact_type")
    artifact = _artifact(result, expected_type=artifact_type)
    evidence: _RULIntervalEvidence | _SOHBandEvidence
    if artifact_type == ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE:
        evidence = _RULIntervalEvidence.model_validate(
                {
                    name: artifact.get(name)
                    for name in _RULIntervalEvidence.model_fields
                }
        )
    elif artifact_type == ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE:
        evidence = _SOHBandEvidence.model_validate(
                {
                    name: artifact.get(name)
                    for name in _SOHBandEvidence.model_fields
                }
        )
    else:
        raise ValueError(
            "Conformal result is not an issued interval or finite band"
        )
    if (
        result.model_version != evidence.model_version
        or result.data_version != evidence.data_version
        or result.feature_version != evidence.feature_version
    ):
        raise ValueError("Conformal ToolResult versions do not match its evidence")
    return evidence, artifact_type


def _validate_shared_identity(
    rul: AdvancedRULInference,
    soh: AdvancedSOHInference,
) -> None:
    prediction = rul.prediction
    if (
        prediction.dataset_id != soh.dataset_id
        or prediction.cell_id != soh.cell_id
        or prediction.cutoff_cycle != soh.cutoff_cycle
        or prediction.data_version != soh.data_version
        or prediction.feature_version != soh.feature_version
        or prediction.split_version != soh.split_version
        or rul.raw_sequence_input_sha256 != soh.raw_sequence_input_sha256
    ):
        raise ValueError("RUL and SOH results do not describe the same input cell")


def _validate_conformal_binding(
    conformal: _RULIntervalEvidence | _SOHBandEvidence,
    *,
    rul: AdvancedRULInference,
    soh_result: ToolResult,
    soh: AdvancedSOHInference,
) -> None:
    prediction = rul.prediction
    if (
        conformal.dataset_id != prediction.dataset_id
        or conformal.cell_id != prediction.cell_id
        or conformal.cutoff_cycle != prediction.cutoff_cycle
        or conformal.data_version != prediction.data_version
        or conformal.feature_version != prediction.feature_version
        or conformal.split_version != prediction.split_version
    ):
        raise ValueError("Conformal result does not match the report cell")
    if isinstance(conformal, _RULIntervalEvidence):
        if prediction.cutoff_cycle == 20:
            if (
                rul.route_role is not AdvancedModelRouteRole.DEFAULT
                or conformal.route_role is not AdvancedModelRouteRole.DEFAULT
                or conformal.point_prediction_cycle
                != prediction.predicted_cycle
                or conformal.model_version != prediction.model_version
                or conformal.artifact_kind != rul.artifact_kind
                or conformal.artifact_id != rul.artifact_id
                or conformal.artifact_manifest_sha256
                != rul.artifact_manifest_sha256
                or conformal.normalization_statistics_sha256
                != rul.normalization_statistics_sha256
            ):
                raise ValueError(
                    "RUL Conformal result does not bind the default RUL prediction"
                )
        elif (
            rul.route_role is not AdvancedModelRouteRole.POINT_ACCURACY
            or conformal.route_role is not AdvancedModelRouteRole.COVERAGE
        ):
            raise ValueError("RUL Conformal result does not bind the RUL prediction")
        return
    if (
        conformal.prediction_result_id != soh_result.result_id
        or conformal.route_role is not soh.route_role
        or conformal.model_version != soh.model_version
        or conformal.artifact_kind != soh.artifact_kind
        or conformal.artifact_id != soh.artifact_id
        or conformal.artifact_manifest_sha256
        != soh.artifact_manifest_sha256
        or conformal.normalization_statistics_sha256
        != soh.normalization_statistics_sha256
        or conformal.prediction_cycles != soh.prediction_cycles
        or conformal.predicted_soh != soh.predicted_soh
    ):
        raise ValueError("SOH Conformal result does not bind the SOH prediction")


def _provenance(results: Sequence[ToolResult]) -> list[ProvenanceRecord]:
    seen: set[str] = set()
    output: list[ProvenanceRecord] = []
    for result in results:
        for record in result.provenance:
            digest = sha256_canonical(record.model_dump(mode="json"))
            if digest not in seen:
                seen.add(digest)
                output.append(record)
    if not output:
        raise ValueError("report upstream results require provenance")
    return output


def _rul_evidence(
    result: ToolResult,
    inference: AdvancedRULInference,
) -> dict[str, object]:
    prediction = inference.prediction
    return {
        "result_id": result.result_id,
        "target": prediction.target.value,
        "predicted_cycle": prediction.predicted_cycle,
        "derived_remaining_cycles": prediction.derived_remaining_cycles,
        "route_role": inference.route_role.value,
        "artifact_kind": inference.artifact_kind,
        "artifact_id": inference.artifact_id,
        "artifact_manifest_sha256": inference.artifact_manifest_sha256,
    }


def _soh_evidence(
    result: ToolResult,
    inference: AdvancedSOHInference,
) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "route_role": inference.route_role.value,
        "artifact_kind": inference.artifact_kind,
        "artifact_id": inference.artifact_id,
        "artifact_manifest_sha256": inference.artifact_manifest_sha256,
        "prediction_cycles": list(inference.prediction_cycles),
        "predicted_soh": list(inference.predicted_soh),
        "finite_horizon_only": True,
    }


def _conformal_evidence(
    result: ToolResult,
    evidence: _RULIntervalEvidence | _SOHBandEvidence,
    artifact_type: str,
) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "artifact_type": artifact_type,
        **evidence.model_dump(mode="json"),
    }


def _markdown(
    *,
    rul: AdvancedRULInference,
    soh: AdvancedSOHInference,
    rul_conformal: _RULIntervalEvidence,
    soh_conformal: _SOHBandEvidence,
) -> str:
    prediction = rul.prediction
    lines = [
        "# Advanced single-cell audited report",
        "",
        "## Scope",
        "",
        (
            "- RUL target: MATR official cycle life; this is not a unified "
            "EOL80 definition."
        ),
        (
            "- SOH result: finite horizon trajectory only; it is not an "
            "official lifetime label."
        ),
        (
            "- Split Conformal result: prediction interval, not a parameter "
            "confidence interval."
        ),
        "",
        "## Cell identity",
        "",
        f"- Dataset: `{prediction.dataset_id}`",
        f"- Cell: `{prediction.cell_id}`",
        f"- Cutoff cycle: `{prediction.cutoff_cycle}`",
        "",
        "## Official cycle-life prediction",
        "",
        f"- Point-accuracy predicted cycle: `{prediction.predicted_cycle}`",
        f"- Derived remaining cycles: `{prediction.derived_remaining_cycles}`",
        f"- Route role: `{rul.route_role.value}`",
        "",
        "## SOH trajectory",
        "",
        f"- First prediction cycle: `{soh.prediction_cycles[0]}`",
        f"- Horizon end cycle: `{soh.prediction_cycles[-1]}`",
        f"- Initial predicted SOH: `{soh.predicted_soh[0]}`",
        f"- Horizon-end predicted SOH: `{soh.predicted_soh[-1]}`",
        f"- Route role: `{soh.route_role.value}`",
        "",
        "## Split Conformal",
        "",
        "### RUL interval",
        "",
        f"- Coverage target: `{rul_conformal.coverage_target}`",
        (
            "- Coverage-route interval center: "
            f"`{rul_conformal.point_prediction_cycle}`"
        ),
        f"- Coverage route role: `{rul_conformal.route_role.value}`",
        (
            f"- Cycle interval: `[{rul_conformal.lower_cycle}, "
            f"{rul_conformal.upper_cycle}]`"
        ),
        (
            f"- Remaining-cycle interval: `[{rul_conformal.lower_rul_cycle}, "
            f"{rul_conformal.upper_rul_cycle}]`"
        ),
        "",
        "### SOH simultaneous finite band",
        "",
        f"- Coverage target: `{soh_conformal.coverage_target}`",
        f"- Band first cycle: `{soh_conformal.prediction_cycles[0]}`",
        f"- Band horizon end: `{soh_conformal.prediction_cycles[-1]}`",
        f"- Horizon-end lower SOH: `{soh_conformal.lower_soh[-1]}`",
        f"- Horizon-end upper SOH: `{soh_conformal.upper_soh[-1]}`",
    ]
    return "\n".join(lines) + "\n"


def execute_generate_advanced_cell_report_tool(
    input_value: GenerateAdvancedCellReportToolInput,
    *,
    context: VerifiedProjectInvocationContext,
    project_audit_ledger: ProjectResultLedger,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Resolve four same-project results and render one fixed audited report."""

    validated = GenerateAdvancedCellReportToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    rul_result = _resolve_project_result(
        project_audit_ledger,
        context,
        validated.rul_result_id,
    )
    soh_result = _resolve_project_result(
        project_audit_ledger,
        context,
        validated.soh_result_id,
    )
    rul_conformal_result = _resolve_project_result(
        project_audit_ledger,
        context,
        validated.rul_conformal_result_id,
    )
    soh_conformal_result = _resolve_project_result(
        project_audit_ledger,
        context,
        validated.soh_conformal_result_id,
    )
    rul, _rul_artifact = _decode_rul(rul_result)
    soh, _soh_artifact = _decode_soh(soh_result)
    rul_conformal, rul_conformal_type = _decode_conformal(
        rul_conformal_result
    )
    soh_conformal, soh_conformal_type = _decode_conformal(
        soh_conformal_result
    )
    if not isinstance(rul_conformal, _RULIntervalEvidence):
        raise ValueError("rul_conformal_result_id must reference a RUL interval")
    if not isinstance(soh_conformal, _SOHBandEvidence):
        raise ValueError("soh_conformal_result_id must reference a SOH band")
    _validate_shared_identity(rul, soh)
    _validate_conformal_binding(
        rul_conformal,
        rul=rul,
        soh_result=soh_result,
        soh=soh,
    )
    _validate_conformal_binding(
        soh_conformal,
        rul=rul,
        soh_result=soh_result,
        soh=soh,
    )
    created_at = _timestamp(clock)
    prediction = rul.prediction
    upstream = (
        rul_result,
        soh_result,
        rul_conformal_result,
        soh_conformal_result,
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.GENERATE_AUDITED_REPORT.value,
        tool_version=ADVANCED_CELL_REPORT_TOOL_VERSION,
        model_version=ADVANCED_CELL_REPORT_MODEL_VERSION,
        data_version=prediction.data_version,
        feature_version=ADVANCED_CELL_REPORT_FEATURE_VERSION,
        input_hash=sha256_canonical(validated.model_dump(mode="json")),
        values={
            "artifact_type": ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
            "artifact": {
                "dataset_id": prediction.dataset_id,
                "cell_id": prediction.cell_id,
                "cutoff_cycle": prediction.cutoff_cycle,
                "split_version": prediction.split_version,
                "rul": _rul_evidence(rul_result, rul),
                "soh": _soh_evidence(soh_result, soh),
                "rul_conformal": _conformal_evidence(
                    rul_conformal_result,
                    rul_conformal,
                    rul_conformal_type,
                ),
                "soh_conformal": _conformal_evidence(
                    soh_conformal_result,
                    soh_conformal,
                    soh_conformal_type,
                ),
                "upstream_result_ids": [
                    result.result_id for result in upstream
                ],
            },
            "markdown": _markdown(
                rul=rul,
                soh=soh,
                rul_conformal=rul_conformal,
                soh_conformal=soh_conformal,
            ),
        },
        uncertainty={
            "rul": rul_conformal_result.uncertainty,
            "soh": soh_conformal_result.uncertainty,
        },
        warnings=[
            "MATR_OFFICIAL_CYCLE_LIFE_IS_NOT_UNIFIED_EOL80",
            "SOH_IS_FINITE_HORIZON_ONLY",
            "CONFORMAL_IS_PREDICTION_INTERVAL_NOT_PARAMETER_CONFIDENCE",
        ],
        provenance=_provenance(upstream),
        created_at=created_at,
    )


def register_project_generate_advanced_cell_report_tool(
    registry: ToolRegistry,
    *,
    project_audit_ledger: ProjectResultLedger,
    clock: Clock = _utc_now,
) -> RegisteredTool[GenerateAdvancedCellReportToolInput]:
    """Register the fixed single-cell report behind the PROJECT boundary."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.GENERATE_AUDITED_REPORT,
            tool_version=ADVANCED_CELL_REPORT_TOOL_VERSION,
            input_model=GenerateAdvancedCellReportToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_generate_advanced_cell_report_tool(
                    input_value,
                    context=context,
                    project_audit_ledger=project_audit_ledger,
                    clock=clock,
                )
            ),
        )
    )


__all__ = [
    "ADVANCED_CELL_REPORT_EVIDENCE_TYPE",
    "ADVANCED_CELL_REPORT_TOOL_VERSION",
    "GenerateAdvancedCellReportToolInput",
    "execute_generate_advanced_cell_report_tool",
    "register_project_generate_advanced_cell_report_tool",
]
