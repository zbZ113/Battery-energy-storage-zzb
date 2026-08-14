from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from quanxin_life.agents.execution_adapter import AgentExecutionContextResolver
from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthCookieConfig, create_auth_http_adapter
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceIdentity,
    AdvancedCalibrationSourceRegistration,
    RegisteredAdvancedCalibrationEvidenceResolver,
)
from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationDispatchReceipt,
)
from quanxin_life.application.advanced_calibration_materialization import (
    AdvancedCalibrationCellInput,
)
from quanxin_life.application.advanced_runtime import (
    RULInferenceAdapter,
    SOHInferenceAdapter,
    VerifiedAdvancedRuntime,
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.assembly import (
    AdvancedCalibrationAssemblyDependencies,
    create_advanced_calibration_components,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactKind,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.audit.project_ledger import BoundProjectResultResolver
from quanxin_life.auth import (
    Argon2idPasswordHasher,
    AuthService,
    PasswordPolicy,
    SqlAlchemyAuthTransactionFactory,
)
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    AgentIntent,
    ProjectStatus,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.data.matr_multibatch import (
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrCellConversionEvidence,
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    AdvancedCalibrationSampleBinding,
    AgentRun,
    Dataset,
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
    RecordBatchBinding,
    SessionRecord,
    User,
)
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE,
    ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE,
    AdvancedSplitConformalToolInput,
    execute_advanced_split_conformal_tool,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_project_report import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    GenerateAdvancedCellReportToolInput,
    execute_generate_advanced_cell_report_tool,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch
from quanxin_life.tools.registry import StandardToolName, ToolRegistry

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
ORIGIN = "https://app.example.test"
PASSWORD = "AdvancedCalibrationE2E-2026!"
PROJECT_ID = "d8bab88e-4d39-4f2a-9352-40a0cde65fd1"
USER_ID = "af3df14a-f32b-41a8-b1ee-95ac89cc45a5"
SESSION_ID = "ff7f0016-9035-4360-b905-a23d42d38f7b"
RUN_ID = "6e5dce4d-dc1a-4635-822a-8c8707d32350"
DATASET_ID = "51c22edf-4568-447f-9c5b-afc8a9e8d5f8"
RECORD_BATCH_ID = "85ff988c-f037-4592-bbc6-633c3904166f"
TARGET_CELL_ID = "matr-test-only-target"
CUTOFF = 100
DATA_VERSION = "matr-three-batch-test-v1"
FEATURE_VERSION = "cyclepatch-multichannel-test-v1"
SPLIT_VERSION = "matr-three-batch-split-v1"
SOURCE_REGISTRATION_ID = "matr-three-batch-test-only-v1"
NORMALIZATION_SHA256 = "2" * 64
_MANIFEST_PATH = Path("reports/data_quality/matr_three_batch_manifest_v1.json")


@dataclass(frozen=True, slots=True)
class _Route:
    task: AdvancedModelTask
    role: AdvancedModelRouteRole
    artifact_id: str
    decision_event_id: str
    artifact_sha256: str
    ledger_sha256: str
    model_version: str
    artifact_kind: DeepArtifactKind
    output_target: AdvancedOutputTarget


RUL_COVERAGE_ROUTE = _Route(
    task=AdvancedModelTask.RUL,
    role=AdvancedModelRouteRole.COVERAGE,
    artifact_id="9f03a6db-c531-4486-a410-c2d76c46e84e",
    decision_event_id="cd94d8f5-0987-46ad-8596-4902760490e0",
    artifact_sha256="3" * 64,
    ledger_sha256="4" * 64,
    model_version="cyclepatch-coverage-test-only-v1",
    artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
    output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
)
RUL_POINT_ROUTE = _Route(
    task=AdvancedModelTask.RUL,
    role=AdvancedModelRouteRole.POINT_ACCURACY,
    artifact_id="37f9e775-0aa6-4b9f-853a-8200167b89b4",
    decision_event_id="1ce2e195-866f-4217-b3cb-21c53f05bdce",
    artifact_sha256="d" * 64,
    ledger_sha256="e" * 64,
    model_version="cyclepatch-point-test-only-v1",
    artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
    output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
)
SOH_ROUTE = _Route(
    task=AdvancedModelTask.SOH,
    role=AdvancedModelRouteRole.MEAN_ACCURACY,
    artifact_id="26f8c5b3-4d4f-4d93-b7cc-a6bed31d4b77",
    decision_event_id="6055090b-849a-4f11-8fbc-b26c3591268e",
    artifact_sha256="5" * 64,
    ledger_sha256="6" * 64,
    model_version="current-hybrid-test-only-v1",
    artifact_kind=DeepArtifactKind.CURRENT_HYBRID,
    output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
)
CALIBRATION_ROUTES = (RUL_COVERAGE_ROUTE, SOH_ROUTE)
ACTIVE_ROUTES = (RUL_COVERAGE_ROUTE, RUL_POINT_ROUTE, SOH_ROUTE)


def test_project_advanced_calibration_reaches_registered_report_over_http(
    tmp_path: Path,
) -> None:
    """Prove the production-assembled HTTP-to-report chain with test-only data."""

    evidence = _build_registered_matr_evidence(tmp_path / "matr-evidence")
    session_factory = _database(tmp_path / "advanced-calibration-e2e.sqlite3")
    hasher = Argon2idPasswordHasher()
    _seed_project(
        session_factory,
        credential_hash=hasher.hash_password(PASSWORD),
    )
    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    ledger = SqlProjectAuditLedger(
        session_factory,
        context_validator=context_service,
    )
    auth_service = AuthService(
        transactions=SqlAlchemyAuthTransactionFactory(session_factory),
        password_hasher=hasher,
        password_policy=PasswordPolicy(),
        session_ttl=timedelta(hours=12),
    )
    auth_adapter = create_auth_http_adapter(
        auth_service,
        AuthCookieConfig(
            environment="production",
            allowed_origins=(ORIGIN,),
        ),
    )
    runtime_resolver = _RuntimeResolver(
        tuple(_runtime(route) for route in ACTIVE_ROUTES)
    )
    queue = _IdentityOnlyQueue()
    components = create_advanced_calibration_components(
        AdvancedCalibrationAssemblyDependencies(
            session_factory=session_factory,
            context_service=context_service,
            evidence_resolver=evidence.resolver,
            runtime_resolver=runtime_resolver,
            cell_input_resolver=_TestCellInputResolver(evidence.resolver),
            project_materializer=ledger,
            queue=queue,
            auth_adapter=auth_adapter,
            agent_context_delegate=cast(
                AgentExecutionContextResolver,
                _NoopDelegate(),
            ),
            target_record_batch_resolver=_RecordBatchResolver(_target_batch()),
        )
    )
    tool_service = ToolInvocationService(
        registry=ToolRegistry(project_context_validator=context_service),
        project_audit_ledger=ledger,
    )
    app = create_fastapi_app(
        tool_service,
        auth_adapter=auth_adapter,
        advanced_calibration_adapter=components.http_adapter,
        project_invocation_context_service=context_service,
    )
    client = TestClient(app, base_url="https://api.example.test")
    _login(client)

    materialization_ids: dict[AdvancedModelTask, str] = {}
    for route in CALIBRATION_ROUTES:
        created = client.post(
            _materialization_path(),
            headers={
                "Origin": ORIGIN,
                "Idempotency-Key": (
                    f"test-only-{route.task.value.lower()}-materialization"
                ),
            },
            json={
                "task": route.task.value,
                "cutoff_cycle": CUTOFF,
                "route_role": route.role.value,
                "source_registration_id": SOURCE_REGISTRATION_ID,
            },
        )
        assert created.status_code == 202, created.json()
        created_payload = created.json()
        materialization_id = created_payload["materialization"][
            "materialization_id"
        ]
        materialization_ids[route.task] = materialization_id
        assert created_payload["materialization"]["status"] == "PENDING"
        assert queue.calls[-1] == {"materialization_id": materialization_id}
        assert _recursive_keys(created_payload).isdisjoint(
            {
                "cell_id",
                "sample_ids",
                "result_ids",
                "observed_soh",
                "predicted_soh",
            }
        )

        worker_status = components.worker.execute(
            materialization_id=materialization_id
        )
        worker_summary = client.get(
            f"{_materialization_path()}/{materialization_id}"
        ).json()
        assert (
            worker_status is AdvancedCalibrationMaterializationStatus.READY
        ), worker_summary["failure_code"]
        ready = client.get(f"{_materialization_path()}/{materialization_id}")
        assert ready.status_code == 200
        ready_payload = ready.json()
        assert ready_payload["status"] == "READY"
        assert ready_payload["sample_count"] == 9
        assert len(ready_payload["sample_manifest_sha256"]) == 64
        assert _recursive_keys(ready_payload).isdisjoint(
            {
                "cell_id",
                "sample_ids",
                "result_ids",
                "observed_soh",
                "predicted_soh",
            }
        )

    assert queue.calls == [
        {"materialization_id": materialization_ids[AdvancedModelTask.RUL]},
        {"materialization_id": materialization_ids[AdvancedModelTask.SOH]},
    ]
    _assert_persisted_ready_cohorts(
        session_factory,
        materialization_ids=materialization_ids,
    )

    rul_sample_ids = cast(
        tuple[str, ...],
        components.agent_context_resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.rul_calibration_sample_result_ids",
        ),
    )
    soh_sample_ids = cast(
        tuple[str, ...],
        components.agent_context_resolver.resolve_context(
            run_id=RUN_ID,
            project_id=PROJECT_ID,
            reference="context.soh_calibration_sample_result_ids",
        ),
    )
    assert rul_sample_ids == _persisted_sample_ids(
        session_factory,
        materialization_ids[AdvancedModelTask.RUL],
    )
    assert soh_sample_ids == _persisted_sample_ids(
        session_factory,
        materialization_ids[AdvancedModelTask.SOH],
    )

    admin_context = context_service.resolve_http(
        _authenticated_principal(session_factory),
        PROJECT_ID,
    )
    agent_context = context_service.resolve_agent_run(RUN_ID)
    agent_results = BoundProjectResultResolver(ledger, agent_context)
    rul_calibration = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="calibrate",
            task=AdvancedModelTask.RUL,
            route_role=RUL_COVERAGE_ROUTE.role,
            alpha=0.1,
            calibration_sample_result_ids=rul_sample_ids,
        ),
        context=agent_context,
        result_resolver=agent_results,
        clock=lambda: NOW,
    )
    soh_calibration = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="calibrate",
            task=AdvancedModelTask.SOH,
            route_role=SOH_ROUTE.role,
            alpha=0.1,
            calibration_sample_result_ids=soh_sample_ids,
        ),
        context=agent_context,
        result_resolver=agent_results,
        clock=lambda: NOW,
    )
    ledger.register_result(admin_context, rul_calibration)
    ledger.register_result(admin_context, soh_calibration)
    rul_coverage_prediction = ledger.register_result(
        admin_context,
        _target_prediction(RUL_COVERAGE_ROUTE),
    )
    rul_point_prediction = ledger.register_result(
        admin_context,
        _target_prediction(RUL_POINT_ROUTE),
    )
    soh_prediction = ledger.register_result(
        admin_context,
        _target_prediction(SOH_ROUTE),
    )
    rul_interval = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="issue",
            task=AdvancedModelTask.RUL,
            route_role=RUL_COVERAGE_ROUTE.role,
            prediction_result_id=rul_coverage_prediction.result_id,
            calibration_result_id=rul_calibration.result_id,
        ),
        context=agent_context,
        result_resolver=agent_results,
        clock=lambda: NOW,
    )
    soh_band = execute_advanced_split_conformal_tool(
        AdvancedSplitConformalToolInput(
            operation="issue",
            task=AdvancedModelTask.SOH,
            route_role=SOH_ROUTE.role,
            prediction_result_id=soh_prediction.result_id,
            calibration_result_id=soh_calibration.result_id,
        ),
        context=agent_context,
        result_resolver=agent_results,
        clock=lambda: NOW,
    )
    ledger.register_result(admin_context, rul_interval)
    ledger.register_result(admin_context, soh_band)
    assert (
        rul_interval.values["artifact_type"]
        == ADVANCED_RUL_SPLIT_INTERVAL_EVIDENCE_TYPE
    )
    assert (
        soh_band.values["artifact_type"]
        == ADVANCED_SOH_SPLIT_BAND_EVIDENCE_TYPE
    )

    report = execute_generate_advanced_cell_report_tool(
        GenerateAdvancedCellReportToolInput(
            rul_result_id=rul_point_prediction.result_id,
            soh_result_id=soh_prediction.result_id,
            rul_conformal_result_id=rul_interval.result_id,
            soh_conformal_result_id=soh_band.result_id,
        ),
        context=agent_context,
        project_audit_ledger=ledger,
        clock=lambda: NOW,
    )
    registered_report = ledger.register_result(admin_context, report)
    resolved_report = BoundProjectResultResolver(
        ledger,
        agent_context,
    ).resolve_registered_result(registered_report.result_id)
    assert resolved_report.result_id == registered_report.result_id
    assert (
        resolved_report.values["artifact_type"]
        == ADVANCED_CELL_REPORT_EVIDENCE_TYPE
    )
    assert resolved_report.values["artifact"]["upstream_result_ids"] == [
        rul_point_prediction.result_id,
        soh_prediction.result_id,
        rul_interval.result_id,
        soh_band.result_id,
    ]

    report_response = client.get(
        f"/v1/projects/{PROJECT_ID}/results/{registered_report.result_id}"
    )
    assert report_response.status_code == 200
    report_payload = report_response.json()
    assert report_payload["result_id"] == registered_report.result_id
    assert report_payload["values"]["artifact"]["cell_id"] == TARGET_CELL_ID
    assert report_payload["values"]["artifact"]["upstream_result_ids"] == (
        resolved_report.values["artifact"]["upstream_result_ids"]
    )


@dataclass(frozen=True, slots=True)
class _RegisteredMatrEvidence:
    root: Path
    manifest_sha256: str
    resolver: RegisteredAdvancedCalibrationEvidenceResolver


def _build_registered_matr_evidence(root: Path) -> _RegisteredMatrEvidence:
    root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    references: list[MatrBatchArtifactReference] = []
    total_cells = 0
    for batch_index, batch_date in (
        (1, date(2017, 5, 12)),
        (2, date(2017, 6, 30)),
        (3, date(2018, 4, 12)),
    ):
        cell_ids = tuple(
            f"MATR_b{batch_index}c{cell_index}"
            for cell_index in range(6)
        )
        split = SplitManifest(
            dataset_id="MATR",
            seed=20260727,
            train=(cell_ids[1],),
            validation=(cell_ids[2],),
            calibration=(cell_ids[0], cell_ids[3], cell_ids[4]),
            test=(cell_ids[5],),
        )
        for partition in combined:
            combined[partition].extend(getattr(split, partition))
        conversion = MatrBatchConversionReport(
            batch_index=batch_index,
            batch_date=batch_date,
            raw_relative_path=f"data/raw-batch-{batch_index}.mat",
            raw_size_bytes=1024,
            raw_sha256=_digest(f"raw-{batch_index}"),
            source_uri=f"https://example.test/matr/{batch_index}",
            license_name="test-only",
            adapter_version="matr-e2e-test-only-v1",
            time_unit="minutes",
            max_cycle_index=150,
            cell_count=len(cell_ids),
            total_row_count=len(cell_ids) * 600,
            quality_issue_counts={},
            cells=tuple(
                MatrCellConversionEvidence(
                    cell_id=cell_id,
                    raw_cell_id=f"b{batch_index}c{cell_index}",
                    official_life_label=610 + batch_index * 10 + cell_index,
                    official_life_right_censored=False,
                    protocol_id=f"protocol-{batch_index}",
                    reference_capacity_ah=1.0,
                    row_count=600,
                    cycle_count=150,
                    quality_issue_counts={},
                    manifest_relative_path=f"cells/{cell_id}/manifest.json",
                    parquet_sha256=_digest(f"processed-parquet-{cell_id}"),
                    metadata_sha256=_digest(f"processed-metadata-{cell_id}"),
                )
                for cell_index, cell_id in enumerate(cell_ids)
            ),
            created_at=NOW,
        )
        conversion_path = (
            root / f"reports/data_quality/conversion-{batch_index}.json"
        )
        split_path = root / f"configs/data_splits/split-{batch_index}.json"
        eligibility_path = (
            root / f"reports/data_quality/eligibility-{batch_index}.json"
        )
        supervision_path = (
            root / f"reports/data_quality/supervision-{batch_index}.json"
        )
        processed_root = root / f"data/processed-{batch_index}"
        processed_root.mkdir(parents=True)
        supervision_root = root / f"data/supervision-{batch_index}"
        parquet_path = supervision_root / "trajectories/labels.parquet"
        _write_json(conversion_path, conversion.model_dump(mode="json"))
        _write_json(split_path, split.model_dump(mode="json"))
        eligibility = MatrTrajectoryEligibilityAudit(
            batch_index=batch_index,
            horizon_cycle=500,
            eligible_cell_ids=cell_ids,
            excluded=(),
            created_at=NOW,
        )
        _write_json(eligibility_path, eligibility.model_dump(mode="json"))
        _write_supervision_parquet(parquet_path, cell_ids)
        supervision = MatrSupervisionArtifact(
            source_report_sha256=sha256_canonical(
                conversion.model_dump(mode="json")
            ),
            raw_sha256=conversion.raw_sha256,
            horizon_cycle=500,
            cell_count=len(cell_ids),
            row_count=len(cell_ids) * 500,
            parquet_relative_path="trajectories/labels.parquet",
            parquet_sha256=_sha256_file(parquet_path),
            cells=tuple(
                MatrSupervisionCellEvidence(
                    cell_id=cell_id,
                    raw_cell_id=f"b{batch_index}c{cell_index}",
                    official_life_label=610 + batch_index * 10 + cell_index,
                    official_life_right_censored=False,
                    reference_capacity_ah=1.0,
                    observed_cycle_count=700,
                )
                for cell_index, cell_id in enumerate(cell_ids)
            ),
            created_at=NOW,
        )
        _write_json(supervision_path, supervision.model_dump(mode="json"))
        references.append(
            MatrBatchArtifactReference(
                batch_index=batch_index,
                batch_date=batch_date,
                raw_relative_path=f"data/raw-batch-{batch_index}.mat",
                raw_manifest=f"configs/data_manifests/raw-{batch_index}.json",
                raw_sha256=conversion.raw_sha256,
                processed_root=processed_root.relative_to(root).as_posix(),
                conversion_report=conversion_path.relative_to(root).as_posix(),
                conversion_report_sha256=_sha256_file(conversion_path),
                split_manifest=split_path.relative_to(root).as_posix(),
                split_manifest_sha256=_sha256_file(split_path),
                supervision_root=supervision_root.relative_to(root).as_posix(),
                supervision_report=supervision_path.relative_to(root).as_posix(),
                supervision_report_sha256=_sha256_file(supervision_path),
                eligibility_report=eligibility_path.relative_to(root).as_posix(),
                eligibility_report_sha256=_sha256_file(eligibility_path),
                cell_count=len(cell_ids),
                scalar_label_count=len(cell_ids),
                hybrid_eligible_count=len(cell_ids),
                hybrid_excluded_count=0,
            )
        )
        total_cells += len(cell_ids)
    combined_split = SplitManifest(
        dataset_id="MATR",
        seed=20260727,
        train=tuple(combined["train"]),
        validation=tuple(combined["validation"]),
        calibration=tuple(combined["calibration"]),
        test=tuple(combined["test"]),
    )
    combined_path = root / "configs/data_splits/combined.json"
    _write_json(combined_path, combined_split.model_dump(mode="json"))
    manifest = MatrThreeBatchManifest(
        data_version=DATA_VERSION,
        split_version=SPLIT_VERSION,
        combined_split_manifest=combined_path.relative_to(root).as_posix(),
        combined_split_sha256=_sha256_file(combined_path),
        batches=tuple(references),
        total_cell_count=total_cells,
        scalar_label_count=total_cells,
        hybrid_eligible_count=total_cells,
        hybrid_excluded_count=0,
        created_at=NOW,
    )
    manifest_path = root / _MANIFEST_PATH
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    manifest_sha256 = _sha256_file(manifest_path)
    resolver = RegisteredAdvancedCalibrationEvidenceResolver(
        (
            AdvancedCalibrationSourceRegistration(
                registration_id=SOURCE_REGISTRATION_ID,
                evidence_root=root,
                three_batch_manifest_sha256=manifest_sha256,
            ),
        )
    )
    return _RegisteredMatrEvidence(
        root=root,
        manifest_sha256=manifest_sha256,
        resolver=resolver,
    )


def _write_supervision_parquet(
    path: Path,
    cell_ids: tuple[str, ...],
) -> None:
    rows: list[dict[str, object]] = []
    for cell_index, cell_id in enumerate(cell_ids):
        for cycle in range(1, 501):
            soh = 0.999 - cycle / 10000 - cell_index / 100000
            rows.append(
                {
                    "dataset_id": "MATR",
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "discharge_capacity_ah": soh,
                    "reference_capacity_ah": 1.0,
                    "soh": soh,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class _TestCellInputResolver:
    """Use the registered source identity, with explicit test-only tensors."""

    def __init__(
        self,
        source_resolver: RegisteredAdvancedCalibrationEvidenceResolver,
    ) -> None:
        self._source_resolver = source_resolver

    def resolve(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        task: AdvancedModelTask,
        cell_id: str,
        cutoff_cycle: int,
        feature_version: str,
    ) -> AdvancedCalibrationCellInput:
        source = self._source_resolver.resolve_cell_source(
            source_registration_id,
            source_identity=source_identity,
            task=task,
            cutoff_cycle=cutoff_cycle,
            cell_id=cell_id,
        )
        if source.source_identity != source_identity:
            raise ValueError("test-only cell source identity changed")
        return AdvancedCalibrationCellInput(
            source_registration_id=source_registration_id,
            source_identity_sha256=source_identity.source_identity_sha256,
            cell_id=cell_id,
            cutoff_cycle=cutoff_cycle,
            data_version=source_identity.data_version,
            feature_version=feature_version,
            split_version=source_identity.split_version,
            raw_sequence=_sequence(cell_id),
            initial_soh=source.initial_soh,
        )


@dataclass(frozen=True, slots=True)
class _RULTestInference:
    normalization_statistics_sha256: str = NORMALIZATION_SHA256

    def predict_cycle(self, raw_sequence: EarlyCycleSequence) -> float:
        raw_sequence.verify_input_hash()
        return 650.0


@dataclass(frozen=True, slots=True)
class _SOHTestInference:
    normalization_statistics_sha256: str = NORMALIZATION_SHA256

    def predict_trajectory(
        self,
        raw_sequence: EarlyCycleSequence,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        raw_sequence.verify_input_hash()
        cycles = tuple(range(CUTOFF + 1, 501))
        values = tuple(
            initial_soh - index * 0.0004 for index in range(len(cycles))
        )
        return cycles, values


class _RuntimeResolver:
    def __init__(self, runtimes: tuple[VerifiedAdvancedRuntime, ...]) -> None:
        self._runtimes = {
            (runtime.task, runtime.cutoff_cycle, runtime.role): runtime
            for runtime in runtimes
        }

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedAdvancedRuntime:
        if context.project_id != PROJECT_ID:
            raise LookupError("runtime is outside the test project")
        return self._runtimes[(task, cutoff_cycle, role)]

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
        runtime: VerifiedAdvancedRuntime,
    ) -> VerifiedAdvancedRuntime:
        current = self.resolve(
            context,
            task=runtime.task,
            cutoff_cycle=runtime.cutoff_cycle,
            role=runtime.role,
        )
        if current != runtime:
            raise ValueError("test runtime identity changed")
        return current


class _IdentityOnlyQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def enqueue(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationDispatchReceipt:
        payload = {"materialization_id": materialization_id}
        self.calls.append(payload)
        return AdvancedCalibrationDispatchReceipt(
            materialization_id=materialization_id,
            task_id=f"test-only-calibration-{len(self.calls):04d}",
        )


class _RecordBatchResolver:
    def __init__(self, batch: VerifiedEarlyCycleBatch) -> None:
        self._batch = batch

    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        if (
            context.project_id != PROJECT_ID
            or record_batch_id != RECORD_BATCH_ID
        ):
            raise LookupError("test record batch is unavailable")
        return self._batch


class _NoopDelegate:
    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        del run_id, project_id, dataset_id
        raise AssertionError("calibration E2E must use the assembled resolver")

    def resolve_context(
        self,
        *,
        run_id: str,
        project_id: str,
        reference: str,
    ) -> object:
        del run_id, project_id, reference
        raise AssertionError("calibration E2E must not delegate its references")


def _database(path: Path) -> SessionFactory:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{path}")
    )
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def _seed_project(
    session_factory: SessionFactory,
    *,
    credential_hash: str,
) -> None:
    intent = AgentIntent(
        intent_id=str(uuid4()),
        project_id=PROJECT_ID,
        goal="test-only Advanced calibration vertical workflow",
        dataset_ids=(RECORD_BATCH_ID,),
        requested_outputs=("advanced_single_cell_analysis",),
        created_at=NOW,
    )
    with session_factory.begin() as session:
        session.add(
            User(
                id=USER_ID,
                username="advanced-calibration-e2e@example.test",
                credential_hash=credential_hash,
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=SESSION_ID,
                user_id=USER_ID,
                token_hash="8" * 64,
                status=SessionStatus.ACTIVE.value,
                created_at=NOW,
                expires_at=NOW + timedelta(hours=2),
            )
        )
        session.add(
            Project(
                id=PROJECT_ID,
                owner_user_id=USER_ID,
                name="Advanced calibration E2E",
                status=ProjectStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Dataset(
                id=DATASET_ID,
                project_id=PROJECT_ID,
                name="Test-only frozen MATR target",
                data_version=DATA_VERSION,
                schema_version="canonical-cycle-v1",
                status="FROZEN",
                manifest_uri="test-only://matr-target/manifest",
                manifest_sha256="9" * 64,
                created_at=NOW,
                frozen_at=NOW,
            )
        )
        session.add(
            RecordBatchBinding(
                id=RECORD_BATCH_ID,
                binding_schema_version="record-batch-binding-v1",
                content_batch_id="sha256:" + "a" * 64,
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID,
                source_manifest_sha256="b" * 64,
                registration_sha256="c" * 64,
                content_dataset_id="MATR",
                dataset_schema_version="canonical-cycle-v1",
                cell_id=TARGET_CELL_ID,
                cutoff_cycle=CUTOFF,
                data_version=DATA_VERSION,
                split_version=SPLIT_VERSION,
                feature_version=FEATURE_VERSION,
                created_by_user_id=USER_ID,
                created_at=NOW,
            )
        )
        session.add(
            AgentRun(
                id=RUN_ID,
                project_id=PROJECT_ID,
                session_id=SESSION_ID,
                created_by_user_id=USER_ID,
                idempotency_key_hash="d" * 64,
                request_hash="e" * 64,
                status="RUNNING",
                planning_mode="FALLBACK",
                intent_json=intent.model_dump(mode="json"),
                plan_json={"schema_version": "test-only-agent-plan"},
                plan_hash="f" * 64,
                execution_plan_hash=None,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        for route in ACTIVE_ROUTES:
            _add_route(session, route)


def _add_route(session: Session, route: _Route) -> None:
    session.add(
        ModelArtifact(
            id=route.artifact_id,
            project_id=PROJECT_ID,
            artifact_format="safetensors",
            object_uri=f"artifact://advanced-model/{route.artifact_id}",
            sha256=route.artifact_sha256,
            status="VERIFIED",
            created_at=NOW,
        )
    )
    session.add(
        ModelManifest(
            id=str(uuid4()),
            artifact_id=route.artifact_id,
            model_version=route.model_version,
            manifest_uri=f"artifact://advanced-model/{route.artifact_id}/manifest",
            manifest_sha256=route.artifact_sha256,
            metadata_json={"schema_version": "deep-model-artifact-v2"},
            created_at=NOW,
        )
    )
    session.add(
        ModelRouteActivationEvent(
            id=route.decision_event_id,
            project_id=PROJECT_ID,
            stream_sequence=1,
            task=route.task.value,
            cutoff_cycle=CUTOFF,
            route_role=route.role.value,
            decision_type="ACTIVATE",
            artifact_id=route.artifact_id,
            artifact_sha256=route.artifact_sha256,
            manifest_sha256=route.artifact_sha256,
            deployment_bundle_manifest_sha256="7" * 64,
            route_provenance_sha256="8" * 64,
            rollback_target_event_id=None,
            previous_event_sha256="9" * 64,
            event_sha256=route.ledger_sha256,
            actor_user_id=USER_ID,
            reason="Activate test-only E2E route",
            idempotency_key_sha256=sha256_canonical(
                {
                    "activation": route.task.value,
                    "cutoff_cycle": CUTOFF,
                    "route_role": route.role.value,
                }
            ),
            request_sha256=sha256_canonical(
                {
                    "activation_request": route.task.value,
                    "cutoff_cycle": CUTOFF,
                    "route_role": route.role.value,
                }
            ),
            created_at=NOW,
        )
    )
    session.add(
        ModelRouteActivationStreamHead(
            project_id=PROJECT_ID,
            task=route.task.value,
            cutoff_cycle=CUTOFF,
            route_role=route.role.value,
            head_event_id=route.decision_event_id,
            head_sequence=1,
            head_event_sha256=route.ledger_sha256,
            updated_at=NOW,
        )
    )


def _runtime(route: _Route) -> VerifiedAdvancedRuntime:
    if route.task is AdvancedModelTask.RUL:
        return VerifiedRULRuntime(
            project_id=PROJECT_ID,
            task=route.task,
            cutoff_cycle=CUTOFF,
            role=route.role,
            output_target=route.output_target,
            artifact_kind=route.artifact_kind,
            dataset_id="MATR",
            data_version=DATA_VERSION,
            feature_version=FEATURE_VERSION,
            split_version=SPLIT_VERSION,
            normalization_sha256=NORMALIZATION_SHA256,
            artifact_id=route.artifact_id,
            artifact_manifest_sha256=route.artifact_sha256,
            model_version=route.model_version,
            decision_event_id=route.decision_event_id,
            ledger_sequence_number=1,
            ledger_head_sha256=route.ledger_sha256,
            inference=cast(RULInferenceAdapter, _RULTestInference()),
        )
    return VerifiedSOHRuntime(
        project_id=PROJECT_ID,
        task=route.task,
        cutoff_cycle=CUTOFF,
        role=route.role,
        output_target=route.output_target,
        artifact_kind=route.artifact_kind,
        dataset_id="MATR",
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        normalization_sha256=NORMALIZATION_SHA256,
        artifact_id=route.artifact_id,
        artifact_manifest_sha256=route.artifact_sha256,
        model_version=route.model_version,
        decision_event_id=route.decision_event_id,
        ledger_sequence_number=1,
        ledger_head_sha256=route.ledger_sha256,
        inference=cast(SOHInferenceAdapter, _SOHTestInference()),
    )


def _sequence(cell_id: str) -> EarlyCycleSequence:
    sample_mask = torch.ones((CUTOFF + 1, 2, 150), dtype=torch.bool)
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=CUTOFF,
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        cycle_indices=tuple(range(CUTOFF + 1)),
        values=torch.ones(
            (CUTOFF + 1, 2, 150, 3),
            dtype=torch.float32,
        ),
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=("temperature_c", "charge_rate_c"),
        condition_values=torch.tensor([25.0, 1.0]),
        condition_mask=torch.tensor([True, True]),
    )


def _target_batch() -> VerifiedEarlyCycleBatch:
    records = tuple(
        CycleRecord(
            dataset_id="MATR",
            cell_id=TARGET_CELL_ID,
            cycle_index=cycle,
            sample_index=index,
            time_s=float(index),
            voltage_v=3.2,
            current_a=-1.0,
            temperature_c=25.0,
            charge_capacity_ah=1.1,
            discharge_capacity_ah=capacity,
            internal_resistance_ohm=0.01,
            diagnostic=True,
            valid=True,
        )
        for index, (cycle, capacity) in enumerate(((1, 1.1), (CUTOFF, 1.0)))
    )
    return VerifiedEarlyCycleBatch.model_validate(
        {
            "record_batch_id": RECORD_BATCH_ID,
            "records": records,
            "metadata": {
                "dataset_id": "MATR",
                "cell_id": TARGET_CELL_ID,
                "chemistry": "LFP/graphite",
                "nominal_capacity_ah": 1.1,
                "source_uri": "test-only://matr-target",
                "source_sha256": "a" * 64,
                "schema_version": "cycle-record-v1",
            },
            "feature_config": EarlyCycleFeatureConfig(
                cutoff_cycle=CUTOFF,
                feature_version=FEATURE_VERSION,
            ),
            "data_version": DATA_VERSION,
            "split_version": SPLIT_VERSION,
            "source_manifest_hash": "a" * 64,
            "provenance": (
                ProvenanceRecord(
                    source_id="test-only-target",
                    source_kind=SourceKind.OBSERVED,
                    uri="test-only://matr-target",
                    sha256="a" * 64,
                    description="Explicit test-only verified target batch",
                    created_at=NOW,
                ),
            ),
        }
    )


def _target_prediction(route: _Route) -> ToolResult:
    runtime: dict[str, object] = {
        "task": route.task.value,
        "route_role": route.role.value,
        "output_target": route.output_target.value,
        "artifact_kind": _artifact_kind(route),
        "artifact_id": route.artifact_id,
        "artifact_manifest_sha256": route.artifact_sha256,
        "model_version": route.model_version,
        "dataset_id": "MATR",
        "cutoff_cycle": CUTOFF,
        "data_version": DATA_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "normalization_statistics_sha256": NORMALIZATION_SHA256,
        "decision_event_id": route.decision_event_id,
        "ledger_sequence_number": 1,
        "ledger_head_sha256": route.ledger_sha256,
    }
    artifact: dict[str, object]
    if route.task is AdvancedModelTask.RUL:
        artifact_type = ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1
        tool_name = StandardToolName.PREDICT_CYCLE_LIFE
        tool_version = ADVANCED_RUL_PREDICTION_TOOL_VERSION
        artifact = {
            **runtime,
            "record_batch_id": RECORD_BATCH_ID,
            "cell_id": TARGET_CELL_ID,
            "cycle_life_prediction": {
                "dataset_id": "MATR",
                "cell_id": TARGET_CELL_ID,
                "cutoff_cycle": CUTOFF,
                "target": "matr_official_cycle_life",
                "predicted_cycle": 680.0,
                "observed_cycle": None,
                "right_censored": True,
                "feature_version": FEATURE_VERSION,
                "split_version": SPLIT_VERSION,
                "model_version": route.model_version,
                "data_version": DATA_VERSION,
            },
            "derived_remaining_cycles": 580.0,
            "upstream_result_id": str(uuid4()),
            "raw_sequence_input_sha256": "a" * 64,
            "transform_config_sha256": "b" * 64,
            "source_manifest_hash": "c" * 64,
        }
    else:
        cycles = list(range(CUTOFF + 1, 501))
        artifact_type = ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1
        tool_name = StandardToolName.PREDICT_SOH_TRAJECTORY
        tool_version = ADVANCED_SOH_PREDICTION_TOOL_VERSION
        artifact = {
            **runtime,
            "record_batch_id": RECORD_BATCH_ID,
            "cell_id": TARGET_CELL_ID,
            "prediction_cycles": cycles,
            "predicted_soh": [
                0.98 - index * 0.0005 for index in range(len(cycles))
            ],
            "horizon_end_cycle": 500,
            "upstream_result_id": str(uuid4()),
            "raw_sequence_input_sha256": "a" * 64,
            "transform_config_sha256": "b" * 64,
            "source_manifest_hash": "c" * 64,
        }
    semantic = {
        "test_only": True,
        "artifact_type": artifact_type,
        "artifact": artifact,
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version=tool_version,
        model_version=route.model_version,
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        input_hash=sha256_canonical(semantic),
        values={"artifact_type": artifact_type, "artifact": artifact},
        uncertainty=None,
        warnings=["TEST_ONLY_DETERMINISTIC_PREDICTION"],
        provenance=[
            ProvenanceRecord(
                source_id=f"test-only-target-{route.task.value}",
                source_kind=SourceKind.OBSERVED,
                uri="test-only://matr-target",
                sha256="a" * 64,
                description="Explicit test-only target input",
                created_at=NOW,
            ),
            ProvenanceRecord(
                source_id=f"advanced-model-{route.artifact_id}",
                source_kind=SourceKind.PREDICTED,
                uri=f"artifact://advanced-model/{route.artifact_id}",
                sha256=route.artifact_sha256,
                description="Verified test-only active Advanced runtime",
                created_at=NOW,
            ),
        ],
        created_at=NOW,
    )


def _login(client: TestClient) -> None:
    response = client.post(
        "/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={
            "username": "advanced-calibration-e2e@example.test",
            "password": PASSWORD,
        },
    )
    assert response.status_code == 200


def _authenticated_principal(session_factory: SessionFactory) -> Any:
    from quanxin_life.auth import AuthPrincipal

    with session_factory() as session:
        active_session = session.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.user_id == USER_ID,
                SessionRecord.id != SESSION_ID,
                SessionRecord.status == SessionStatus.ACTIVE.value,
            )
            .order_by(SessionRecord.created_at.desc())
        )
    assert active_session is not None
    return AuthPrincipal(
        user_id=USER_ID,
        session_id=active_session.id,
        username="advanced-calibration-e2e@example.test",
        role=UserRole.ADMIN,
        must_change_password=False,
    )


def _assert_persisted_ready_cohorts(
    session_factory: SessionFactory,
    *,
    materialization_ids: dict[AdvancedModelTask, str],
) -> None:
    with session_factory() as session:
        rows = tuple(
            session.scalars(
                select(AdvancedCalibrationMaterialization).where(
                    AdvancedCalibrationMaterialization.id.in_(
                        tuple(materialization_ids.values())
                    )
                )
            )
        )
        bindings = tuple(
            session.scalars(
                select(AdvancedCalibrationSampleBinding).where(
                    AdvancedCalibrationSampleBinding.materialization_id.in_(
                        tuple(materialization_ids.values())
                    )
                )
            )
        )
    assert len(rows) == 2
    assert all(
        row.status == AdvancedCalibrationMaterializationStatus.READY.value
        and row.sample_count == 9
        and row.sample_manifest_sha256 is not None
        and len(row.sample_manifest_sha256) == 64
        for row in rows
    )
    assert len(bindings) == 18
    assert all(len(binding.sample_sha256) == 64 for binding in bindings)


def _persisted_sample_ids(
    session_factory: SessionFactory,
    materialization_id: str,
) -> tuple[str, ...]:
    with session_factory() as session:
        return tuple(
            session.scalars(
                select(AdvancedCalibrationSampleBinding.result_id)
                .where(
                    AdvancedCalibrationSampleBinding.materialization_id
                    == materialization_id
                )
                .order_by(AdvancedCalibrationSampleBinding.ordinal)
            )
        )


def _materialization_path() -> str:
    return (
        f"/v1/projects/{PROJECT_ID}/advanced-calibration/materializations"
    )


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_recursive_keys(item))
        return keys
    if isinstance(value, list):
        nested_keys: set[str] = set()
        for item in value:
            nested_keys.update(_recursive_keys(item))
        return nested_keys
    return set()


def _artifact_kind(route: _Route) -> str:
    return {
        DeepArtifactKind.CYCLEPATCH_DIRECT: "cyclepatch_direct",
        DeepArtifactKind.CURRENT_HYBRID: "current_hybrid",
    }[route.artifact_kind]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
