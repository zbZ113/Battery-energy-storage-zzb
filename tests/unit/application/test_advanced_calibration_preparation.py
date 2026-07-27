from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceIdentity,
    AdvancedRULCalibrationEvidence,
    AdvancedRULObservedCell,
)
from quanxin_life.application.advanced_calibration_materialization import (
    AdvancedCalibrationMaterializationRequest,
)
from quanxin_life.application.advanced_calibration_preparation import (
    ActiveAdvancedCalibrationPreparationResolver,
)
from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask


@dataclass(frozen=True, slots=True)
class _Runtime:
    project_id: str = "project-1"
    task: AdvancedModelTask = AdvancedModelTask.RUL
    cutoff_cycle: int = 100
    role: AdvancedModelRouteRole = AdvancedModelRouteRole.COVERAGE
    dataset_id: str = "MATR"
    data_version: str = "matr-data-v1"
    split_version: str = "matr-split-v1"
    feature_version: str = "advanced-feature-v1"
    artifact_id: str = "artifact-1"
    artifact_manifest_sha256: str = "1" * 64
    model_version: str = "model-v1"
    normalization_sha256: str = "2" * 64
    decision_event_id: str = "event-1"
    ledger_sequence_number: int = 7
    ledger_head_sha256: str = "3" * 64


class _RuntimeResolver:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[object, object, int, object]] = []

    def resolve(
        self,
        context: object,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> _Runtime:
        self.calls.append((context, task, cutoff_cycle, role))
        return self.runtime


class _EvidenceResolver:
    def __init__(self, evidence: AdvancedRULCalibrationEvidence) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, object, int]] = []

    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedRULCalibrationEvidence:
        self.calls.append((registration_id, task, cutoff_cycle))
        return self.evidence


def test_freezes_exact_runtime_and_registered_source_identity() -> None:
    request = _request()
    context = cast(
        VerifiedProjectInvocationContext,
        SimpleNamespace(project_id=request.project_id),
    )
    runtime_resolver = _RuntimeResolver(_Runtime(project_id=request.project_id))
    evidence_resolver = _EvidenceResolver(_evidence())
    resolver = ActiveAdvancedCalibrationPreparationResolver(
        runtime_resolver=cast(Any, runtime_resolver),
        evidence_resolver=cast(Any, evidence_resolver),
    )

    preparation = resolver.resolve(context, request)

    assert preparation.data_version == "matr-data-v1"
    assert preparation.split_version == "matr-split-v1"
    assert preparation.feature_version == "advanced-feature-v1"
    assert preparation.artifact_id == "artifact-1"
    assert preparation.artifact_manifest_sha256 == "1" * 64
    assert preparation.model_version == "model-v1"
    assert preparation.normalization_statistics_sha256 == "2" * 64
    assert preparation.decision_event_id == "event-1"
    assert preparation.ledger_sequence_number == 7
    assert preparation.ledger_head_sha256 == "3" * 64
    assert preparation.source_registration_id == "matr-three-batch-final-v1"
    assert preparation.source_identity_sha256 == "4" * 64
    assert runtime_resolver.calls == [
        (
            context,
            AdvancedModelTask.RUL,
            100,
            AdvancedModelRouteRole.COVERAGE,
        )
    ]
    assert evidence_resolver.calls == [
        ("matr-three-batch-final-v1", AdvancedModelTask.RUL, 100)
    ]


@pytest.mark.parametrize(
    ("runtime_update", "identity_update"),
    [
        ({"project_id": "another-project"}, {}),
        ({"task": AdvancedModelTask.SOH}, {}),
        ({"dataset_id": "OTHER"}, {}),
        ({"data_version": "changed"}, {}),
        ({"split_version": "changed"}, {}),
        ({}, {"registration_id": "another-registration"}),
        ({}, {"data_version": "changed"}),
        ({}, {"split_version": "changed"}),
    ],
)
def test_rejects_runtime_or_source_identity_drift(
    runtime_update: dict[str, object],
    identity_update: dict[str, object],
) -> None:
    request = _request()
    context = cast(
        VerifiedProjectInvocationContext,
        SimpleNamespace(project_id=request.project_id),
    )
    runtime = _Runtime(**runtime_update)
    evidence = _evidence(identity_update=identity_update)
    resolver = ActiveAdvancedCalibrationPreparationResolver(
        runtime_resolver=cast(Any, _RuntimeResolver(runtime)),
        evidence_resolver=cast(Any, _EvidenceResolver(evidence)),
    )

    with pytest.raises(
        ValueError,
        match="does not match the materialization request",
    ):
        resolver.resolve(context, request)


def test_import_does_not_eagerly_load_torch_or_pyarrow() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import quanxin_life.application.advanced_calibration_preparation; "
                "assert 'torch' not in sys.modules; "
                "assert 'pyarrow' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _request() -> AdvancedCalibrationMaterializationRequest:
    return AdvancedCalibrationMaterializationRequest(
        project_id="project-1",
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        route_role=AdvancedModelRouteRole.COVERAGE,
        source_registration_id="matr-three-batch-final-v1",
    )


def _evidence(
    *,
    identity_update: dict[str, object] | None = None,
) -> AdvancedRULCalibrationEvidence:
    identity = {
        "registration_id": "matr-three-batch-final-v1",
        "dataset_id": "MATR",
        "data_version": "matr-data-v1",
        "split_version": "matr-split-v1",
        "three_batch_manifest_sha256": "5" * 64,
        "combined_split_sha256": "6" * 64,
        "conversion_report_sha256s": ("7" * 64,) * 3,
        "component_split_sha256s": ("8" * 64,) * 3,
        "eligibility_report_sha256s": ("9" * 64,) * 3,
        "supervision_report_sha256s": ("a" * 64,) * 3,
        "supervision_parquet_sha256s": ("b" * 64,) * 3,
        "source_identity_sha256": "4" * 64,
    }
    identity.update(identity_update or {})
    return AdvancedRULCalibrationEvidence(
        source_identity=AdvancedCalibrationSourceIdentity.model_validate(
            identity
        ),
        cells=(
            AdvancedRULObservedCell(
                cell_id="calibration-cell-1",
                observed_cycle=700,
            ),
        ),
    )
