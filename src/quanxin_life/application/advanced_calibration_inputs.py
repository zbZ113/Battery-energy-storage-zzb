"""Resolve registered MATR cells into verified label-free model inputs."""

from __future__ import annotations

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationCellSourceResolver,
    AdvancedCalibrationSourceIdentity,
)
from quanxin_life.application.advanced_calibration_materialization import (
    AdvancedCalibrationCellInput,
)
from quanxin_life.core import AdvancedModelTask
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig
from quanxin_life.features.verified_matr_sequence import (
    load_verified_matr_early_sequence,
)


class RegisteredMatrAdvancedCalibrationCellInputResolver:
    """Build one runtime input only from a freshly verified source cell."""

    def __init__(
        self,
        *,
        source_resolver: AdvancedCalibrationCellSourceResolver,
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
        normalized_task = AdvancedModelTask(task)
        source = self._source_resolver.resolve_cell_source(
            source_registration_id,
            source_identity=source_identity,
            task=normalized_task,
            cutoff_cycle=cutoff_cycle,
            cell_id=cell_id,
        )
        if (
            source.source_identity != source_identity
            or source.cell_evidence.cell_id != cell_id
            or (
                normalized_task is AdvancedModelTask.RUL
                and source.initial_soh is not None
            )
            or (
                normalized_task is AdvancedModelTask.SOH
                and source.initial_soh is None
            )
        ):
            raise ValueError(
                "Advanced calibration cell source does not match the request"
            )
        loaded = load_verified_matr_early_sequence(
            processed_root=source.processed_root,
            evidence=source.cell_evidence,
            reference_capacity_ah=(
                source.cell_evidence.reference_capacity_ah
            ),
            raw_sha256=source.raw_sha256,
            config=MultichannelCycleConfig(
                cutoff_cycle=cutoff_cycle,
                feature_version=feature_version,
            ),
            data_version=source_identity.data_version,
        )
        return AdvancedCalibrationCellInput(
            source_registration_id=source_registration_id,
            source_identity_sha256=(
                source_identity.source_identity_sha256
            ),
            cell_id=cell_id,
            cutoff_cycle=cutoff_cycle,
            data_version=source_identity.data_version,
            feature_version=feature_version,
            split_version=source_identity.split_version,
            raw_sequence=loaded.sequence,
            initial_soh=source.initial_soh,
        )


__all__ = [
    "RegisteredMatrAdvancedCalibrationCellInputResolver",
]
