"""Label-free CellMetadata projection shared by audited tool consumers."""

from __future__ import annotations

from collections.abc import Mapping

from quanxin_life.core import CellMetadata

AUDITED_CELL_METADATA_FIELDS = frozenset(
    {
        "adapter_version",
        "cell_id",
        "chemistry",
        "dataset_id",
        "eol_threshold",
        "nominal_capacity_ah",
        "protocol_description",
        "protocol_id",
        "raw_cell_id",
        "reference_capacity_ah",
        "schema_version",
        "source_sha256",
        "source_uri",
    }
)


class CellMetadataEvidenceError(ValueError):
    """Raised when an audited metadata projection is malformed."""


class CellMetadataEvidenceIdentityError(CellMetadataEvidenceError):
    """Raised when metadata does not identify the result's dataset and cell."""


def cell_metadata_evidence_payload(
    metadata: CellMetadata | None,
) -> dict[str, object] | None:
    """Return an exact display projection without labels or ingest parameters."""

    if metadata is None:
        return None
    checked = CellMetadata.model_validate(metadata.model_dump(mode="json"))
    payload = checked.model_dump(
        mode="json",
        include=set(AUDITED_CELL_METADATA_FIELDS),
    )
    validate_cell_metadata_evidence_payload(
        payload,
        expected_dataset_id=checked.dataset_id,
        expected_cell_id=checked.cell_id,
    )
    return payload


def validate_cell_metadata_evidence_payload(
    value: object,
    *,
    expected_dataset_id: str,
    expected_cell_id: str,
) -> CellMetadata:
    """Validate an exact label-free projection and its evidence identity."""

    if not isinstance(value, Mapping):
        raise CellMetadataEvidenceError("cell metadata evidence must be an object")
    if set(value) != AUDITED_CELL_METADATA_FIELDS:
        raise CellMetadataEvidenceError(
            "cell metadata evidence fields do not match the contract"
        )
    try:
        metadata = CellMetadata.model_validate(dict(value))
    except (TypeError, ValueError) as exc:
        raise CellMetadataEvidenceError(
            "cell metadata evidence does not satisfy CellMetadata"
        ) from exc
    if (
        metadata.dataset_id != expected_dataset_id
        or metadata.cell_id != expected_cell_id
    ):
        raise CellMetadataEvidenceIdentityError(
            "cell metadata evidence identity does not match the result"
        )
    if (
        metadata.official_life_label is not None
        or metadata.official_life_label_name is not None
        or metadata.ingestion_parameters
    ):
        raise CellMetadataEvidenceError(
            "cell metadata evidence contains training-only fields"
        )
    return metadata


def validate_versioned_cell_metadata_evidence(
    artifact: object,
    *,
    artifact_type: object,
    legacy_artifact_type: str,
    metadata_artifact_type: str,
) -> CellMetadata | None:
    """Enforce that v1 omits metadata and v2 carries the strict projection."""

    if not isinstance(artifact, Mapping):
        raise CellMetadataEvidenceError("analysis artifact must be an object")
    if artifact_type == legacy_artifact_type:
        if "cell_metadata" in artifact:
            raise CellMetadataEvidenceError(
                "legacy analysis artifact cannot carry cell metadata"
            )
        return None
    if artifact_type != metadata_artifact_type:
        raise CellMetadataEvidenceError(
            "analysis artifact has an unsupported metadata version"
        )
    if "cell_metadata" not in artifact:
        raise CellMetadataEvidenceError(
            "metadata-bearing analysis artifact is missing cell metadata"
        )
    dataset_id = artifact.get("dataset_id")
    cell_id = artifact.get("cell_id")
    if not isinstance(dataset_id, str) or not isinstance(cell_id, str):
        raise CellMetadataEvidenceIdentityError(
            "analysis artifact metadata identity is invalid"
        )
    return validate_cell_metadata_evidence_payload(
        artifact.get("cell_metadata"),
        expected_dataset_id=dataset_id,
        expected_cell_id=cell_id,
    )


__all__ = [
    "AUDITED_CELL_METADATA_FIELDS",
    "CellMetadataEvidenceError",
    "CellMetadataEvidenceIdentityError",
    "cell_metadata_evidence_payload",
    "validate_cell_metadata_evidence_payload",
    "validate_versioned_cell_metadata_evidence",
]
