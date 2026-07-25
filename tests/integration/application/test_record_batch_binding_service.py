from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.ingestion import (
    CANONICAL_CYCLE_CSV_FIELDS,
    CanonicalCsvBatchRegistration,
    FileSystemVerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    ProjectInvocationNotFoundError,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.record_batch_bindings import (
    RecordBatchBindingNotFoundError,
    RecordBatchBindingService,
    RecordBatchBindingStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    CellMetadata,
    DatasetStatus,
    ProjectStatus,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    UserRole,
    UserStatus,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.persistence import (
    Base,
    create_engine_from_config,
    create_session_factory,
)
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    Dataset,
    Project,
    RecordBatchBinding,
    SessionRecord,
    User,
)

NOW = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)


def _principal(name: str) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        username=f"{name}@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _payload() -> bytes:
    header = ",".join(CANONICAL_CYCLE_CSV_FIELDS)
    rows = (
        "UPLOAD,cell-1,1,0,0,3.1,-1,25,1.1,1.0,0.01,true,true",
        "UPLOAD,cell-1,1,1,1,3.2,-1,25,1.1,1.0,0.01,true,true",
        "UPLOAD,cell-1,20,0,0,3.1,-1,25,1.0,0.9,0.02,true,true",
        "UPLOAD,cell-1,20,1,1,3.2,-1,25,1.0,0.9,0.02,true,true",
    )
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _registration(payload: bytes) -> CanonicalCsvBatchRegistration:
    digest = hashlib.sha256(payload).hexdigest()
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="cell-1",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="upload://canonical/cell-1.csv",
            source_sha256=digest,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="upload-data-v1",
        split_version="upload-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="canonical-upload-cell-1",
                source_kind=SourceKind.OBSERVED,
                uri="upload://canonical/cell-1.csv",
                sha256=digest,
                description="Observed canonical CSV fixture",
                created_at=NOW,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Context:
    service: RecordBatchBindingService
    context_service: ProjectInvocationContextService
    dataset_service: DatasetService
    session_factory: SessionFactory
    store_root: Path
    principals: dict[str, AuthPrincipal]
    project_ids: dict[str, str]
    dataset_ids: dict[str, str]


@pytest.fixture
def context(tmp_path: Path) -> _Context:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'bindings.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals = {
        "owner_a": _principal("owner-a"),
        "owner_b": _principal("owner-b"),
        "outsider": _principal("outsider"),
    }
    with session_factory.begin() as session:
        for principal in principals.values():
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only-not-used-for-authentication",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            session.add(
                SessionRecord(
                    id=principal.session_id,
                    user_id=principal.user_id,
                    token_hash=hashlib.sha256(principal.session_id.encode()).hexdigest(),
                    status=SessionStatus.ACTIVE.value,
                    created_at=NOW,
                    expires_at=NOW + timedelta(hours=12),
                    revoked_at=None,
                )
            )

    project_service = ProjectService(session_factory)
    dataset_service = DatasetService(session_factory)
    project_ids: dict[str, str] = {}
    dataset_ids: dict[str, str] = {}
    for owner_name in ("owner_a", "owner_b"):
        principal = principals[owner_name]
        project = project_service.create_project(
            principal,
            name=f"{owner_name} project",
            now=NOW,
        )
        dataset = dataset_service.create_dataset(
            principal,
            project_id=project.project_id,
            name=f"{owner_name} canonical upload",
            data_version="upload-data-v1",
            schema_version="canonical-cycle-v1",
            now=NOW,
        )
        project_ids[owner_name] = project.project_id
        dataset_ids[owner_name] = dataset.dataset_id

    store_root = tmp_path / "verified-batches"
    context_service = ProjectInvocationContextService(
        session_factory,
        clock=lambda: NOW,
    )
    service = RecordBatchBindingService(
        session_factory,
        FileSystemVerifiedEarlyCycleBatchStore(store_root),
        context_validator=context_service,
    )
    return _Context(
        service=service,
        context_service=context_service,
        dataset_service=dataset_service,
        session_factory=session_factory,
        store_root=store_root,
        principals=principals,
        project_ids=project_ids,
        dataset_ids=dataset_ids,
    )


def _register(context: _Context, owner_name: str = "owner_a"):
    payload = _payload()
    return context.service.register_canonical_csv(
        context.principals[owner_name],
        context.dataset_ids[owner_name],
        payload,
        _registration(payload),
        now=NOW,
    )


def _freeze(context: _Context, owner_name: str = "owner_a") -> None:
    context.dataset_service.freeze_dataset(
        context.principals[owner_name],
        context.dataset_ids[owner_name],
        now=NOW,
    )


def _invocation_context(context: _Context, owner_name: str = "owner_a"):
    return context.context_service.resolve_http(
        context.principals[owner_name],
        context.project_ids[owner_name],
    )


def _content_batch_id(context: _Context, record_batch_id: str) -> str:
    with context.session_factory() as session:
        binding = session.get(RecordBatchBinding, record_batch_id)
        assert binding is not None
        return binding.content_batch_id


def test_same_content_is_project_bound_and_registration_retry_is_idempotent(
    context: _Context,
) -> None:
    first = _register(context, "owner_a")
    repeated = _register(context, "owner_a")
    other_project = _register(context, "owner_b")
    first_content_batch_id = _content_batch_id(context, first.record_batch_id)
    other_content_batch_id = _content_batch_id(
        context,
        other_project.record_batch_id,
    )

    assert first == repeated
    assert UUID(first.record_batch_id).version == 4
    assert first.binding_schema_version == "record-batch-binding-v1"
    assert first.project_id == context.project_ids["owner_a"]
    assert first.dataset_id == context.dataset_ids["owner_a"]
    assert first.content_dataset_id == "UPLOAD"
    assert first.cell_id == "cell-1"
    assert first.cutoff_cycle == 20
    assert first.data_version == "upload-data-v1"
    assert first.split_version == "upload-split-v1"
    assert first.dataset_schema_version == "canonical-cycle-v1"
    assert first_content_batch_id == other_content_batch_id
    assert first.record_batch_id != first_content_batch_id
    assert first.record_batch_id != other_project.record_batch_id
    assert first.dataset_id != other_project.dataset_id
    assert "content_batch_id" not in first.model_dump(mode="json")


def test_upload_hides_invisible_or_archived_project_and_rejects_frozen_dataset(
    context: _Context,
) -> None:
    payload = _payload()
    registration = _registration(payload)

    with pytest.raises(RecordBatchBindingNotFoundError):
        context.service.register_canonical_csv(
            context.principals["outsider"],
            context.dataset_ids["owner_a"],
            payload,
            registration,
            now=NOW,
        )

    _freeze(context)
    with pytest.raises(RecordBatchBindingStateError, match="DRAFT"):
        context.service.register_canonical_csv(
            context.principals["owner_a"],
            context.dataset_ids["owner_a"],
            payload,
            registration,
            now=NOW,
        )

    with context.session_factory.begin() as session:
        project = session.get(Project, context.project_ids["owner_b"])
        assert project is not None
        project.status = ProjectStatus.ARCHIVED.value
    with pytest.raises(RecordBatchBindingNotFoundError):
        context.service.register_canonical_csv(
            context.principals["owner_b"],
            context.dataset_ids["owner_b"],
            payload,
            registration,
            now=NOW,
        )


def test_resolution_requires_frozen_dataset_and_returns_binding_identifier(
    context: _Context,
) -> None:
    registered = _register(context)
    invocation = _invocation_context(context)

    with pytest.raises(RecordBatchBindingStateError, match="FROZEN"):
        context.service.resolve_verified_early_cycle_batch(
            invocation,
            registered.record_batch_id,
        )

    _freeze(context)
    resolved = context.service.resolve_verified_early_cycle_batch(
        invocation,
        registered.record_batch_id,
    )

    assert resolved.record_batch_id == registered.record_batch_id
    assert resolved.metadata.dataset_id == "UPLOAD"
    assert resolved.metadata.cell_id == "cell-1"
    assert resolved.source_manifest_hash == registered.source_manifest_sha256


def test_resolution_hides_cross_project_binding_and_revalidates_live_context(
    context: _Context,
) -> None:
    registered = _register(context)
    _freeze(context)

    with pytest.raises(RecordBatchBindingNotFoundError):
        context.service.resolve_verified_early_cycle_batch(
            _invocation_context(context, "owner_b"),
            registered.record_batch_id,
        )

    invocation = _invocation_context(context)
    with context.session_factory.begin() as session:
        project = session.get(Project, context.project_ids["owner_a"])
        assert project is not None
        project.status = ProjectStatus.ARCHIVED.value
    with pytest.raises(ProjectInvocationNotFoundError):
        context.service.resolve_verified_early_cycle_batch(
            invocation,
            registered.record_batch_id,
        )


def test_resolution_fails_closed_when_underlying_content_store_is_tampered(
    context: _Context,
) -> None:
    registered = _register(context)
    _freeze(context)
    content_batch_id = _content_batch_id(context, registered.record_batch_id)
    csv_path = context.store_root / f"{content_batch_id}.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    with pytest.raises(RecordBatchBindingStateError, match="content"):
        context.service.resolve_verified_early_cycle_batch(
            _invocation_context(context),
            registered.record_batch_id,
        )


def test_same_content_with_changed_registration_is_not_idempotent(
    context: _Context,
) -> None:
    _register(context)
    payload = _payload()
    registration = _registration(payload)
    changed_provenance = registration.provenance[0].model_copy(
        update={"description": "changed provenance description"}
    )
    changed = registration.model_copy(update={"provenance": (changed_provenance,)})

    with pytest.raises(RecordBatchBindingStateError, match="registration"):
        context.service.register_canonical_csv(
            context.principals["owner_a"],
            context.dataset_ids["owner_a"],
            payload,
            changed,
            now=NOW,
        )


def test_shared_content_rejects_changed_registration_from_another_project(
    context: _Context,
) -> None:
    _register(context, "owner_a")
    payload = _payload()
    registration = _registration(payload)
    changed_provenance = registration.provenance[0].model_copy(
        update={"description": "different project provenance"}
    )
    changed = registration.model_copy(update={"provenance": (changed_provenance,)})

    with pytest.raises(RecordBatchBindingStateError, match="registration"):
        context.service.register_canonical_csv(
            context.principals["owner_b"],
            context.dataset_ids["owner_b"],
            payload,
            changed,
            now=NOW,
        )


def test_resolution_detects_registration_provenance_tampering(
    context: _Context,
) -> None:
    registered = _register(context)
    _freeze(context)
    content_batch_id = _content_batch_id(context, registered.record_batch_id)
    registration_path = context.store_root / f"{content_batch_id}.registration.json"
    registration_payload = json.loads(registration_path.read_text(encoding="utf-8"))
    registration_payload["provenance"][0]["description"] = "tampered provenance"
    registration_path.write_text(
        json.dumps(registration_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(RecordBatchBindingStateError, match="content snapshot"):
        context.service.resolve_verified_early_cycle_batch(
            _invocation_context(context),
            registered.record_batch_id,
        )


def test_resolution_fails_closed_when_dataset_snapshot_drifts(
    context: _Context,
) -> None:
    registered = _register(context)
    _freeze(context)
    with context.session_factory.begin() as session:
        dataset = session.get(Dataset, context.dataset_ids["owner_a"])
        assert dataset is not None
        assert dataset.status == DatasetStatus.FROZEN.value
        dataset.data_version = "tampered-data-version"

    with pytest.raises(RecordBatchBindingStateError, match="snapshot"):
        context.service.resolve_verified_early_cycle_batch(
            _invocation_context(context),
            registered.record_batch_id,
        )
