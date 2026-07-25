from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import quanxin_life.application.model_artifact_catalog as catalog_module
from quanxin_life.application.model_artifact_catalog import (
    AdvancedModelArtifactProvenance,
    AdvancedModelRouteProvenance,
    ClassicModelArtifactCatalogSource,
    ModelArtifactCatalogAccessError,
    ModelArtifactCatalogNotFoundError,
    ModelArtifactCatalogService,
    ModelArtifactCatalogSourceError,
    ModelArtifactCatalogStateError,
    VerifiedModelArtifactMetadata,
    VerifiedModelArtifactRegistration,
)
from quanxin_life.application.model_artifacts import (
    ArtifactFormat,
    ArtifactKind,
    ModelArtifactManifest,
    ModelArtifactRegistry,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    ModelArtifact,
    ModelManifest,
    Project,
    User,
    UserProjectRole,
)

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=UTC)
_ADVANCED_MANIFEST_SHA256 = "a" * 64


class _AdvancedBatchSource:
    def __init__(
        self,
        registrations: tuple[VerifiedModelArtifactRegistration, ...] | None = None,
    ) -> None:
        self.registrations = registrations or _advanced_registrations()
        self.resolve_all_calls = 0

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration:
        return next(item for item in self.registrations if item.artifact_id == artifact_id)

    def resolve_all(self) -> tuple[VerifiedModelArtifactRegistration, ...]:
        self.resolve_all_calls += 1
        return self.registrations


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


@pytest.fixture
def catalog_context(
    tmp_path: Path,
) -> tuple[
    ModelArtifactCatalogService,
    dict[str, AuthPrincipal],
    str,
    str,
    Path,
    SessionFactory,
]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "model.ubj"
    artifact_path.write_bytes(b"verified-xgboost-native-bytes")
    artifact_id = str(uuid4())
    manifest = ModelArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_UBJ,
        relative_path=artifact_path.name,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        size_bytes=artifact_path.stat().st_size,
        model_version="xgboost-v1",
        data_version="matr-three-batch-v1",
        feature_version="matr-early-features-v1",
        split_version="matr-three-batch-split-v1",
        schema_version="model-artifact-v1",
        dataset_id="MATR",
        cutoff_cycle=20,
        feature_names=("capacity_delta", "resistance_growth"),
        created_at=NOW,
    )
    registry = ModelArtifactRegistry(artifact_root)
    registry.register(manifest)

    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals = {
        "admin": _principal(str(uuid4()), UserRole.ADMIN),
        "member": _principal(str(uuid4()), UserRole.MEMBER),
        "judge": _principal(str(uuid4()), UserRole.JUDGE),
        "outsider": _principal(str(uuid4()), UserRole.MEMBER),
    }
    project_id = str(uuid4())
    with session_factory.begin() as session:
        for principal in principals.values():
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.add(
            Project(
                id=project_id,
                owner_user_id=principals["member"].user_id,
                name="verified artifacts",
                status="ACTIVE",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=principals["judge"].user_id,
                project_id=project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    service = ModelArtifactCatalogService(
        session_factory,
        source=ClassicModelArtifactCatalogSource(registry),
    )
    return (
        service,
        principals,
        project_id,
        artifact_id,
        artifact_path,
        session_factory,
    )


def test_admin_registration_is_verified_idempotent_and_queryable(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, _, session_factory = catalog_context

    first = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )
    second = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW.replace(hour=19),
    )

    assert second == first
    assert first.project_id == project_id
    assert first.artifact_kind == "xgboost"
    assert first.artifact_format == "xgboost-ubj"
    assert first.status == "VERIFIED"
    assert first.dataset_id == "MATR"
    assert first.cutoff_cycle == 20
    assert first.object_uri.startswith(f"verified-model-artifact://{artifact_id}/")
    assert first.manifest_uri == f"verified-model-artifact://{artifact_id}/manifest"
    assert service.get_artifact(principals["judge"], artifact_id) == first
    assert service.list_artifacts(
        principals["member"],
        project_id=project_id,
        artifact_kind="xgboost",
        cutoff_cycle=20,
    ) == (first,)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 1
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 1


def test_admin_registers_all_advanced_candidates_in_one_idempotent_batch(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    _, principals, project_id, _, _, session_factory = catalog_context
    source = _AdvancedBatchSource()
    service = ModelArtifactCatalogService(session_factory, source=source)

    first = service.register_advanced_candidates(
        principals["admin"],
        project_id=project_id,
        registered_at=NOW,
    )
    second = service.register_advanced_candidates(
        principals["admin"],
        project_id=project_id,
        registered_at=NOW.replace(hour=19),
    )

    assert second == first
    assert first.artifact_count == 15
    assert first.project_id == project_id
    assert first.deployment_bundle_manifest_sha256 == _ADVANCED_MANIFEST_SHA256
    assert first.lifecycle_status == "REGISTERED_CANDIDATE"
    assert first.activation_status == "NOT_ACTIVATED"
    assert tuple(item.artifact_id for item in first.artifacts) == tuple(
        sorted(item.artifact_id for item in source.registrations)
    )
    assert source.resolve_all_calls == 2
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 15
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 15


def test_advanced_candidates_cannot_bypass_atomic_batch_registration(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    _, principals, project_id, _, _, session_factory = catalog_context
    source = _AdvancedBatchSource()
    service = ModelArtifactCatalogService(session_factory, source=source)

    with pytest.raises(ModelArtifactCatalogStateError, match=r"batch|Advanced"):
        service.register_artifact(
            principals["admin"],
            project_id=project_id,
            artifact_id=source.registrations[0].artifact_id,
            registered_at=NOW,
        )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 0


def test_identical_concurrent_advanced_batch_recovers_idempotently(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, principals, project_id, _, _, session_factory = catalog_context
    source = _AdvancedBatchSource()
    service = ModelArtifactCatalogService(session_factory, source=source)
    expected = service.register_advanced_candidates(
        principals["admin"],
        project_id=project_id,
        registered_at=NOW,
    )
    original_scope = catalog_module.session_scope
    calls = 0

    @contextmanager
    def conflict_once(factory: SessionFactory) -> Iterator[Session]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IntegrityError("INSERT", {}, RuntimeError("unique constraint"))
        with original_scope(factory) as session:
            yield session

    monkeypatch.setattr(catalog_module, "session_scope", conflict_once)

    recovered = service.register_advanced_candidates(
        principals["admin"],
        project_id=project_id,
        registered_at=NOW.replace(hour=19),
    )

    assert recovered == expected
    assert calls == 2


def test_advanced_candidate_batch_rolls_back_on_last_digest_conflict(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    _, principals, project_id, _, _, session_factory = catalog_context
    source = _AdvancedBatchSource()
    conflicting = source.registrations[-1]
    existing_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            ModelArtifact(
                id=existing_id,
                project_id=project_id,
                artifact_format="safetensors-bundle",
                object_uri=f"verified-model-artifact://{existing_id}/bundle",
                sha256=conflicting.artifact_sha256,
                status="VERIFIED",
                created_at=NOW,
            )
        )
    service = ModelArtifactCatalogService(session_factory, source=source)

    with pytest.raises(ModelArtifactCatalogStateError, match="digest"):
        service.register_advanced_candidates(
            principals["admin"],
            project_id=project_id,
            registered_at=NOW,
        )

    with session_factory() as session:
        persisted_ids = set(session.scalars(select(ModelArtifact.id)))
        assert persisted_ids == {existing_id}
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 0


def test_advanced_candidate_batch_rejects_incomplete_source_role_and_naive_time(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    _, principals, project_id, _, _, session_factory = catalog_context
    incomplete = _AdvancedBatchSource(_advanced_registrations()[:-1])
    service = ModelArtifactCatalogService(session_factory, source=incomplete)

    with pytest.raises(ModelArtifactCatalogSourceError, match="15"):
        service.register_advanced_candidates(
            principals["admin"],
            project_id=project_id,
            registered_at=NOW,
        )
    with pytest.raises(ModelArtifactCatalogAccessError):
        service.register_advanced_candidates(
            principals["member"],
            project_id=project_id,
            registered_at=NOW,
        )
    with pytest.raises(ValueError, match="timezone"):
        service.register_advanced_candidates(
            principals["admin"],
            project_id=project_id,
            registered_at=datetime(2026, 7, 18, 18, 0),
        )
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0
        assert session.scalar(select(func.count()).select_from(ModelManifest)) == 0


def test_registration_requires_admin_and_queries_hide_invisible_projects(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, _, _ = catalog_context

    with pytest.raises(ModelArtifactCatalogAccessError):
        service.register_artifact(
            principals["member"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )
    registered = service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )

    assert service.list_artifacts(principals["outsider"]) == ()
    with pytest.raises(ModelArtifactCatalogNotFoundError):
        service.get_artifact(principals["outsider"], registered.artifact_id)


def _advanced_registrations() -> tuple[VerifiedModelArtifactRegistration, ...]:
    coordinates = (
        ("RUL", 20, "DEFAULT", "cyclepatch_direct"),
        ("RUL", 50, "POINT_ACCURACY", "cyclepatch_direct"),
        ("RUL", 50, "COVERAGE", "cyclepatch_batlinet"),
        ("RUL", 100, "POINT_ACCURACY", "cyclepatch_batlinet"),
        ("RUL", 100, "COVERAGE", "cyclepatch_direct"),
        ("RUL", 150, "POINT_ACCURACY", "cyclepatch_direct"),
        ("RUL", 150, "COVERAGE", "cyclepatch_batlinet"),
        *(("SOH", cutoff, "MEAN_ACCURACY", "hybridpatch_v2") for cutoff in (20, 50, 100, 150)),
        *(("SOH", cutoff, "TAIL_EFFICIENCY", "current_hybrid") for cutoff in (20, 50, 100, 150)),
    )
    registrations = []
    for ordinal, (task, cutoff, role, family) in enumerate(coordinates, start=1):
        artifact_id = str(uuid5(NAMESPACE_URL, f"advanced:{task}:{cutoff}:{role}:{family}"))
        candidate_id = f"candidate-{family}"
        route = AdvancedModelRouteProvenance(
            task=task,
            role=role,
            family=family,
            candidate_id=candidate_id,
            cutoff_cycle=cutoff,
            seed=38,
            best_epoch=ordinal,
            run_id=f"matr-{family}-{candidate_id}-c{cutoff}-s38",
            checkpoint_manifest_sha256=_digest(f"checkpoint-manifest-{ordinal}"),
            checkpoint_model_sha256=_digest(f"checkpoint-model-{ordinal}"),
            checkpoint_context_sha256=_digest(f"checkpoint-context-{ordinal}"),
        )
        registrations.append(
            VerifiedModelArtifactRegistration(
                artifact_id=artifact_id,
                artifact_format="safetensors-bundle",
                object_uri=f"verified-model-artifact://{artifact_id}/bundle",
                artifact_sha256=_digest(f"artifact-{ordinal}"),
                model_version=route.run_id,
                manifest_uri=f"verified-model-artifact://{artifact_id}/manifest",
                manifest_sha256=_digest(f"manifest-{ordinal}"),
                metadata=VerifiedModelArtifactMetadata(
                    artifact_kind=f"advanced-{family}",
                    dataset_id="MATR",
                    data_version="matr-three-batch-v1",
                    feature_version="cyclepatch-multichannel-v1",
                    split_version="matr-three-batch-cell-split-v1",
                    schema_version="deep-model-artifact-v1",
                    cutoff_cycle=cutoff,
                    feature_names=("cycle_index", "capacity"),
                    advanced_provenance=AdvancedModelArtifactProvenance(
                        deployment_bundle_manifest_sha256=_ADVANCED_MANIFEST_SHA256,
                        source_commit="2" * 40,
                        final_output_sha256="b" * 64,
                        final_config_sha256="c" * 64,
                        promotion_manifest_sha256="d" * 64,
                        promotion_decisions_sha256="e" * 64,
                        promotion_source_evidence_sha256="f" * 64,
                        selection_manifest_sha256="1" * 64,
                        training_input_bundle_sha256="3" * 64,
                        local_reconstructed_input_bundle_sha256="4" * 64,
                        input_bundle_hashes_match=False,
                        candidate_config_sha256=_digest(f"candidate-{family}"),
                        normalization_sha256=_digest(f"normalization-{family}-{cutoff}"),
                        target_scaler_context_sha256=_digest(f"target-scaler-{ordinal}")
                        if task == "RUL"
                        else None,
                        reference_library_sha256=_digest("reference-library")
                        if family == "cyclepatch_batlinet"
                        else None,
                        routes=(route,),
                    ),
                ),
            )
        )
    return tuple(sorted(registrations, key=lambda item: item.artifact_id))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_registration_reverifies_bytes_and_persisted_context(
    catalog_context: tuple[
        ModelArtifactCatalogService,
        dict[str, AuthPrincipal],
        str,
        str,
        Path,
        SessionFactory,
    ],
) -> None:
    service, principals, project_id, artifact_id, artifact_path, session_factory = (
        catalog_context
    )
    service.register_artifact(
        principals["admin"],
        project_id=project_id,
        artifact_id=artifact_id,
        registered_at=NOW,
    )
    artifact_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        service.register_artifact(
            principals["admin"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )

    artifact_path.write_bytes(b"verified-xgboost-native-bytes")
    with session_factory.begin() as session:
        row = session.get(ModelManifest, session.scalar(select(ModelManifest.id)))
        assert row is not None
        row.model_version = "tampered-version"

    with pytest.raises(ModelArtifactCatalogStateError, match="persisted"):
        service.register_artifact(
            principals["admin"],
            project_id=project_id,
            artifact_id=artifact_id,
            registered_at=NOW,
        )
