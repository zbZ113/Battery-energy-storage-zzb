from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quanxin_life.application.aily_mcp_authorization import (
    ProjectBoundAilyMcpCallerAuthorizer,
    load_aily_mcp_identity_bindings,
)
from quanxin_life.integrations.feishu.jobs import FeishuAnalysisJobOrigin
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask


class _JobStore:
    def __init__(self) -> None:
        self.root = SimpleNamespace(
            job_id="root-run",
            source_job_id=None,
            job_origin=FeishuAnalysisJobOrigin.FEISHU,
            event_type="im.message.receive_v1",
            task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            chat_id="oc-project",
            sender_id="ou-uploader",
        )
        self.child = SimpleNamespace(
            job_id="analysis-run",
            source_job_id="root-run",
            job_origin=FeishuAnalysisJobOrigin.AILY,
            event_type="aily.analysis_task.create_v1",
            task_type=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
            chat_id="oc-project",
            sender_id="ou-uploader",
        )

    def get(self, job_id: str) -> SimpleNamespace:
        if job_id == self.root.job_id:
            return self.root
        if job_id == self.child.job_id:
            return self.child
        raise ValueError("unknown job")


class _ContextService:
    def __init__(self, actor_user_id: str) -> None:
        self.actor_user_id = actor_user_id
        self.resolve_calls: list[tuple[str, str]] = []
        self.revalidate_calls = 0

    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> SimpleNamespace:
        self.resolve_calls.append((chat_id, sender_open_id))
        return SimpleNamespace(actor_user_id=self.actor_user_id)

    def revalidate(self, context: SimpleNamespace) -> SimpleNamespace:
        self.revalidate_calls += 1
        return context


def _bindings_file(tmp_path: Path, *, duplicate: bool = False) -> Path:
    path = tmp_path / "aily-mcp-identities.json"
    bindings = [
        {"aily_user_id": "aily-user-1", "local_user_id": "local-user-1"}
    ]
    if duplicate:
        bindings.append(
            {"aily_user_id": "aily-user-1", "local_user_id": "local-user-2"}
        )
    path.write_text(
        __import__("json").dumps(
            {
                "schema_version": "quanxin-aily-mcp-identity-bindings-v1",
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_project_bound_authorizer_maps_aily_user_to_the_root_upload_actor(
    tmp_path: Path,
) -> None:
    bindings = load_aily_mcp_identity_bindings(_bindings_file(tmp_path))
    contexts = _ContextService(actor_user_id="local-user-1")
    authorizer = ProjectBoundAilyMcpCallerAuthorizer(
        job_store=_JobStore(),
        context_service=contexts,
        identity_bindings=bindings,
    )

    authorizer.authorize_identity(aily_user_id="aily-user-1")

    authorizer.authorize_run_reference(
        aily_user_id="aily-user-1",
        run_id="analysis-run",
    )
    authorizer.authorize_run_reference(
        aily_user_id="aily-user-1",
        run_id="root-run",
    )

    assert contexts.resolve_calls == [
        ("oc-project", "ou-uploader"),
        ("oc-project", "ou-uploader"),
    ]
    assert contexts.revalidate_calls == 2


def test_project_bound_authorizer_rejects_unknown_or_cross_user_references(
    tmp_path: Path,
) -> None:
    bindings = load_aily_mcp_identity_bindings(_bindings_file(tmp_path))
    contexts = _ContextService(actor_user_id="another-local-user")
    authorizer = ProjectBoundAilyMcpCallerAuthorizer(
        job_store=_JobStore(),
        context_service=contexts,
        identity_bindings=bindings,
    )

    for aily_user_id in ("unknown-user", "aily-user-1"):
        with pytest.raises(ValueError, match="not authorized"):
            authorizer.authorize_run_reference(
                aily_user_id=aily_user_id,
                run_id="analysis-run",
            )


def test_identity_bindings_reject_duplicate_aily_users(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        load_aily_mcp_identity_bindings(_bindings_file(tmp_path, duplicate=True))
