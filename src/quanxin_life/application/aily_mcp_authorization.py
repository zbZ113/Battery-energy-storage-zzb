"""Project-bound caller authorization for the opt-in Aily MCP transport."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask

_IDENTITY_SCHEMA_VERSION = "quanxin-aily-mcp-identity-bindings-v1"
_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}\Z")
_MAX_BINDINGS_FILE_BYTES = 64 * 1024


class _IdentityBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aily_user_id: str = Field(min_length=1, max_length=200)
    local_user_id: str = Field(min_length=1, max_length=200)

    @field_validator("aily_user_id", "local_user_id")
    @classmethod
    def references_are_safe(cls, value: str) -> str:
        normalized = value.strip()
        if _SAFE_REFERENCE.fullmatch(normalized) is None:
            raise ValueError("identity binding references must be safe")
        return normalized


class _IdentityBindingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    bindings: tuple[_IdentityBinding, ...] = Field(min_length=1, max_length=10_000)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        if value != _IDENTITY_SCHEMA_VERSION:
            raise ValueError("Aily MCP identity binding schema is unsupported")
        return value

    @model_validator(mode="after")
    def identities_are_one_to_one(self) -> _IdentityBindingDocument:
        aily_user_ids = tuple(item.aily_user_id for item in self.bindings)
        local_user_ids = tuple(item.local_user_id for item in self.bindings)
        if len(set(aily_user_ids)) != len(aily_user_ids) or len(
            set(local_user_ids)
        ) != len(local_user_ids):
            raise ValueError("Aily MCP identity bindings must be one-to-one and unique")
        return self


@dataclass(frozen=True, slots=True)
class AilyMcpIdentityBindings:
    """Operator-reviewed one-to-one Aily caller to local actor mapping."""

    _local_user_by_aily_user: Mapping[str, str]

    def resolve_local_user_id(self, aily_user_id: str) -> str:
        normalized = aily_user_id.strip() if isinstance(aily_user_id, str) else ""
        if _SAFE_REFERENCE.fullmatch(normalized) is None:
            raise ValueError("Aily MCP caller is not authorized")
        try:
            return self._local_user_by_aily_user[normalized]
        except KeyError as exc:
            raise ValueError("Aily MCP caller is not authorized") from exc


class AilyMcpJobStore(Protocol):
    def get(self, job_id: str) -> FeishuAnalysisJobRecord: ...


class AilyMcpProjectContextService(Protocol):
    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> Any: ...

    def revalidate(self, context: Any) -> Any: ...


class ProjectBoundAilyMcpCallerAuthorizer:
    """Revalidate the original Feishu upload and require the mapped local actor."""

    def __init__(
        self,
        *,
        job_store: AilyMcpJobStore,
        context_service: AilyMcpProjectContextService,
        identity_bindings: AilyMcpIdentityBindings,
    ) -> None:
        self._job_store = job_store
        self._context_service = context_service
        self._identity_bindings = identity_bindings

    def authorize_identity(self, *, aily_user_id: str) -> None:
        self._identity_bindings.resolve_local_user_id(aily_user_id)

    def authorize_run_reference(
        self,
        *,
        aily_user_id: str,
        run_id: str,
    ) -> None:
        try:
            expected_local_user_id = self._identity_bindings.resolve_local_user_id(
                aily_user_id
            )
            referenced = self._job_store.get(run_id)
            root = (
                referenced
                if referenced.source_job_id is None
                else self._job_store.get(referenced.source_job_id)
            )
            if not _is_root_feishu_upload(root):
                raise ValueError("invalid root upload")
            if referenced.source_job_id is not None and (
                referenced.job_origin
                not in {
                    FeishuAnalysisJobOrigin.AILY,
                    FeishuAnalysisJobOrigin.FEISHU,
                }
                or referenced.chat_id != root.chat_id
                or referenced.sender_id != root.sender_id
            ):
                raise ValueError("derived run identity changed")
            assert root.chat_id is not None
            assert root.sender_id is not None
            context = self._context_service.resolve_feishu(
                chat_id=root.chat_id,
                sender_open_id=root.sender_id,
            )
            verified = self._context_service.revalidate(context)
            actor_user_id = getattr(verified, "actor_user_id", None)
            if not isinstance(actor_user_id, str) or not hmac.compare_digest(
                actor_user_id,
                expected_local_user_id,
            ):
                raise ValueError("caller does not own the source upload")
        except (AttributeError, KeyError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Aily MCP caller is not authorized") from exc


def load_aily_mcp_identity_bindings(path: Path) -> AilyMcpIdentityBindings:
    """Load one versioned operator file without persisting Aily email addresses."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Aily MCP identity bindings path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Aily MCP identity bindings must be a regular file")
    if path.stat().st_size > _MAX_BINDINGS_FILE_BYTES:
        raise ValueError("Aily MCP identity bindings file is too large")
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        document = _IdentityBindingDocument.model_validate(payload)
    except ValidationError as exc:
        if any("one-to-one and unique" in error["msg"] for error in exc.errors()):
            raise ValueError(
                "Aily MCP identity bindings must be one-to-one and unique"
            ) from exc
        raise ValueError("Aily MCP identity bindings are invalid") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Aily MCP identity bindings are invalid") from exc
    mapping = MappingProxyType(
        {item.aily_user_id: item.local_user_id for item in document.bindings}
    )
    return AilyMcpIdentityBindings(_local_user_by_aily_user=mapping)


def _is_root_feishu_upload(job: object) -> bool:
    return bool(
        getattr(job, "source_job_id", None) is None
        and getattr(job, "job_origin", None) is FeishuAnalysisJobOrigin.FEISHU
        and getattr(job, "event_type", None) == "im.message.receive_v1"
        and getattr(job, "task_type", None) is FeishuAnalysisTask.PREDICT_CYCLE_LIFE
        and getattr(job, "chat_id", None)
        and getattr(job, "sender_id", None)
    )


__all__ = [
    "AilyMcpIdentityBindings",
    "ProjectBoundAilyMcpCallerAuthorizer",
    "load_aily_mcp_identity_bindings",
]
