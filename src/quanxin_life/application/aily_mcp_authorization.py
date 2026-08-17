"""Project-bound caller authorization for the opt-in Aily MCP transport."""

from __future__ import annotations

import hashlib
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

from quanxin_life.core import ToolResult
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)

_IDENTITY_SCHEMA_VERSION = "quanxin-aily-mcp-identity-bindings-v1"
_HASHED_IDENTITY_SCHEMA_VERSION = "quanxin-aily-mcp-identity-bindings-v2"
_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}\Z")
_SHA256_REFERENCE = re.compile(r"[0-9a-f]{64}\Z")
_TASK_LABEL = re.compile(
    r"(?P<cell_id>[A-Za-z0-9][A-Za-z0-9._:@-]{0,199})"
    r"\s*\|\s*(?i:cutoff)-(?P<cutoff_cycle>[1-9][0-9]{0,5})\Z"
)
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


class _HashedIdentityBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aily_user_id_sha256: str
    local_user_id: str = Field(min_length=1, max_length=200)

    @field_validator("aily_user_id_sha256")
    @classmethod
    def identity_hash_is_canonical(cls, value: str) -> str:
        if _SHA256_REFERENCE.fullmatch(value) is None:
            raise ValueError("Aily user identity SHA-256 must be canonical")
        return value

    @field_validator("local_user_id")
    @classmethod
    def local_reference_is_safe(cls, value: str) -> str:
        normalized = value.strip()
        if _SAFE_REFERENCE.fullmatch(normalized) is None:
            raise ValueError("identity binding references must be safe")
        return normalized


class _HashedIdentityBindingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    bindings: tuple[_HashedIdentityBinding, ...] = Field(
        min_length=1,
        max_length=10_000,
    )

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        if value != _HASHED_IDENTITY_SCHEMA_VERSION:
            raise ValueError("Aily MCP identity binding schema is unsupported")
        return value

    @model_validator(mode="after")
    def identities_are_one_to_one(self) -> _HashedIdentityBindingDocument:
        aily_user_hashes = tuple(
            item.aily_user_id_sha256 for item in self.bindings
        )
        local_user_ids = tuple(item.local_user_id for item in self.bindings)
        if len(set(aily_user_hashes)) != len(aily_user_hashes) or len(
            set(local_user_ids)
        ) != len(local_user_ids):
            raise ValueError("Aily MCP identity bindings must be one-to-one and unique")
        return self


@dataclass(frozen=True, slots=True)
class AilyMcpIdentityBindings:
    """Operator-reviewed one-to-one Aily caller to local actor mapping."""

    _local_user_by_aily_user: Mapping[str, str]
    _local_user_by_aily_user_sha256: Mapping[str, str]

    def resolve_local_user_id(self, aily_user_id: str) -> str:
        normalized = aily_user_id.strip() if isinstance(aily_user_id, str) else ""
        if _SAFE_REFERENCE.fullmatch(normalized) is None:
            raise ValueError("Aily MCP caller is not authorized")
        local_user_id = self._local_user_by_aily_user.get(normalized)
        if local_user_id is None:
            identity_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            local_user_id = self._local_user_by_aily_user_sha256.get(identity_hash)
        if local_user_id is None:
            raise ValueError("Aily MCP caller is not authorized")
        return local_user_id


class AilyMcpJobStore(Protocol):
    def get(self, job_id: str) -> FeishuAnalysisJobRecord: ...

    def list_succeeded_root_uploads(
        self,
        *,
        cell_reference: str,
        limit: int = 100,
    ) -> tuple[FeishuAnalysisJobRecord, ...]: ...


class AilyMcpResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class AilyMcpProjectContextService(Protocol):
    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> Any: ...

    def revalidate(self, context: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ResolvedAilyAnalysisSource:
    task_label: str
    source_run_id: str
    data_batch_id: str
    cell_id: str
    cutoff_cycle: int


class ProjectBoundAilyMcpCallerAuthorizer:
    """Revalidate the original Feishu upload and require the mapped local actor."""

    def __init__(
        self,
        *,
        job_store: AilyMcpJobStore,
        context_service: AilyMcpProjectContextService,
        identity_bindings: AilyMcpIdentityBindings,
        result_resolver: AilyMcpResultResolver | None = None,
    ) -> None:
        self._job_store = job_store
        self._context_service = context_service
        self._identity_bindings = identity_bindings
        self._result_resolver = result_resolver

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
            self._require_owned_root(
                root,
                expected_local_user_id=expected_local_user_id,
            )
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
        except (AttributeError, KeyError, LookupError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Aily MCP caller is not authorized") from exc

    def resolve_analysis_source(
        self,
        *,
        aily_user_id: str,
        task_label: str,
    ) -> ResolvedAilyAnalysisSource:
        try:
            expected_local_user_id = self._identity_bindings.resolve_local_user_id(
                aily_user_id
            )
            cell_id, requested_cutoff, normalized_label = _parse_task_label(task_label)
            if self._result_resolver is None:
                raise ValueError("analysis result resolver is unavailable")
            candidates = self._job_store.list_succeeded_root_uploads(
                cell_reference=cell_id,
            )
            for root in candidates:
                try:
                    self._require_owned_root(
                        root,
                        expected_local_user_id=expected_local_user_id,
                    )
                    source_run_id = _required_reference(
                        getattr(root, "run_id", None),
                        field_name="source run",
                    )
                    if source_run_id != getattr(root, "job_id", None):
                        raise ValueError("root run identity changed")
                    data_batch_id = _required_reference(
                        getattr(root, "record_batch_id", None),
                        field_name="data batch",
                    )
                    result_id = _required_reference(
                        getattr(root, "analysis_result_id", None),
                        field_name="analysis result",
                    )
                    result = self._result_resolver.resolve_registered_result(result_id)
                    checked_result = ToolResult.model_validate(
                        result.model_dump(mode="json")
                    )
                    if checked_result.result_id != result_id:
                        raise ValueError("analysis result identity changed")
                    result_cell_id, result_cutoff = _cycle_life_identity(
                        checked_result
                    )
                    if (
                        result_cell_id != cell_id
                        or result_cell_id != getattr(root, "cell_reference", None)
                        or result_cutoff != requested_cutoff
                    ):
                        raise ValueError("task label does not match audited result")
                    return ResolvedAilyAnalysisSource(
                        task_label=normalized_label,
                        source_run_id=source_run_id,
                        data_batch_id=data_batch_id,
                        cell_id=result_cell_id,
                        cutoff_cycle=result_cutoff,
                    )
                except (
                    AttributeError,
                    KeyError,
                    LookupError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    continue
            raise ValueError("owned analysis source was not found")
        except (
            AttributeError,
            KeyError,
            LookupError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError("Aily MCP caller is not authorized") from exc

    def _require_owned_root(
        self,
        root: object,
        *,
        expected_local_user_id: str,
    ) -> None:
        if not _is_root_feishu_upload(root):
            raise ValueError("invalid root upload")
        if getattr(root, "job_status", FeishuAnalysisJobStatus.SUCCEEDED) is not (
            FeishuAnalysisJobStatus.SUCCEEDED
        ):
            raise ValueError("root upload is not complete")
        chat_id = getattr(root, "chat_id", None)
        sender_id = getattr(root, "sender_id", None)
        assert isinstance(chat_id, str)
        assert isinstance(sender_id, str)
        context = self._context_service.resolve_feishu(
            chat_id=chat_id,
            sender_open_id=sender_id,
        )
        verified = self._context_service.revalidate(context)
        actor_user_id = getattr(verified, "actor_user_id", None)
        if not isinstance(actor_user_id, str) or not hmac.compare_digest(
            actor_user_id,
            expected_local_user_id,
        ):
            raise ValueError("caller does not own the source upload")


def _parse_task_label(value: object) -> tuple[str, int, str]:
    normalized = value.strip() if isinstance(value, str) else ""
    match = _TASK_LABEL.fullmatch(normalized)
    if match is None:
        raise ValueError("analysis task label is invalid")
    cell_id = match.group("cell_id")
    cutoff_cycle = int(match.group("cutoff_cycle"))
    return cell_id, cutoff_cycle, f"{cell_id} | cutoff-{cutoff_cycle}"


def _required_reference(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} reference is invalid")
    return normalized


def _cycle_life_identity(result: ToolResult) -> tuple[str, int]:
    if (
        result.tool_name != "predict_cycle_life"
        or result.tool_version != ADVANCED_RUL_PREDICTION_TOOL_VERSION
        or result.values.get("artifact_type")
        not in ADVANCED_RUL_PREDICTION_EVIDENCE_TYPES
    ):
        raise ValueError("analysis result is not a supported cycle-life result")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("cycle-life artifact is invalid")
    cell_id = _required_reference(
        artifact.get("cell_id"),
        field_name="cycle-life cell",
    )
    cutoff_cycle = artifact.get("cutoff_cycle")
    if (
        isinstance(cutoff_cycle, bool)
        or not isinstance(cutoff_cycle, int)
        or cutoff_cycle < 1
    ):
        raise ValueError("cycle-life cutoff is invalid")
    return cell_id, cutoff_cycle


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
        if not isinstance(payload, Mapping):
            raise ValueError("identity binding document must be an object")
        schema_version = payload.get("schema_version")
        if schema_version == _IDENTITY_SCHEMA_VERSION:
            document = _IdentityBindingDocument.model_validate(payload)
            raw_mapping = {
                item.aily_user_id: item.local_user_id for item in document.bindings
            }
            hashed_mapping: dict[str, str] = {}
        elif schema_version == _HASHED_IDENTITY_SCHEMA_VERSION:
            hashed_document = _HashedIdentityBindingDocument.model_validate(payload)
            raw_mapping = {}
            hashed_mapping = {
                item.aily_user_id_sha256: item.local_user_id
                for item in hashed_document.bindings
            }
        else:
            raise ValueError("identity binding schema is unsupported")
    except ValidationError as exc:
        if any("one-to-one and unique" in error["msg"] for error in exc.errors()):
            raise ValueError(
                "Aily MCP identity bindings must be one-to-one and unique"
            ) from exc
        raise ValueError("Aily MCP identity bindings are invalid") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Aily MCP identity bindings are invalid") from exc
    return AilyMcpIdentityBindings(
        _local_user_by_aily_user=MappingProxyType(raw_mapping),
        _local_user_by_aily_user_sha256=MappingProxyType(hashed_mapping),
    )


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
    "ResolvedAilyAnalysisSource",
    "load_aily_mcp_identity_bindings",
]
