"""Source-bound battery-domain evidence retrieval.

Retrieval is deliberately separate from numerical prediction.  A caller may
ask a text question, but the corpus scope, source metadata, citation locations,
and hybrid-ranking output are owned by a reviewed server-side resolver and
backend.  This prevents agents and clients from attaching invented references
or treating narrative evidence as a battery-model result.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.core import (
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

BATTERY_EVIDENCE_RETRIEVAL_TOOL_VERSION = "battery-evidence-retrieval-tool-v1"
BATTERY_EVIDENCE_ARTIFACT_TYPE = "quanxin_life.battery_evidence_retrieval.v1"
NOT_NUMERICAL_PREDICTION_WARNING = "RETRIEVAL_OUTPUT_IS_NOT_A_NUMERICAL_PREDICTION"
DEGRADED_RETRIEVAL_WARNING = "KNOWLEDGE_RETRIEVAL_DEGRADED_FROM_PGVECTOR_HYBRID"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _TrustedKnowledgeModel(ContractModel):
    """Strict contracts returned only by the server-side knowledge service."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ApprovedKnowledgeDocument(_TrustedKnowledgeModel):
    """One review-approved source document eligible for result citations."""

    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_sha256: Sha256
    license_name: str = Field(min_length=1)
    evidence_level: EvidenceLevel


class VerifiedKnowledgeScope(_TrustedKnowledgeModel):
    """Corpus identity and source allowlist for one scoped retrieval request."""

    knowledge_scope_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    retrieval_policy_version: str = Field(min_length=1)
    approved_documents: tuple[ApprovedKnowledgeDocument, ...] = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator(
        "knowledge_scope_id", "corpus_version", "index_version", "retrieval_policy_version"
    )
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge scope identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_unique_sources_and_observed_provenance(self) -> VerifiedKnowledgeScope:
        document_ids = [item.document_id for item in self.approved_documents]
        source_ids = [item.source_id for item in self.approved_documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("knowledge scope document_id values must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("knowledge scope source_id values must be unique")
        if not any(item.source_kind is SourceKind.OBSERVED for item in self.provenance):
            raise ValueError("knowledge scope provenance must include an OBSERVED source")
        return self


class HybridEvidenceHit(_TrustedKnowledgeModel):
    """A backend-produced citation candidate with explicit hybrid score parts."""

    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section_label: str | None = Field(default=None, min_length=1)
    excerpt: str = Field(min_length=1)
    evidence_level: EvidenceLevel
    bm25_score: float = Field(allow_inf_nan=False)
    vector_score: float = Field(allow_inf_nan=False)
    reranker_score: float = Field(allow_inf_nan=False)
    final_score: float = Field(allow_inf_nan=False)


class HybridEvidenceSearchResponse(_TrustedKnowledgeModel):
    """One deterministic backend response; query results are not a free-text claim."""

    backend_version: str = Field(min_length=1)
    retrieval_mode: str = Field(min_length=1)
    hits: tuple[HybridEvidenceHit, ...]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_unique_chunks(self) -> HybridEvidenceSearchResponse:
        chunk_ids = [item.chunk_id for item in self.hits]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("hybrid retrieval hits must use unique chunk_id values")
        return self


class VerifiedKnowledgeScopeResolver(Protocol):
    """Server-side lookup for the corpus scope approved for one request."""

    def resolve_verified_knowledge_scope(
        self, knowledge_scope_id: str
    ) -> VerifiedKnowledgeScope: ...


class HybridBatteryEvidenceBackend(Protocol):
    """Backend contract for pgvector + BM25 + reranker or a declared fallback."""

    def search(
        self,
        *,
        scope: VerifiedKnowledgeScope,
        query: str,
        top_k: int,
    ) -> HybridEvidenceSearchResponse: ...


class RetrieveBatteryEvidenceToolInput(ContractModel):
    """Public retrieval request with no document payload, score or citation fields."""

    knowledge_scope_id: str = Field(min_length=1)
    query: str = Field(min_length=2, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("knowledge_scope_id", "query")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge retrieval text fields must not be blank")
        return normalized


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_scope(
    resolver: VerifiedKnowledgeScopeResolver,
    knowledge_scope_id: str,
) -> VerifiedKnowledgeScope:
    resolved = resolver.resolve_verified_knowledge_scope(knowledge_scope_id)
    try:
        scope = VerifiedKnowledgeScope.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(f"trusted knowledge scope violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted knowledge scope is invalid") from exc
    if scope.knowledge_scope_id != knowledge_scope_id:
        raise ValueError("knowledge resolver returned a mismatched knowledge_scope_id")
    return scope


def _resolve_backend_response(
    *,
    backend: HybridBatteryEvidenceBackend,
    scope: VerifiedKnowledgeScope,
    query: str,
    top_k: int,
) -> HybridEvidenceSearchResponse:
    resolved = backend.search(scope=scope, query=query, top_k=top_k)
    try:
        response = HybridEvidenceSearchResponse.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(
            f"hybrid evidence backend returned an invalid response: {message}"
        ) from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("hybrid evidence backend returned an invalid response") from exc
    if len(response.hits) > top_k:
        raise ValueError("hybrid evidence backend returned more hits than the requested top_k")
    return response


def _citation_payload(
    *,
    hit: HybridEvidenceHit,
    document: ApprovedKnowledgeDocument,
) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "chunk_id": hit.chunk_id,
        "title": document.title,
        "source_id": document.source_id,
        "source_uri": document.source_uri,
        "source_sha256": document.source_sha256,
        "license_name": document.license_name,
        "page_number": hit.page_number,
        "section_label": hit.section_label,
        "excerpt": hit.excerpt,
        "evidence_level": hit.evidence_level.value,
        "bm25_score": hit.bm25_score,
        "vector_score": hit.vector_score,
        "reranker_score": hit.reranker_score,
        "final_score": hit.final_score,
    }


def _validate_and_cite_hits(
    *,
    scope: VerifiedKnowledgeScope,
    response: HybridEvidenceSearchResponse,
) -> list[dict[str, object]]:
    approved_documents = {item.document_id: item for item in scope.approved_documents}
    citations: list[dict[str, object]] = []
    for hit in response.hits:
        document = approved_documents.get(hit.document_id)
        if document is None:
            raise ValueError("hybrid evidence hit does not reference an approved document")
        if hit.evidence_level is not document.evidence_level:
            raise ValueError("hybrid evidence hit evidence_level must match its approved document")
        citations.append(_citation_payload(hit=hit, document=document))
    return citations


def execute_retrieve_battery_evidence_tool(
    input_value: RetrieveBatteryEvidenceToolInput,
    *,
    resolver: VerifiedKnowledgeScopeResolver,
    backend: HybridBatteryEvidenceBackend,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Return source-checked explanatory evidence without a numeric prediction."""

    validated_input = RetrieveBatteryEvidenceToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    scope = _resolve_verified_scope(resolver, validated_input.knowledge_scope_id)
    response = _resolve_backend_response(
        backend=backend,
        scope=scope,
        query=validated_input.query,
        top_k=validated_input.top_k,
    )
    citations = _validate_and_cite_hits(scope=scope, response=response)
    warnings = [NOT_NUMERICAL_PREDICTION_WARNING, *response.warnings]
    if response.retrieval_mode != "pgvector_bm25_reranker":
        warnings.append(DEGRADED_RETRIEVAL_WARNING)
    artifact = {
        "knowledge_scope_id": scope.knowledge_scope_id,
        "corpus_version": scope.corpus_version,
        "index_version": scope.index_version,
        "retrieval_policy_version": scope.retrieval_policy_version,
        "backend_version": response.backend_version,
        "retrieval_mode": response.retrieval_mode,
        "query": validated_input.query,
        "top_k": validated_input.top_k,
        "citations": citations,
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.RETRIEVE_BATTERY_EVIDENCE.value,
        tool_version=BATTERY_EVIDENCE_RETRIEVAL_TOOL_VERSION,
        model_version=response.backend_version,
        data_version=scope.corpus_version,
        feature_version=scope.retrieval_policy_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={"artifact_type": BATTERY_EVIDENCE_ARTIFACT_TYPE, "artifact": artifact},
        uncertainty=None,
        warnings=list(dict.fromkeys(warnings)),
        provenance=list(scope.provenance),
        created_at=_execution_timestamp(clock),
    )


def register_retrieve_battery_evidence_tool(
    registry: ToolRegistry,
    *,
    resolver: VerifiedKnowledgeScopeResolver,
    backend: HybridBatteryEvidenceBackend,
    clock: Clock = _utc_now,
) -> RegisteredTool[RetrieveBatteryEvidenceToolInput]:
    """Bind one reviewed corpus resolver and hybrid backend to the shared registry."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.RETRIEVE_BATTERY_EVIDENCE,
            tool_version=BATTERY_EVIDENCE_RETRIEVAL_TOOL_VERSION,
            input_model=RetrieveBatteryEvidenceToolInput,
            executor=lambda input_value: execute_retrieve_battery_evidence_tool(
                input_value,
                resolver=resolver,
                backend=backend,
                clock=clock,
            ),
        )
    )
