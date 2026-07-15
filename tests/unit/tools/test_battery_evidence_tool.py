"""Contracts for source-bound battery knowledge retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.tools.registry import StandardToolName, ToolRegistry

if TYPE_CHECKING:
    from quanxin_life.tools.battery_evidence import VerifiedKnowledgeScope


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="public-lfp-review-fixture",
            source_kind=SourceKind.OBSERVED,
            uri="https://example.invalid/lfp-review",
            sha256=sha256_canonical({"fixture": "lfp-review"}),
            description="Reviewed public LFP evidence document fixture",
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
        ),
    )


def _scope() -> VerifiedKnowledgeScope:
    from quanxin_life.tools.battery_evidence import VerifiedKnowledgeScope

    return VerifiedKnowledgeScope.model_validate(
        {
            "knowledge_scope_id": "lfp-public-evidence-v1",
            "corpus_version": "lfp-corpus-v1",
            "index_version": "pgvector-bm25-v1",
            "retrieval_policy_version": "hybrid-rerank-v1",
            "approved_documents": [
                {
                    "document_id": "lfp-review-2025",
                    "source_id": "public-lfp-review-fixture",
                    "title": "Public LFP degradation review",
                    "source_uri": "https://example.invalid/lfp-review",
                    "source_sha256": sha256_canonical({"fixture": "lfp-review"}),
                    "license_name": "CC-BY-4.0",
                    "evidence_level": EvidenceLevel.DOMAIN_KNOWLEDGE.value,
                }
            ],
            "provenance": [item.model_dump(mode="json") for item in _provenance()],
        }
    )


class _ScopeResolver:
    def __init__(self, scope: VerifiedKnowledgeScope) -> None:
        self.scope = scope

    def resolve_verified_knowledge_scope(self, knowledge_scope_id: str) -> VerifiedKnowledgeScope:
        if knowledge_scope_id != self.scope.knowledge_scope_id:
            raise ValueError("knowledge scope was not found")
        return self.scope


class _EvidenceBackend:
    def search(self, *, scope: VerifiedKnowledgeScope, query: str, top_k: int):
        from quanxin_life.tools.battery_evidence import HybridEvidenceSearchResponse

        assert query == "磷酸铁锂温度退化机理"
        assert top_k == 3
        return HybridEvidenceSearchResponse.model_validate(
            {
                "backend_version": "pgvector-bm25-reranker-v1",
                "retrieval_mode": "pgvector_bm25_reranker",
                "hits": [
                    {
                        "document_id": "lfp-review-2025",
                        "chunk_id": "lfp-review-2025:p12:c03",
                        "page_number": 12,
                        "section_label": "Temperature-dependent ageing",
                        "excerpt": (
                            "The reviewed evidence discusses temperature-dependent ageing pathways."
                        ),
                        "evidence_level": EvidenceLevel.DOMAIN_KNOWLEDGE.value,
                        "bm25_score": 4.2,
                        "vector_score": 0.81,
                        "reranker_score": 0.92,
                        "final_score": 0.89,
                    }
                ],
                "warnings": [],
            }
        )


def test_retrieve_battery_evidence_returns_source_checked_citations() -> None:
    from quanxin_life.tools.battery_evidence import (
        BATTERY_EVIDENCE_ARTIFACT_TYPE,
        RetrieveBatteryEvidenceToolInput,
        register_retrieve_battery_evidence_tool,
    )

    scope = _scope()
    registry = ToolRegistry()
    register_retrieve_battery_evidence_tool(
        registry,
        resolver=_ScopeResolver(scope),
        backend=_EvidenceBackend(),
    )

    result = registry.execute(
        StandardToolName.RETRIEVE_BATTERY_EVIDENCE,
        RetrieveBatteryEvidenceToolInput(
            knowledge_scope_id=scope.knowledge_scope_id,
            query="磷酸铁锂温度退化机理",
            top_k=3,
        ),
    )

    assert result.tool_name == StandardToolName.RETRIEVE_BATTERY_EVIDENCE.value
    assert result.tool_version == "battery-evidence-retrieval-tool-v1"
    assert result.data_version == "lfp-corpus-v1"
    assert result.feature_version == "hybrid-rerank-v1"
    assert result.values["artifact_type"] == BATTERY_EVIDENCE_ARTIFACT_TYPE
    artifact = result.values["artifact"]
    assert artifact["knowledge_scope_id"] == scope.knowledge_scope_id
    assert artifact["retrieval_mode"] == "pgvector_bm25_reranker"
    assert artifact["citations"][0]["page_number"] == 12
    assert artifact["citations"][0]["source_sha256"] == sha256_canonical(
        {"fixture": "lfp-review"}
    )
    assert "RETRIEVAL_OUTPUT_IS_NOT_A_NUMERICAL_PREDICTION" in result.warnings


def test_evidence_public_input_rejects_document_content_scores_and_provenance() -> None:
    from quanxin_life.tools.battery_evidence import RetrieveBatteryEvidenceToolInput

    with pytest.raises(ValueError, match="Extra inputs"):
        RetrieveBatteryEvidenceToolInput.model_validate(
            {
                "knowledge_scope_id": "lfp-public-evidence-v1",
                "query": "磷酸铁锂温度退化机理",
                "top_k": 3,
                "documents": [{"content": "client supplied source"}],
                "scores": [0.99],
                "provenance": ["client supplied"],
            }
        )


def test_evidence_tool_rejects_unapproved_document_or_mismatched_scope() -> None:
    from quanxin_life.tools.battery_evidence import (
        HybridEvidenceSearchResponse,
        RetrieveBatteryEvidenceToolInput,
        execute_retrieve_battery_evidence_tool,
    )

    class _UnsafeBackend:
        def search(self, *, scope: VerifiedKnowledgeScope, query: str, top_k: int):
            return HybridEvidenceSearchResponse.model_validate(
                {
                    "backend_version": "pgvector-bm25-reranker-v1",
                    "retrieval_mode": "pgvector_bm25_reranker",
                    "hits": [
                        {
                            "document_id": "unapproved-document",
                            "chunk_id": "unapproved:p1:c1",
                            "page_number": 1,
                            "section_label": "Unknown",
                            "excerpt": "Unapproved text",
                            "evidence_level": EvidenceLevel.DOMAIN_KNOWLEDGE.value,
                            "bm25_score": 1.0,
                            "vector_score": 0.5,
                            "reranker_score": 0.5,
                            "final_score": 0.5,
                        }
                    ],
                    "warnings": [],
                }
            )

    scope = _scope()
    tool_input = RetrieveBatteryEvidenceToolInput(
        knowledge_scope_id=scope.knowledge_scope_id,
        query="磷酸铁锂温度退化机理",
        top_k=3,
    )
    with pytest.raises(ValueError, match="approved document"):
        execute_retrieve_battery_evidence_tool(
            tool_input,
            resolver=_ScopeResolver(scope),
            backend=_UnsafeBackend(),
        )

    mismatched_scope = scope.model_copy(update={"knowledge_scope_id": "other"})

    class _MismatchedResolver:
        def resolve_verified_knowledge_scope(
            self, knowledge_scope_id: str
        ) -> VerifiedKnowledgeScope:
            return mismatched_scope

    with pytest.raises(ValueError, match="mismatched"):
        execute_retrieve_battery_evidence_tool(
            tool_input,
            resolver=_MismatchedResolver(),
            backend=_EvidenceBackend(),
        )
