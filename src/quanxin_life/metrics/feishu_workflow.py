"""Measure Feishu workflow outcomes only from persisted audit evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path

from sqlalchemy import select

from quanxin_life.audit import AuditLedgerError, SqlAuditLedger, SqlProjectAuditLedger
from quanxin_life.core import sha256_canonical
from quanxin_life.integrations.feishu.recheck_actions import (
    RECHECK_ACTION_EVENT_TYPE,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence.database import SessionFactory
from quanxin_life.persistence.models import (
    FeishuEventReceipt,
    GlobalToolResultBindingRecord,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    ToolResultRecord,
)

FEISHU_WORKFLOW_METRIC_VERSION = "feishu-workflow-metrics-v1"
TOOL_SELECTION_CORPUS_SCHEMA_VERSION = "feishu-tool-selection-corpus-v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_VALUE_QUANTUM = Decimal("0.000001")


class WorkflowMetricName(StrEnum):
    FILE_TO_RESULT_LATENCY = "file_to_result_mean_seconds"
    TRACEABILITY_RATE = "successful_job_traceability_ratio"
    DUPLICATE_JOB_RATE = "duplicate_semantic_job_ratio"
    RECHECK_ACTION_SUCCESS_RATE = "recheck_action_success_ratio"
    TOOL_SELECTION_ACCURACY = "tool_selection_accuracy_ratio"


class FeishuWorkflowMetricEvidenceError(RuntimeError):
    """Raised when persisted evidence cannot support a requested metric."""


@dataclass(frozen=True, slots=True)
class ToolSelectionLabel:
    """Reviewed crosswalk from a pre-selection intent to one persisted Aily job."""

    intent_sha256: str
    request_sha256: str
    expected_task_type: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent_sha256, str)
            or _SHA256.fullmatch(self.intent_sha256) is None
        ):
            raise ValueError("intent_sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.request_sha256, str)
            or _SHA256.fullmatch(self.request_sha256) is None
        ):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.expected_task_type, str)
            or _SAFE_TEXT.fullmatch(self.expected_task_type) is None
        ):
            raise ValueError("expected_task_type must be a safe task identifier")
        try:
            FeishuAnalysisTask(self.expected_task_type)
        except ValueError as exc:
            raise ValueError(
                "expected_task_type must be a supported Feishu analysis task"
            ) from exc


@dataclass(frozen=True, slots=True)
class ToolSelectionCorpus:
    schema_version: str
    corpus_id: str
    corpus_version: str
    labels: tuple[ToolSelectionLabel, ...]
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class FeishuWorkflowMetric:
    metric_name: WorkflowMetricName
    metric_version: str
    window_start: datetime
    window_end: datetime
    sample_count: int
    numerator: Decimal
    denominator: Decimal
    value: Decimal
    unit: str
    evidence_sha256: str
    corpus_id: str | None = None
    corpus_version: str | None = None
    corpus_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _AuditedResultEvidence:
    result: ToolResultRecord
    ledger_valid: bool
    manifest: Mapping[str, object]


def build_tool_selection_corpus(
    *,
    corpus_id: str,
    corpus_version: str,
    labels: Sequence[ToolSelectionLabel],
) -> ToolSelectionCorpus:
    """Build a versioned, tamper-evident label set without any raw prompts."""

    checked_id = _safe_text(corpus_id, field_name="corpus_id")
    checked_version = _safe_text(corpus_version, field_name="corpus_version")
    normalized = tuple(
        ToolSelectionLabel(
            intent_sha256=label.intent_sha256,
            request_sha256=label.request_sha256,
            expected_task_type=label.expected_task_type,
        )
        for label in labels
    )
    intent_hashes = [label.intent_sha256 for label in normalized]
    if len(intent_hashes) != len(set(intent_hashes)):
        raise ValueError("tool selection corpus intent hashes must be unique")
    request_hashes = [label.request_sha256 for label in normalized]
    if len(request_hashes) != len(set(request_hashes)):
        raise ValueError("tool selection corpus request hashes must be unique")
    digest = sha256_canonical(
        {
            "schema_version": TOOL_SELECTION_CORPUS_SCHEMA_VERSION,
            "corpus_id": checked_id,
            "corpus_version": checked_version,
            "labels": [
                {
                    "expected_task_type": label.expected_task_type,
                    "intent_sha256": label.intent_sha256,
                    "request_sha256": label.request_sha256,
                }
                for label in sorted(normalized, key=lambda item: item.request_sha256)
            ],
        }
    )
    return ToolSelectionCorpus(
        schema_version=TOOL_SELECTION_CORPUS_SCHEMA_VERSION,
        corpus_id=checked_id,
        corpus_version=checked_version,
        labels=normalized,
        corpus_sha256=digest,
    )


def load_tool_selection_corpus(path: str | Path) -> ToolSelectionCorpus:
    """Load an exact versioned label corpus and verify its canonical SHA-256."""

    checked_path = Path(path)
    try:
        raw = json.loads(checked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus cannot be read as UTF-8 JSON"
        ) from exc
    expected_fields = {
        "corpus_id",
        "corpus_sha256",
        "corpus_version",
        "labels",
        "schema_version",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus fields do not match the schema"
        )
    if raw.get("schema_version") != TOOL_SELECTION_CORPUS_SCHEMA_VERSION:
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus schema version is unsupported"
        )
    labels_value = raw.get("labels")
    if not isinstance(labels_value, list):
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus labels must be a list"
        )
    labels: list[ToolSelectionLabel] = []
    for item in labels_value:
        if not isinstance(item, dict) or set(item) != {
            "expected_task_type",
            "intent_sha256",
            "request_sha256",
        }:
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus label fields are invalid"
            )
        try:
            labels.append(
                ToolSelectionLabel(
                    intent_sha256=item["intent_sha256"],
                    request_sha256=item["request_sha256"],
                    expected_task_type=item["expected_task_type"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus label is invalid"
            ) from exc
    try:
        corpus = build_tool_selection_corpus(
            corpus_id=raw["corpus_id"],
            corpus_version=raw["corpus_version"],
            labels=tuple(labels),
        )
    except (TypeError, ValueError) as exc:
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus identity is invalid"
        ) from exc
    if raw.get("corpus_sha256") != corpus.corpus_sha256:
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection corpus SHA-256 does not match"
        )
    return corpus


class SqlAlchemyFeishuWorkflowMetrics:
    """Read operational evidence without mutating receipts or ToolResults."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory
        self._global_ledger = SqlAuditLedger(session_factory)

    def measure_file_to_result_latency(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> FeishuWorkflowMetric:
        start, end = _window(window_start, window_end)
        file_jobs = tuple(
            row
            for row in self._receipt_rows(start, end)
            if row.job_id is not None
            and row.job_origin == "FEISHU"
            and row.source_job_id is None
            and row.message_id is not None
            and row.file_key is not None
        )
        if not file_jobs:
            raise FeishuWorkflowMetricEvidenceError(
                "no file-to-result latency evidence in the requested window"
            )
        if any(row.job_status not in {"SUCCEEDED", "REJECTED"} for row in file_jobs):
            raise FeishuWorkflowMetricEvidenceError(
                "file-to-result latency window contains undelivered file jobs"
            )
        rows = file_jobs
        total_seconds = Decimal(0)
        evidence: list[dict[str, object]] = []
        for row in rows:
            if row.job_created_at is None or row.job_completed_at is None:
                raise FeishuWorkflowMetricEvidenceError(
                    "completed file job timestamp evidence is incomplete"
                )
            created = _utc(row.job_created_at)
            completed = _utc(row.job_completed_at)
            if completed < created:
                raise FeishuWorkflowMetricEvidenceError(
                    "completed file job timestamp evidence is inconsistent"
                )
            if row.result_card_message_id is None or (
                row.job_status == "SUCCEEDED" and row.analysis_result_id is None
            ):
                raise FeishuWorkflowMetricEvidenceError(
                    "completed file job result delivery evidence is incomplete"
                )
            elapsed_microseconds = _timedelta_microseconds(completed - created)
            elapsed_seconds = Decimal(elapsed_microseconds) / Decimal(1_000_000)
            total_seconds += elapsed_seconds
            evidence.append(
                {
                    "completed_at": completed.isoformat(),
                    "created_at": created.isoformat(),
                    "job_id": row.job_id,
                    "result_id": row.analysis_result_id or row.validation_result_id,
                }
            )
        return _metric(
            metric_name=WorkflowMetricName.FILE_TO_RESULT_LATENCY,
            start=start,
            end=end,
            sample_count=len(rows),
            numerator=total_seconds,
            denominator=Decimal(len(rows)),
            unit="seconds",
            evidence=evidence,
        )

    def measure_traceability_rate(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> FeishuWorkflowMetric:
        start, end = _window(window_start, window_end)
        jobs = tuple(
            row
            for row in self._receipt_rows(start, end)
            if row.job_id is not None and row.job_status == "SUCCEEDED"
        )
        if not jobs:
            raise FeishuWorkflowMetricEvidenceError(
                "no successful-job traceability evidence in the requested window"
            )
        result_ids = tuple(
            row.analysis_result_id
            for row in jobs
            if row.analysis_result_id is not None
        )
        results = self._result_evidence(result_ids)
        traceable = 0
        evidence: list[dict[str, object]] = []
        for job in jobs:
            audited = (
                results.get(job.analysis_result_id)
                if job.analysis_result_id is not None
                else None
            )
            valid = _traceable_job(job, audited)
            traceable += int(valid)
            evidence.append(
                {
                    "audit_manifest": (
                        audited.manifest
                        if audited is not None
                        else {"status": "MISSING_RESULT"}
                    ),
                    "job_id": job.job_id,
                    "job_manifest": _traceability_job_manifest(job),
                    "result_id": job.analysis_result_id,
                    "traceable": valid,
                }
            )
        return _metric(
            metric_name=WorkflowMetricName.TRACEABILITY_RATE,
            start=start,
            end=end,
            sample_count=len(jobs),
            numerator=Decimal(traceable),
            denominator=Decimal(len(jobs)),
            unit="ratio",
            evidence=evidence,
        )

    def measure_duplicate_job_rate(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> FeishuWorkflowMetric:
        start, end = _window(window_start, window_end)
        jobs = tuple(
            row for row in self._receipt_rows(start, end) if row.job_id is not None
        )
        if not jobs:
            raise FeishuWorkflowMetricEvidenceError(
                "no semantic job identity evidence in the requested window"
            )
        identities: list[str] = []
        evidence: list[dict[str, object]] = []
        for job in jobs:
            identity = _semantic_job_identity(job)
            identities.append(identity)
            evidence.append(
                {
                    "job_id": job.job_id,
                    "semantic_identity": identity,
                }
            )
        duplicate_count = sum(count - 1 for count in Counter(identities).values())
        return _metric(
            metric_name=WorkflowMetricName.DUPLICATE_JOB_RATE,
            start=start,
            end=end,
            sample_count=len(jobs),
            numerator=Decimal(duplicate_count),
            denominator=Decimal(len(jobs)),
            unit="ratio",
            evidence=evidence,
        )

    def measure_recheck_action_success_rate(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> FeishuWorkflowMetric:
        start, end = _window(window_start, window_end)
        actions = tuple(
            row
            for row in self._receipt_rows(start, end)
            if row.event_type == RECHECK_ACTION_EVENT_TYPE
        )
        if not actions:
            raise FeishuWorkflowMetricEvidenceError(
                "no recheck action evidence in the requested window"
            )
        if any(row.status != "PROCESSED" for row in actions):
            raise FeishuWorkflowMetricEvidenceError(
                "recheck action evidence is not terminal"
            )
        evidence: list[dict[str, object]] = []
        for row in actions:
            if row.processed_at is None:
                raise FeishuWorkflowMetricEvidenceError(
                    "processed recheck action timestamp evidence is incomplete"
                )
            evidence.append(
                {
                    "attempt_count": row.attempt_count,
                    "event_id": row.event_id,
                    "status": row.status,
                }
            )
        return _metric(
            metric_name=WorkflowMetricName.RECHECK_ACTION_SUCCESS_RATE,
            start=start,
            end=end,
            sample_count=len(actions),
            numerator=Decimal(len(actions)),
            denominator=Decimal(len(actions)),
            unit="ratio",
            evidence=evidence,
        )

    def measure_tool_selection_accuracy(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
        corpus: ToolSelectionCorpus,
    ) -> FeishuWorkflowMetric:
        start, end = _window(window_start, window_end)
        if corpus.schema_version != TOOL_SELECTION_CORPUS_SCHEMA_VERSION:
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus schema version is unsupported"
            )
        checked_corpus = build_tool_selection_corpus(
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.corpus_version,
            labels=corpus.labels,
        )
        if checked_corpus.corpus_sha256 != corpus.corpus_sha256:
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus SHA-256 does not match"
            )
        if not checked_corpus.labels:
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus is empty"
            )
        expected = {label.request_sha256: label for label in checked_corpus.labels}
        matched: dict[str, FeishuEventReceipt] = {}
        for row in self._receipt_rows(start, end):
            request_hash = row.job_request_sha256
            if row.job_origin != "AILY" or request_hash not in expected:
                continue
            if request_hash in matched:
                raise FeishuWorkflowMetricEvidenceError(
                    "tool selection corpus does not match unique persisted jobs"
                )
            matched[request_hash] = row
        if set(matched) != set(expected):
            raise FeishuWorkflowMetricEvidenceError(
                "tool selection corpus does not match complete persisted jobs"
            )
        correct = 0
        evidence: list[dict[str, object]] = []
        for request_hash in sorted(expected):
            row = matched[request_hash]
            if not row.task_type:
                raise FeishuWorkflowMetricEvidenceError(
                    "persisted tool selection task identity is incomplete"
                )
            label = expected[request_hash]
            is_correct = row.task_type == label.expected_task_type
            correct += int(is_correct)
            evidence.append(
                {
                    "expected_task_type": label.expected_task_type,
                    "intent_sha256": label.intent_sha256,
                    "job_id": row.job_id,
                    "request_sha256": request_hash,
                    "selected_task_type": row.task_type,
                }
            )
        return _metric(
            metric_name=WorkflowMetricName.TOOL_SELECTION_ACCURACY,
            start=start,
            end=end,
            sample_count=len(expected),
            numerator=Decimal(correct),
            denominator=Decimal(len(expected)),
            unit="ratio",
            evidence=evidence,
            corpus=checked_corpus,
        )

    def _receipt_rows(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[FeishuEventReceipt, ...]:
        session = self._session_factory()
        try:
            return tuple(
                session.scalars(
                    select(FeishuEventReceipt)
                    .where(
                        FeishuEventReceipt.received_at >= start,
                        FeishuEventReceipt.received_at < end,
                    )
                    .order_by(
                        FeishuEventReceipt.received_at,
                        FeishuEventReceipt.event_id,
                    )
                ).all()
            )
        finally:
            session.close()

    def _result_evidence(
        self,
        result_ids: Sequence[str],
    ) -> dict[str, _AuditedResultEvidence]:
        if not result_ids:
            return {}
        checked_ids = tuple(sorted(set(result_ids)))
        session = self._session_factory()
        try:
            result_rows = {
                row.id: row
                for row in session.scalars(
                    select(ToolResultRecord).where(ToolResultRecord.id.in_(checked_ids))
                ).all()
            }
            provenance_rows: dict[str, list[ProvenanceRecordRow]] = {
                result_id: [] for result_id in checked_ids
            }
            for provenance_row in session.scalars(
                select(ProvenanceRecordRow).where(
                    ProvenanceRecordRow.tool_result_id.in_(checked_ids)
                )
            ).all():
                provenance_rows.setdefault(
                    provenance_row.tool_result_id, []
                ).append(provenance_row)
            global_bindings = {
                binding_row.result_id: binding_row
                for binding_row in session.scalars(
                    select(GlobalToolResultBindingRecord).where(
                        GlobalToolResultBindingRecord.result_id.in_(checked_ids)
                    )
                ).all()
            }
            project_bindings = {
                binding_row.result_id: binding_row
                for binding_row in session.scalars(
                    select(ProjectToolResultBindingRecord).where(
                        ProjectToolResultBindingRecord.result_id.in_(checked_ids)
                    )
                ).all()
            }
        finally:
            session.close()
        evidence: dict[str, _AuditedResultEvidence] = {}
        for result_id, result_row in result_rows.items():
            provenance = tuple(provenance_rows.get(result_id, ()))
            global_binding = global_bindings.get(result_id)
            project_binding = project_bindings.get(result_id)
            evidence[result_id] = _AuditedResultEvidence(
                result=result_row,
                ledger_valid=self._valid_ledger_binding(
                    result_row,
                    provenance=provenance,
                    global_binding=global_binding,
                    project_binding=project_binding,
                ),
                manifest=_result_evidence_manifest(
                    result_row,
                    provenance=provenance,
                    global_binding=global_binding,
                    project_binding=project_binding,
                ),
            )
        return evidence

    def _valid_ledger_binding(
        self,
        row: ToolResultRecord,
        *,
        provenance: tuple[ProvenanceRecordRow, ...],
        global_binding: GlobalToolResultBindingRecord | None,
        project_binding: ProjectToolResultBindingRecord | None,
    ) -> bool:
        if (global_binding is None) == (project_binding is None):
            return False
        try:
            rebuilt = SqlProjectAuditLedger.rebuild_persisted_result(row, provenance)
            if global_binding is not None:
                return self._global_ledger.resolve_registered_result(row.id) == rebuilt
            assert project_binding is not None
            SqlProjectAuditLedger.verify_persisted_binding(project_binding, rebuilt)
        except (AuditLedgerError, TypeError, ValueError):
            return False
        return True


def _metric(
    *,
    metric_name: WorkflowMetricName,
    start: datetime,
    end: datetime,
    sample_count: int,
    numerator: Decimal,
    denominator: Decimal,
    unit: str,
    evidence: Iterable[Mapping[str, object]],
    corpus: ToolSelectionCorpus | None = None,
) -> FeishuWorkflowMetric:
    if sample_count < 1 or denominator <= 0:
        raise FeishuWorkflowMetricEvidenceError(
            f"no {metric_name.value} evidence in the requested window"
        )
    value = (numerator / denominator).quantize(
        _VALUE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
    evidence_items = sorted(
        (dict(item) for item in evidence),
        key=lambda item: sha256_canonical(item),
    )
    digest = sha256_canonical(
        {
            "corpus_sha256": corpus.corpus_sha256 if corpus is not None else None,
            "denominator": str(denominator),
            "evidence": evidence_items,
            "metric_name": metric_name.value,
            "metric_version": FEISHU_WORKFLOW_METRIC_VERSION,
            "numerator": str(numerator),
            "window_end": end.isoformat(),
            "window_start": start.isoformat(),
        }
    )
    return FeishuWorkflowMetric(
        metric_name=metric_name,
        metric_version=FEISHU_WORKFLOW_METRIC_VERSION,
        window_start=start,
        window_end=end,
        sample_count=sample_count,
        numerator=numerator,
        denominator=denominator,
        value=value,
        unit=unit,
        evidence_sha256=digest,
        corpus_id=corpus.corpus_id if corpus is not None else None,
        corpus_version=corpus.corpus_version if corpus is not None else None,
        corpus_sha256=corpus.corpus_sha256 if corpus is not None else None,
    )


def _traceable_job(
    job: FeishuEventReceipt,
    audited: _AuditedResultEvidence | None,
) -> bool:
    result = audited.result if audited is not None else None
    return bool(
        job.run_id
        and job.record_batch_id
        and _is_sha256(job.input_file_sha256)
        and job.analysis_result_id
        and result is not None
        and audited is not None
        and audited.ledger_valid
        and result.tool_name == job.task_type
        and result.tool_version
        and result.model_version
        and result.data_version
        and result.feature_version
        and _is_sha256(result.input_hash)
        and result.values_json
    )


def _traceability_job_manifest(job: FeishuEventReceipt) -> Mapping[str, object]:
    return {
        "analysis_result_id": job.analysis_result_id,
        "cell_reference": job.cell_reference,
        "input_file_sha256": job.input_file_sha256,
        "job_completed_at": (
            _evidence_timestamp(job.job_completed_at)
            if job.job_completed_at is not None
            else None
        ),
        "job_created_at": (
            _evidence_timestamp(job.job_created_at)
            if job.job_created_at is not None
            else None
        ),
        "job_id": job.job_id,
        "job_origin": job.job_origin,
        "job_status": job.job_status,
        "record_batch_id": job.record_batch_id,
        "run_id": job.run_id,
        "scenario_context_id": job.scenario_context_id,
        "source_job_id": job.source_job_id,
        "task_type": job.task_type,
    }


def _result_evidence_manifest(
    row: ToolResultRecord,
    *,
    provenance: tuple[ProvenanceRecordRow, ...],
    global_binding: GlobalToolResultBindingRecord | None,
    project_binding: ProjectToolResultBindingRecord | None,
) -> Mapping[str, object]:
    result_record_sha256 = sha256_canonical(
        {
            "created_at": _evidence_timestamp(row.created_at),
            "data_version": row.data_version,
            "feature_version": row.feature_version,
            "input_hash": row.input_hash,
            "model_version": row.model_version,
            "result_id": row.id,
            "tool_name": row.tool_name,
            "tool_version": row.tool_version,
            "uncertainty": row.uncertainty_json,
            "values": row.values_json,
            "warnings": row.warnings_json,
        }
    )
    provenance_sha256s = sorted(
        sha256_canonical(
            {
                "created_at": _evidence_timestamp(item.created_at),
                "description": item.description,
                "sha256": item.sha256,
                "source_id": item.source_id,
                "source_kind": item.source_kind,
                "uri": item.uri,
            }
        )
        for item in provenance
    )
    bindings = tuple(
        item
        for item in (
            _binding_evidence("GLOBAL", global_binding),
            _binding_evidence("PROJECT", project_binding),
        )
        if item is not None
    )
    return {
        "binding_evidence": bindings,
        "provenance_record_sha256s": provenance_sha256s,
        "result_record_sha256": result_record_sha256,
    }


def _binding_evidence(
    kind: str,
    binding: GlobalToolResultBindingRecord | ProjectToolResultBindingRecord | None,
) -> Mapping[str, object] | None:
    if binding is None:
        return None
    return {
        "binding_kind": kind,
        "binding_schema_version": binding.binding_schema_version,
        "binding_sha256": binding.binding_sha256,
        "created_at": _evidence_timestamp(binding.created_at),
        "result_sha256": binding.result_sha256,
    }


def _evidence_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        return f"NAIVE:{value.isoformat()}"
    return value.astimezone(UTC).isoformat()


def _semantic_job_identity(job: FeishuEventReceipt) -> str:
    if not job.job_id or not job.task_type:
        raise FeishuWorkflowMetricEvidenceError(
            "semantic job identity evidence is incomplete"
        )
    if job.job_origin == "AILY":
        if not _is_sha256(job.job_request_sha256):
            raise FeishuWorkflowMetricEvidenceError(
                "Aily semantic job request identity is incomplete"
            )
        return f"AILY:{job.job_request_sha256}"
    if job.source_job_id is not None:
        return ":".join(
            (
                "FEISHU_SIBLING",
                job.source_job_id,
                job.task_type,
                job.scenario_context_id or "",
                job.default_scenario_profile_id or "",
                job.default_scenario_profile_version or "",
            )
        )
    return f"FEISHU_EVENT:{job.event_id}:{job.task_type}"


def _window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    checked_start = _utc(start)
    checked_end = _utc(end)
    if checked_start >= checked_end:
        raise ValueError("metric window_start must be before window_end")
    return checked_start, checked_end


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("metric timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _timedelta_microseconds(value: timedelta) -> int:
    days = int(value.days)
    seconds = int(value.seconds)
    microseconds = int(value.microseconds)
    return ((days * 86_400) + seconds) * 1_000_000 + microseconds


def _safe_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a safe identifier")
    checked = value.strip()
    if _SAFE_TEXT.fullmatch(checked) is None:
        raise ValueError(f"{field_name} must be a safe identifier")
    return checked


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "FEISHU_WORKFLOW_METRIC_VERSION",
    "TOOL_SELECTION_CORPUS_SCHEMA_VERSION",
    "FeishuWorkflowMetric",
    "FeishuWorkflowMetricEvidenceError",
    "SqlAlchemyFeishuWorkflowMetrics",
    "ToolSelectionCorpus",
    "ToolSelectionLabel",
    "WorkflowMetricName",
    "build_tool_selection_corpus",
    "load_tool_selection_corpus",
]
