from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import SqlAuditLedger, SqlProjectAuditLedger
from quanxin_life.core import (
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.metrics.feishu_workflow import (
    FEISHU_WORKFLOW_METRIC_VERSION,
    TOOL_SELECTION_CORPUS_SCHEMA_VERSION,
    FeishuWorkflowMetricEvidenceError,
    SqlAlchemyFeishuWorkflowMetrics,
    ToolSelectionLabel,
    WorkflowMetricName,
    build_tool_selection_corpus,
    load_tool_selection_corpus,
)
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import SessionFactory
from quanxin_life.persistence.models import (
    FeishuBindingRow,
    FeishuEventReceipt,
    Project,
    ProvenanceRecordRow,
    ToolResultRecord,
    User,
)

START = datetime(2026, 8, 14, tzinfo=UTC)
END = START + timedelta(days=1)


def _metrics() -> tuple[SqlAlchemyFeishuWorkflowMetrics, SessionFactory]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    return SqlAlchemyFeishuWorkflowMetrics(sessions), sessions


def _job(
    *,
    event_id: str,
    job_id: str,
    created_at: datetime,
    completed_at: datetime | None,
    task_type: str = "predict_cycle_life",
    job_status: str = "SUCCEEDED",
    job_origin: str = "FEISHU",
    source_job_id: str | None = None,
    request_sha256: str | None = None,
    analysis_result_id: str | None = None,
    validation_result_id: str | None = None,
    file_upload: bool = False,
    run_id: str | None = None,
    record_batch_id: str = "batch-reviewed",
    input_file_sha256: str = "b" * 64,
) -> FeishuEventReceipt:
    return FeishuEventReceipt(
        id=uuid4().hex,
        event_id=event_id,
        event_type="im.message.receive_v1",
        payload_sha256="a" * 64,
        status="PROCESSED",
        attempt_count=1,
        received_at=created_at,
        processed_at=created_at,
        job_id=job_id,
        job_origin=job_origin,
        job_request_sha256=request_sha256,
        source_job_id=source_job_id,
        job_status=job_status,
        job_stage=job_status,
        task_type=task_type,
        run_id=run_id or str(uuid4()),
        message_id=f"message-{event_id}" if file_upload else None,
        file_key=f"file-{event_id}" if file_upload else None,
        receive_id_type="chat_id",
        event_time=created_at,
        job_attempt_count=1,
        record_batch_id=record_batch_id,
        input_file_sha256=input_file_sha256,
        validation_result_id=validation_result_id,
        analysis_result_id=analysis_result_id,
        result_card_message_id=(
            f"card-{event_id}"
            if job_status in {"SUCCEEDED", "REJECTED"}
            else None
        ),
        job_created_at=created_at,
        job_updated_at=completed_at or created_at,
        job_completed_at=completed_at,
    )


def _action(*, event_id: str, status: str) -> FeishuEventReceipt:
    processed_at = START + timedelta(minutes=2) if status == "PROCESSED" else None
    failed_at = START + timedelta(minutes=2) if status == "RETRYABLE" else None
    return FeishuEventReceipt(
        id=uuid4().hex,
        event_id=event_id,
        event_type="quanxin_life.recheck_action.v1",
        payload_sha256="c" * 64,
        status=status,
        attempt_count=1,
        received_at=START + timedelta(minutes=1),
        processed_at=processed_at,
        failed_at=failed_at,
        job_origin="FEISHU",
        job_attempt_count=0,
    )


def _add_result(
    sessions: SessionFactory,
    *,
    result_id: str,
    bound: bool,
    model_version: str = "model-v1",
) -> None:
    result = ToolResult(
        result_id=result_id,
        tool_name="predict_cycle_life",
        tool_version="advanced-rul-prediction-tool-v1",
        model_version=model_version,
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash="d" * 64,
        values={"artifact": {"cell_id": "cell-1"}},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="verified-source",
                source_kind=SourceKind.OBSERVED,
                uri="trusted-store://metric/source",
                sha256="e" * 64,
                description="Verified metric fixture",
                created_at=START,
            )
        ],
        created_at=START + timedelta(minutes=1),
    )
    if bound:
        SqlAuditLedger(sessions, clock=lambda: START).register_result(result)
        return
    with sessions.begin() as session:
        session.add(
            ToolResultRecord(
                id=result.result_id,
                run_id=None,
                agent_step_id=None,
                tool_name=result.tool_name,
                tool_version=result.tool_version,
                model_version=result.model_version,
                data_version=result.data_version,
                feature_version=result.feature_version,
                input_hash=result.input_hash,
                values_json=result.values,
                uncertainty_json=result.uncertainty,
                warnings_json=list(result.warnings),
                created_at=result.created_at,
            )
        )
        source = result.provenance[0]
        session.add(
            ProvenanceRecordRow(
                id=str(uuid4()),
                tool_result_id=result.result_id,
                source_id=source.source_id,
                source_kind=source.source_kind.value,
                uri=source.uri,
                sha256=source.sha256,
                description=source.description,
                created_at=source.created_at,
            )
        )


def test_file_to_result_latency_uses_only_completed_file_jobs() -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add_all(
            [
                _job(
                    event_id="event-1",
                    job_id="job-1",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=12),
                    analysis_result_id="result-1",
                    file_upload=True,
                ),
                _job(
                    event_id="event-2",
                    job_id="job-2",
                    created_at=START + timedelta(minutes=2),
                    completed_at=START + timedelta(minutes=2, seconds=18),
                    analysis_result_id="result-2",
                    file_upload=True,
                ),
                _job(
                    event_id="event-rejected",
                    job_id="job-rejected",
                    created_at=START + timedelta(minutes=3),
                    completed_at=START + timedelta(minutes=3, seconds=6),
                    job_status="REJECTED",
                    validation_result_id="validation-rejected",
                    file_upload=True,
                ),
            ]
        )

    first = metrics.measure_file_to_result_latency(
        window_start=START,
        window_end=END,
    )
    second = metrics.measure_file_to_result_latency(
        window_start=START,
        window_end=END,
    )

    assert first.metric_name is WorkflowMetricName.FILE_TO_RESULT_LATENCY
    assert first.metric_version == FEISHU_WORKFLOW_METRIC_VERSION
    assert first.sample_count == 3
    assert first.numerator == Decimal("36")
    assert first.denominator == Decimal("3")
    assert first.value == Decimal("12")
    assert first.unit == "seconds"
    assert first.evidence_sha256 == second.evidence_sha256


def test_latency_refuses_completed_job_with_incomplete_timestamps() -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add(
            _job(
                event_id="event-incomplete",
                job_id="job-incomplete",
                created_at=START + timedelta(minutes=1),
                completed_at=None,
                job_status="SUCCEEDED",
                analysis_result_id="result-incomplete",
                file_upload=True,
            )
        )

    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="timestamp"):
        metrics.measure_file_to_result_latency(
            window_start=START,
            window_end=END,
        )


def test_latency_refuses_a_window_with_an_undelivered_file_job() -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add_all(
            [
                _job(
                    event_id="event-delivered",
                    job_id="job-delivered",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=5),
                    analysis_result_id="result-delivered",
                    file_upload=True,
                ),
                _job(
                    event_id="event-running",
                    job_id="job-running",
                    created_at=START + timedelta(minutes=2),
                    completed_at=None,
                    job_status="RUNNING",
                    file_upload=True,
                ),
            ]
        )

    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="undelivered"):
        metrics.measure_file_to_result_latency(
            window_start=START,
            window_end=END,
        )


def test_traceability_rate_joins_persisted_tool_results_and_provenance() -> None:
    metrics, sessions = _metrics()
    traceable_result_id = str(uuid4())
    untraceable_result_id = str(uuid4())
    with sessions.begin() as session:
        session.add_all(
            [
                _job(
                    event_id="event-traceable",
                    job_id="job-traceable",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=5),
                    analysis_result_id=traceable_result_id,
                ),
                _job(
                    event_id="event-untraceable",
                    job_id="job-untraceable",
                    created_at=START + timedelta(minutes=2),
                    completed_at=START + timedelta(minutes=2, seconds=5),
                    analysis_result_id=untraceable_result_id,
                ),
            ]
        )
    _add_result(sessions, result_id=traceable_result_id, bound=True)
    _add_result(sessions, result_id=untraceable_result_id, bound=False)

    metric = metrics.measure_traceability_rate(
        window_start=START,
        window_end=END,
    )

    assert metric.metric_name is WorkflowMetricName.TRACEABILITY_RATE
    assert metric.sample_count == 2
    assert metric.numerator == Decimal("1")
    assert metric.denominator == Decimal("2")
    assert metric.value == Decimal("0.5")
    assert metric.unit == "ratio"


def test_traceability_rate_rejects_shape_only_result_without_a_ledger_binding() -> None:
    metrics, sessions = _metrics()
    result_id = str(uuid4())
    with sessions.begin() as session:
        session.add(
            _job(
                event_id="event-unbound",
                job_id="job-unbound",
                created_at=START + timedelta(minutes=1),
                completed_at=START + timedelta(minutes=1, seconds=5),
                analysis_result_id=result_id,
            )
        )
    _add_result(sessions, result_id=result_id, bound=False)

    metric = metrics.measure_traceability_rate(
        window_start=START,
        window_end=END,
    )

    assert metric.numerator == Decimal("0")
    assert metric.denominator == Decimal("1")


def test_traceability_evidence_sha_binds_valid_result_versions() -> None:
    result_id = "11111111-1111-4111-8111-111111111111"
    run_id = "22222222-2222-4222-8222-222222222222"

    def measure(model_version: str) -> str:
        metrics, sessions = _metrics()
        with sessions.begin() as session:
            session.add(
                _job(
                    event_id="event-versioned",
                    job_id="job-versioned",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=5),
                    analysis_result_id=result_id,
                    run_id=run_id,
                )
            )
        _add_result(
            sessions,
            result_id=result_id,
            bound=True,
            model_version=model_version,
        )
        return metrics.measure_traceability_rate(
            window_start=START,
            window_end=END,
        ).evidence_sha256

    assert measure("model-v1") != measure("model-v2")


def test_traceability_evidence_sha_binds_receipt_identity_fields() -> None:
    result_id = "33333333-3333-4333-8333-333333333333"

    def measure(
        *,
        run_id: str,
        record_batch_id: str,
        input_file_sha256: str,
    ) -> str:
        metrics, sessions = _metrics()
        with sessions.begin() as session:
            session.add(
                _job(
                    event_id="event-receipt-evidence",
                    job_id="job-receipt-evidence",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=5),
                    analysis_result_id=result_id,
                    run_id=run_id,
                    record_batch_id=record_batch_id,
                    input_file_sha256=input_file_sha256,
                )
            )
        _add_result(sessions, result_id=result_id, bound=True)
        return metrics.measure_traceability_rate(
            window_start=START,
            window_end=END,
        ).evidence_sha256

    baseline = measure(
        run_id="44444444-4444-4444-8444-444444444444",
        record_batch_id="batch-a",
        input_file_sha256="a" * 64,
    )
    assert baseline != measure(
        run_id="55555555-5555-4555-8555-555555555555",
        record_batch_id="batch-a",
        input_file_sha256="a" * 64,
    )
    assert baseline != measure(
        run_id="44444444-4444-4444-8444-444444444444",
        record_batch_id="batch-b",
        input_file_sha256="a" * 64,
    )
    assert baseline != measure(
        run_id="44444444-4444-4444-8444-444444444444",
        record_batch_id="batch-a",
        input_file_sha256="b" * 64,
    )


def test_traceability_accepts_a_verified_feishu_project_binding() -> None:
    metrics, sessions = _metrics()
    user_id = str(uuid4())
    project_id = str(uuid4())
    binding_id = str(uuid4())
    result_id = str(uuid4())
    with sessions.begin() as session:
        session.add_all(
            [
                User(
                    id=user_id,
                    username="metrics-project@example.test",
                    credential_hash="not-used-by-feishu",
                    must_change_credential=False,
                    role=UserRole.ADMIN.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=START,
                    updated_at=START,
                ),
                Project(
                    id=project_id,
                    owner_user_id=user_id,
                    name="Metrics project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=START,
                    updated_at=START,
                ),
                FeishuBindingRow(
                    id=binding_id,
                    project_id=project_id,
                    chat_id="chat-metrics",
                    bitable_app_token=None,
                    bitable_table_id=None,
                    user_open_id_map_json={user_id: "sender-metrics"},
                    binding_version="feishu-binding-v1",
                    status="ACTIVE",
                    created_at=START,
                ),
            ]
        )
    context_service = ProjectInvocationContextService(sessions, clock=lambda: START)
    context = context_service.resolve_feishu(
        chat_id="chat-metrics",
        sender_open_id="sender-metrics",
    )
    result = ToolResult(
        result_id=result_id,
        tool_name="predict_cycle_life",
        tool_version="advanced-rul-prediction-tool-v1",
        model_version="model-v1",
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash="d" * 64,
        values={"artifact": {"cell_id": "cell-1"}},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="verified-source",
                source_kind=SourceKind.OBSERVED,
                uri="trusted-store://metric/source",
                sha256="e" * 64,
                description="Verified metric fixture",
                created_at=START,
            )
        ],
        created_at=START + timedelta(minutes=1),
    )
    SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: START,
    ).register_result(context, result)
    with sessions.begin() as session:
        session.add(
            _job(
                event_id="event-project-bound",
                job_id="job-project-bound",
                created_at=START + timedelta(minutes=1),
                completed_at=START + timedelta(minutes=1, seconds=5),
                analysis_result_id=result_id,
            )
        )

    metric = metrics.measure_traceability_rate(
        window_start=START,
        window_end=END,
    )

    assert metric.numerator == Decimal("1")
    assert metric.denominator == Decimal("1")


def test_duplicate_rate_uses_server_semantic_job_identity() -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add_all(
            [
                _job(
                    event_id="sibling-1",
                    job_id="job-sibling-1",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=5),
                    task_type="predict_soh_trajectory",
                    source_job_id="source-job",
                ),
                _job(
                    event_id="sibling-2",
                    job_id="job-sibling-2",
                    created_at=START + timedelta(minutes=2),
                    completed_at=START + timedelta(minutes=2, seconds=5),
                    task_type="predict_soh_trajectory",
                    source_job_id="source-job",
                ),
            ]
        )

    metric = metrics.measure_duplicate_job_rate(
        window_start=START,
        window_end=END,
    )

    assert metric.metric_name is WorkflowMetricName.DUPLICATE_JOB_RATE
    assert metric.sample_count == 2
    assert metric.numerator == Decimal("1")
    assert metric.denominator == Decimal("2")
    assert metric.value == Decimal("0.5")


def test_recheck_action_success_uses_terminal_receipt_state() -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add_all(
            [
                _action(event_id="recheck:success-1", status="PROCESSED"),
                _action(event_id="recheck:success-2", status="PROCESSED"),
            ]
        )

    metric = metrics.measure_recheck_action_success_rate(
        window_start=START,
        window_end=END,
    )

    assert metric.metric_name is WorkflowMetricName.RECHECK_ACTION_SUCCESS_RATE
    assert metric.sample_count == 2
    assert metric.numerator == Decimal("2")
    assert metric.denominator == Decimal("2")
    assert metric.value == Decimal("1")


@pytest.mark.parametrize("status", ("IN_PROGRESS", "RETRYABLE"))
def test_recheck_action_metric_refuses_nonterminal_evidence(status: str) -> None:
    metrics, sessions = _metrics()
    with sessions.begin() as session:
        session.add(_action(event_id=f"recheck:{status}", status=status))

    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="terminal"):
        metrics.measure_recheck_action_success_rate(
            window_start=START,
            window_end=END,
        )


def test_tool_selection_accuracy_requires_complete_versioned_corpus_matches() -> None:
    metrics, sessions = _metrics()
    first_hash = sha256_canonical(
        {
            "schema_version": "aily-analysis-task-v2",
            "job_origin": "AILY",
            "source_job_id": "source-1",
            "record_batch_id": "batch-reviewed",
            "task_type": "predict_cycle_life",
            "scenario_context_id": None,
        }
    )
    second_hash = sha256_canonical(
        {
            "schema_version": "aily-analysis-task-v2",
            "job_origin": "AILY",
            "source_job_id": "source-2",
            "record_batch_id": "batch-reviewed",
            "task_type": "predict_cycle_life",
            "scenario_context_id": None,
        }
    )
    cycle_life_intent_hash = sha256_canonical(
        {"schema_version": "aily-intent-sample-v1", "sample_id": "cycle-life-1"}
    )
    soh_intent_hash = sha256_canonical(
        {"schema_version": "aily-intent-sample-v1", "sample_id": "soh-1"}
    )
    with sessions.begin() as session:
        session.add_all(
            [
                _job(
                    event_id="aily-1",
                    job_id="aily-job-1",
                    created_at=START + timedelta(minutes=1),
                    completed_at=START + timedelta(minutes=1, seconds=2),
                    task_type="predict_cycle_life",
                    job_origin="AILY",
                    request_sha256=first_hash,
                ),
                _job(
                    event_id="aily-2",
                    job_id="aily-job-2",
                    created_at=START + timedelta(minutes=2),
                    completed_at=START + timedelta(minutes=2, seconds=2),
                    task_type="predict_cycle_life",
                    job_origin="AILY",
                    request_sha256=second_hash,
                ),
            ]
        )
    corpus = build_tool_selection_corpus(
        corpus_id="reviewed-feishu-intents",
        corpus_version="v1",
        labels=(
            ToolSelectionLabel(
                intent_sha256=cycle_life_intent_hash,
                request_sha256=first_hash,
                expected_task_type="predict_cycle_life",
            ),
            ToolSelectionLabel(
                intent_sha256=soh_intent_hash,
                request_sha256=second_hash,
                expected_task_type="predict_soh_trajectory",
            ),
        ),
    )

    metric = metrics.measure_tool_selection_accuracy(
        window_start=START,
        window_end=END,
        corpus=corpus,
    )

    assert metric.metric_name is WorkflowMetricName.TOOL_SELECTION_ACCURACY
    assert metric.sample_count == 2
    assert metric.numerator == Decimal("1")
    assert metric.denominator == Decimal("2")
    assert metric.value == Decimal("0.5")
    assert metric.corpus_id == corpus.corpus_id
    assert metric.corpus_version == corpus.corpus_version
    assert metric.corpus_sha256 == corpus.corpus_sha256

    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="schema"):
        metrics.measure_tool_selection_accuracy(
            window_start=START,
            window_end=END,
            corpus=replace(corpus, schema_version="unsupported-schema"),
        )


def test_tool_selection_accuracy_refuses_empty_or_unmatched_corpus() -> None:
    metrics, _ = _metrics()
    empty = build_tool_selection_corpus(
        corpus_id="reviewed-feishu-intents",
        corpus_version="v1",
        labels=(),
    )

    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="empty"):
        metrics.measure_tool_selection_accuracy(
            window_start=START,
            window_end=END,
            corpus=empty,
        )

    with pytest.raises(ValueError, match="supported Feishu analysis task"):
        ToolSelectionLabel(
            intent_sha256="c" * 64,
            request_sha256="5" * 64,
            expected_task_type="invented_tool",
        )

    unmatched = build_tool_selection_corpus(
        corpus_id="reviewed-feishu-intents",
        corpus_version="v1",
        labels=(
            ToolSelectionLabel(
                intent_sha256="d" * 64,
                request_sha256="3" * 64,
                expected_task_type="predict_cycle_life",
            ),
        ),
    )
    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="match"):
        metrics.measure_tool_selection_accuracy(
            window_start=START,
            window_end=END,
            corpus=unmatched,
        )


def test_tool_selection_corpus_loader_verifies_schema_and_sha256(tmp_path) -> None:
    corpus = build_tool_selection_corpus(
        corpus_id="reviewed-feishu-intents",
        corpus_version="v1",
        labels=(
            ToolSelectionLabel(
                intent_sha256="e" * 64,
                request_sha256="4" * 64,
                expected_task_type="compare_operation_scenarios",
            ),
        ),
    )
    path = tmp_path / "corpus.json"
    payload = {
        "schema_version": TOOL_SELECTION_CORPUS_SCHEMA_VERSION,
        "corpus_id": corpus.corpus_id,
        "corpus_version": corpus.corpus_version,
        "corpus_sha256": corpus.corpus_sha256,
        "labels": [
            {
                "intent_sha256": label.intent_sha256,
                "request_sha256": label.request_sha256,
                "expected_task_type": label.expected_task_type,
            }
            for label in corpus.labels
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_tool_selection_corpus(path)

    assert loaded == corpus

    payload["labels"][0]["expected_task_type"] = "predict_cycle_life"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="SHA-256"):
        load_tool_selection_corpus(path)

    payload["corpus_id"] = 42
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FeishuWorkflowMetricEvidenceError, match="identity"):
        load_tool_selection_corpus(path)


def test_tool_selection_corpus_requires_unique_task_independent_intents() -> None:
    with pytest.raises(ValueError, match="intent hashes must be unique"):
        build_tool_selection_corpus(
            corpus_id="reviewed-feishu-intents",
            corpus_version="v1",
            labels=(
                ToolSelectionLabel(
                    intent_sha256="f" * 64,
                    request_sha256="1" * 64,
                    expected_task_type="predict_cycle_life",
                ),
                ToolSelectionLabel(
                    intent_sha256="f" * 64,
                    request_sha256="2" * 64,
                    expected_task_type="predict_soh_trajectory",
                ),
            ),
        )


def test_every_metric_refuses_an_empty_window() -> None:
    metrics, _ = _metrics()

    for measure in (
        metrics.measure_file_to_result_latency,
        metrics.measure_traceability_rate,
        metrics.measure_duplicate_job_rate,
        metrics.measure_recheck_action_success_rate,
    ):
        with pytest.raises(FeishuWorkflowMetricEvidenceError, match=r"no .* evidence"):
            measure(window_start=START, window_end=END)
