"""Measure audited Feishu workflow metrics from a configured database."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from quanxin_life.metrics.feishu_workflow import (
    FeishuWorkflowMetric,
    FeishuWorkflowMetricEvidenceError,
    SqlAlchemyFeishuWorkflowMetrics,
    WorkflowMetricName,
    load_tool_selection_corpus,
)
from quanxin_life.persistence import (
    DatabaseConfig,
    create_engine_from_config,
    create_session_factory,
)

REPORT_SCHEMA_VERSION = "feishu-workflow-metric-report-v1"
_ENVIRONMENT_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DEFAULT_METRICS = (
    WorkflowMetricName.FILE_TO_RESULT_LATENCY,
    WorkflowMetricName.TRACEABILITY_RATE,
    WorkflowMetricName.DUPLICATE_JOB_RATE,
    WorkflowMetricName.RECHECK_ACTION_SUCCESS_RATE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url-env",
        default="DATABASE_URL",
        help="Environment variable containing the SQLAlchemy database URL.",
    )
    parser.add_argument("--window-start", required=True, type=_timestamp)
    parser.add_argument("--window-end", required=True, type=_timestamp)
    parser.add_argument(
        "--metric",
        action="append",
        choices=tuple(item.value for item in WorkflowMetricName),
        help="Metric to measure; repeat for multiple metrics. Defaults to four SQL metrics.",
    )
    parser.add_argument(
        "--intent-corpus",
        help=(
            "Versioned pre-labelled intent-to-job corpus required for tool "
            "selection accuracy."
        ),
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if _ENVIRONMENT_VARIABLE_NAME.fullmatch(args.database_url_env) is None:
        parser.error("database URL environment variable name is invalid")
    database_url = os.environ.get(args.database_url_env, "").strip()
    if not database_url:
        parser.error("database URL environment variable is missing")
    selected = tuple(
        WorkflowMetricName(value)
        for value in (args.metric or tuple(item.value for item in _DEFAULT_METRICS))
    )
    if len(selected) != len(set(selected)):
        parser.error("--metric values must be unique")
    if (
        WorkflowMetricName.TOOL_SELECTION_ACCURACY in selected
        and not args.intent_corpus
    ):
        parser.error("tool selection accuracy requires --intent-corpus")
    if (
        args.intent_corpus
        and WorkflowMetricName.TOOL_SELECTION_ACCURACY not in selected
    ):
        parser.error("--intent-corpus requires the tool selection accuracy metric")

    try:
        engine = create_engine_from_config(DatabaseConfig(url=database_url))
    except (SQLAlchemyError, ValueError):
        parser.error("workflow metric database configuration is invalid")
    metrics = SqlAlchemyFeishuWorkflowMetrics(create_session_factory(engine))
    measured: list[FeishuWorkflowMetric] = []
    try:
        for metric_name in selected:
            measured.append(
                _measure(
                    metrics,
                    metric_name=metric_name,
                    window_start=args.window_start,
                    window_end=args.window_end,
                    intent_corpus=args.intent_corpus,
                )
            )
    except (FeishuWorkflowMetricEvidenceError, ValueError) as exc:
        parser.error(str(exc))
    except SQLAlchemyError:
        parser.error("workflow metric database query failed")
    finally:
        engine.dispose()
    print(
        json.dumps(
            {
                "metrics": [_metric_payload(metric) for metric in measured],
                "schema_version": REPORT_SCHEMA_VERSION,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _measure(
    metrics: SqlAlchemyFeishuWorkflowMetrics,
    *,
    metric_name: WorkflowMetricName,
    window_start: datetime,
    window_end: datetime,
    intent_corpus: str | None,
) -> FeishuWorkflowMetric:
    keyword = {"window_start": window_start, "window_end": window_end}
    if metric_name is WorkflowMetricName.FILE_TO_RESULT_LATENCY:
        return metrics.measure_file_to_result_latency(**keyword)
    if metric_name is WorkflowMetricName.TRACEABILITY_RATE:
        return metrics.measure_traceability_rate(**keyword)
    if metric_name is WorkflowMetricName.DUPLICATE_JOB_RATE:
        return metrics.measure_duplicate_job_rate(**keyword)
    if metric_name is WorkflowMetricName.RECHECK_ACTION_SUCCESS_RATE:
        return metrics.measure_recheck_action_success_rate(**keyword)
    if intent_corpus is None:  # guarded by argparse
        raise FeishuWorkflowMetricEvidenceError(
            "tool selection accuracy requires a labelled corpus"
        )
    return metrics.measure_tool_selection_accuracy(
        **keyword,
        corpus=load_tool_selection_corpus(intent_corpus),
    )


def _metric_payload(metric: FeishuWorkflowMetric) -> dict[str, Any]:
    payload = asdict(metric)
    payload["metric_name"] = metric.metric_name.value
    payload["window_start"] = metric.window_start.isoformat()
    payload["window_end"] = metric.window_end.isoformat()
    for field_name in ("denominator", "numerator", "value"):
        value = payload[field_name]
        if not isinstance(value, Decimal):  # pragma: no cover - dataclass invariant
            raise TypeError(f"metric {field_name} must be Decimal")
        payload[field_name] = format(value, "f")
    return payload


def _timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timestamp must be an ISO-8601 value with timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
