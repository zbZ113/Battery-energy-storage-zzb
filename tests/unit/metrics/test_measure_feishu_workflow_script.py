from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine

from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "measure_feishu_workflow.py"
START = datetime(2026, 8, 14, tzinfo=UTC)
END = START + timedelta(days=1)


def test_metric_script_emits_evidence_fields_without_database_secrets(
    tmp_path: Path,
) -> None:
    database = tmp_path / "metrics.sqlite3"
    database_url = f"sqlite+pysqlite:///{database.as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        session.add(
            FeishuEventReceipt(
                id=uuid4().hex,
                event_id="metric-script-event",
                event_type="im.message.receive_v1",
                payload_sha256="a" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=START + timedelta(minutes=1),
                processed_at=START + timedelta(minutes=1),
                job_id="metric-script-job",
                job_origin="FEISHU",
                job_status="SUCCEEDED",
                job_stage="SUCCEEDED",
                task_type="predict_cycle_life",
                run_id=str(uuid4()),
                message_id="message-safe",
                file_key="file-safe",
                receive_id_type="chat_id",
                event_time=START + timedelta(minutes=1),
                job_attempt_count=1,
                record_batch_id="batch-safe",
                input_file_sha256="b" * 64,
                analysis_result_id="result-safe",
                result_card_message_id="card-safe",
                job_created_at=START + timedelta(minutes=1),
                job_updated_at=START + timedelta(minutes=1, seconds=5),
                job_completed_at=START + timedelta(minutes=1, seconds=5),
            )
        )
    env = {
        **os.environ,
        "QUANXIN_METRICS_DATABASE_URL": database_url,
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database-url-env",
            "QUANXIN_METRICS_DATABASE_URL",
            "--window-start",
            START.isoformat(),
            "--window-end",
            END.isoformat(),
            "--metric",
            "file_to_result_mean_seconds",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "feishu-workflow-metric-report-v1"
    assert len(payload["metrics"]) == 1
    metric = payload["metrics"][0]
    assert metric["metric_name"] == "file_to_result_mean_seconds"
    assert metric["sample_count"] == 1
    assert metric["numerator"] == "5"
    assert metric["denominator"] == "1"
    assert metric["value"] == "5.000000"
    assert metric["evidence_sha256"]
    assert database_url not in completed.stdout


def test_metric_script_refuses_empty_evidence(tmp_path: Path) -> None:
    database = tmp_path / "empty.sqlite3"
    database_url = f"sqlite+pysqlite:///{database.as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    env = {**os.environ, "QUANXIN_METRICS_DATABASE_URL": database_url}

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database-url-env",
            "QUANXIN_METRICS_DATABASE_URL",
            "--window-start",
            START.isoformat(),
            "--window-end",
            END.isoformat(),
            "--metric",
            "file_to_result_mean_seconds",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "no file-to-result latency evidence" in completed.stderr
    assert database_url not in completed.stderr


def test_metric_script_does_not_echo_invalid_database_url() -> None:
    secret_url = "not-a-database-url?password=do-not-print"
    env = {**os.environ, "QUANXIN_METRICS_DATABASE_URL": secret_url}

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database-url-env",
            "QUANXIN_METRICS_DATABASE_URL",
            "--window-start",
            START.isoformat(),
            "--window-end",
            END.isoformat(),
            "--metric",
            "file_to_result_mean_seconds",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert secret_url not in completed.stderr
    assert "database configuration is invalid" in completed.stderr


def test_metric_script_does_not_echo_a_url_mistaken_for_an_env_name() -> None:
    secret_url = "postgresql://user:do-not-print@example.test/metrics"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database-url-env",
            secret_url,
            "--window-start",
            START.isoformat(),
            "--window-end",
            END.isoformat(),
            "--metric",
            "file_to_result_mean_seconds",
        ],
        cwd=PROJECT_ROOT,
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert secret_url not in completed.stderr
    assert "environment variable name is invalid" in completed.stderr
