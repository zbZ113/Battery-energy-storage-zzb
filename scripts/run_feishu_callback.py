"""Run the loopback-only Feishu callback runner for local protocol testing."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any

from pydantic import SecretStr
from sqlalchemy.engine import make_url

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.api.feishu_runner import (  # noqa: E402
    FeishuCallbackSecurityMode,
    create_feishu_callback_app,
)
from quanxin_life.integrations.feishu import (  # noqa: E402
    FeishuAesCbcDecryptor,
    FeishuEventProcessor,
    FeishuEventReference,
    FeishuInboundEventKind,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.jobs import (  # noqa: E402
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask  # noqa: E402
from quanxin_life.persistence import (  # noqa: E402
    Base,
    DatabaseConfig,
    create_engine_from_config,
    create_session_factory,
)

_FILE_TASKS = frozenset(
    {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    }
)
_SCENARIO_TASKS = frozenset(
    {
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    }
)


class _LocalIdentityQueue:
    """Local dispatch probe that receives only already-persisted job identities."""

    def __init__(self) -> None:
        self.job_ids: Queue[str] = Queue()

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_ids.put(job_id)
        return FeishuJobDispatchReceipt(job_id=job_id, task_id=f"local-{job_id}")


def build_local_app(
    database_url: str,
    *,
    now_factory: Callable[[], datetime] | None = None,
    default_file_task: FeishuAnalysisTask = FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
) -> Any:
    verification_token = os.environ.get("FEISHU_VERIFICATION_TOKEN", "").strip()
    encrypt_key = os.environ.get("FEISHU_ENCRYPT_KEY", "").strip()
    if not verification_token or not encrypt_key:
        raise RuntimeError(
            "FEISHU_VERIFICATION_TOKEN and FEISHU_ENCRYPT_KEY are required"
        )

    _prepare_local_sqlite_parent(database_url)
    engine = create_engine_from_config(DatabaseConfig(url=database_url))
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(session_factory)
    jobs = SqlAlchemyFeishuJobStore(session_factory)
    queue = _LocalIdentityQueue()
    clock = now_factory or (lambda: datetime.now(UTC))
    if default_file_task not in _FILE_TASKS:
        raise ValueError("default_file_task must be a supported file analysis task")
    decryptor = FeishuAesCbcDecryptor(SecretStr(encrypt_key))
    processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(
            FeishuWebhookSecrets(
                verification_token=SecretStr(verification_token),
                encrypt_key=SecretStr(encrypt_key),
            )
        ),
        verification_token=SecretStr(verification_token),
        receipts=receipts,
        decryptor=decryptor,
    )
    router = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: _resolve_analysis_task(
            event,
            default_file_task=default_file_task,
        ),
        clock=clock,
    )
    app = create_feishu_callback_app(
        processor,
        router=router,
        security_mode=FeishuCallbackSecurityMode.LOCAL_ENCRYPTED,
        reviewed_decryptor=decryptor,
        now_factory=now_factory,
    )
    app.state.job_queue = queue
    app.state.job_store = jobs
    app.state.database_url = database_url
    return app


def _resolve_analysis_task(
    event: FeishuEventReference,
    *,
    default_file_task: FeishuAnalysisTask,
) -> FeishuAnalysisTask:
    if event.kind is FeishuInboundEventKind.FILE:
        return default_file_task
    if event.kind is FeishuInboundEventKind.CARD_ACTION:
        raw_task = event.action_value.get("task_type")
        if raw_task is None:
            raise ValueError("scenario card action has no task reference")
        try:
            task = FeishuAnalysisTask(raw_task)
        except (TypeError, ValueError) as exc:
            raise ValueError("scenario card action has an invalid task reference") from exc
        if task not in _SCENARIO_TASKS:
            raise ValueError("card actions may dispatch only scenario analysis tasks")
        return task
    raise ValueError("local durable runner accepts only file and scenario card events")


def _prepare_local_sqlite_parent(database_url: str) -> None:
    parsed = make_url(database_url)
    database = parsed.database
    if (
        parsed.get_backend_name() != "sqlite"
        or database is None
        or database in {"", ":memory:"}
    ):
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--database-url",
        default="sqlite:///./.test-tmp/feishu-callback.sqlite3",
        help="Local receipt and durable job database URL; production use is not implied.",
    )
    parser.add_argument(
        "--default-file-task",
        choices=tuple(task.value for task in sorted(_FILE_TASKS, key=str)),
        default=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
        help="Business task staged for canonical CSV file events.",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("local callback runner must bind to loopback")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Feishu callback runner requires the quanxin-life[api] dependencies"
        ) from exc
    uvicorn.run(
        build_local_app(
            args.database_url,
            default_file_task=FeishuAnalysisTask(args.default_file_task),
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
