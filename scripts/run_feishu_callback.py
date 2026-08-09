"""Run the loopback-only Feishu callback runner for local protocol testing."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

from pydantic import SecretStr
from sqlalchemy.engine import make_url

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.api.feishu import FeishuEventRouteStatus  # noqa: E402
from quanxin_life.api.feishu_runner import (  # noqa: E402
    FeishuCallbackSecurityMode,
    create_feishu_callback_app,
)
from quanxin_life.integrations.feishu import (  # noqa: E402
    FeishuAesCbcDecryptor,
    FeishuEventProcessor,
    FeishuEventReference,
    FeishuWebhookSecrets,
    FeishuWebhookVerifier,
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.persistence import (  # noqa: E402
    Base,
    DatabaseConfig,
    create_engine_from_config,
    create_session_factory,
)


class _LocalReferenceSink:
    """In-memory local sink for sanitized references; no raw payload is stored."""

    def __init__(self) -> None:
        self.events: Queue[FeishuEventReference] = Queue()

    def route_event(
        self, *, event: FeishuEventReference, claim_token: str
    ) -> FeishuEventRouteStatus:
        if not claim_token or not claim_token.strip():
            raise ValueError("claim token must be present")
        self.events.put(event)
        return FeishuEventRouteStatus.ENQUEUED


def build_local_app(
    database_url: str,
    *,
    now_factory: Callable[[], datetime] | None = None,
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
    receipts = SqlAlchemyFeishuReceiptStore(create_session_factory(engine))
    processor = FeishuEventProcessor(
        verifier=FeishuWebhookVerifier(
            FeishuWebhookSecrets(
                verification_token=SecretStr(verification_token),
                encrypt_key=SecretStr(encrypt_key),
            )
        ),
        verification_token=SecretStr(verification_token),
        receipts=receipts,
        decryptor=FeishuAesCbcDecryptor(SecretStr(encrypt_key)),
    )
    sink = _LocalReferenceSink()
    app = create_feishu_callback_app(
        processor,
        router=sink,
        security_mode=FeishuCallbackSecurityMode.LOCAL_ENCRYPTED,
        reviewed_decryptor=FeishuAesCbcDecryptor(SecretStr(encrypt_key)),
        now_factory=now_factory,
    )
    app.state.reference_sink = sink
    app.state.database_url = database_url
    return app


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
        help="Local receipt database URL; production persistence is not implied.",
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
    uvicorn.run(build_local_app(args.database_url), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
