"""Start the local-only in-memory Fake Feishu OpenAPI sandbox."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.integrations.feishu.sandbox import (  # noqa: E402
    create_fake_feishu_sandbox_app,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Fake Feishu Sandbox requires the quanxin-life[api] dependencies"
        ) from exc
    uvicorn.run(
        create_fake_feishu_sandbox_app(),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
