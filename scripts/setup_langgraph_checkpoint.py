"""Initialize LangGraph checkpoint tables as an explicit deployment operation."""

from __future__ import annotations

import os

from quanxin_life.infrastructure.langgraph_checkpoint import postgres_checkpoint_saver


def main() -> None:
    dsn = os.environ.get("QUANXIN_LANGGRAPH_POSTGRES_DSN", "")
    if not dsn:
        raise SystemExit("QUANXIN_LANGGRAPH_POSTGRES_DSN is required")
    with postgres_checkpoint_saver(dsn) as saver:
        saver.setup()
    print("LangGraph checkpoint schema is ready.")


if __name__ == "__main__":
    main()
