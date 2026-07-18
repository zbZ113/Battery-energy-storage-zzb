"""Run the audited Quanxin tool registry over an official MCP transport."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.tools import McpHostConfig, McpTransport, run_mcp_host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=tuple(transport.value for transport in McpTransport),
        default=McpTransport.STDIO.value,
    )
    parser.add_argument("--server-name", default="Quanxin Life Tools")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--streamable-http-path", default="/mcp")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved non-secret network configuration and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = McpHostConfig(
        server_name=args.server_name,
        transport=args.transport,
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
    )
    if args.print_config:
        print(
            json.dumps(
                {
                    "host": config.host,
                    "port": config.port,
                    "server_name": config.server_name,
                    "streamable_http_path": config.streamable_http_path,
                    "transport": config.transport.value,
                },
                sort_keys=True,
            )
        )
        return 0

    service = create_available_tool_invocation_service()
    run_mcp_host(service, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
