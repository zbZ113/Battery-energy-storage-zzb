from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mcp_cli_prints_resolved_streamable_http_config_without_starting() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_mcp_host.py",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--streamable-http-path",
            "/quanxin-mcp",
            "--print-config",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "host": "127.0.0.1",
        "port": 8765,
        "server_name": "Quanxin Life Tools",
        "streamable_http_path": "/quanxin-mcp",
        "transport": "streamable-http",
    }
