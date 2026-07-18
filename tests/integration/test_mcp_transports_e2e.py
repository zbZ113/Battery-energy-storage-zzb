from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOL_NAMES = {
    "audit_dataset_split",
    "check_operating_condition",
    "validate_battery_data",
}


async def _list_stdio_tools() -> set[str]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["scripts/run_mcp_host.py", "--transport", "stdio"],
        cwd=ROOT,
    )
    async with (
        stdio_client(parameters) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        response = await session.list_tools()
        return {tool.name for tool in response.tools}


def test_official_stdio_client_lists_the_audited_registry() -> None:
    assert asyncio.run(_list_stdio_tools()) == EXPECTED_TOOL_NAMES


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_tcp_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=5)
            raise AssertionError(
                f"MCP HTTP host exited with {process.returncode}: {stderr}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("MCP HTTP host did not accept connections within 30 seconds")


async def _list_streamable_http_tools(port: int) -> set[str]:
    async with (
        streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        response = await session.list_tools()
        return {tool.name for tool in response.tools}


def test_official_streamable_http_client_lists_the_audited_registry() -> None:
    port = _reserve_tcp_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "scripts/run_mcp_host.py",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--streamable-http-path",
            "/mcp",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_tcp_server(process, port)
        assert asyncio.run(_list_streamable_http_tools(port)) == EXPECTED_TOOL_NAMES
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)
