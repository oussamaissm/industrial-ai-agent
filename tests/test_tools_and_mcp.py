import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.backends import get_backend
from src.mcp_server.server import create_server
from src.tools.dispatch import ToolDispatcher
from src.tools.langchain_tools import build_tools
from src.tools.operations import AGENT_TOOLS, run_tool

ROOT = Path(__file__).resolve().parents[1]


def test_run_tool_never_raises(make_of_backend):
    b, _ = make_of_backend()
    assert "Unknown asset" in run_tool(b, "get_machine_telemetry", machine_id="nope")["error"]
    assert run_tool(b, "get_order_details", order_id="O1")["supported"] is False
    assert "Unknown tool" in run_tool(b, "drop_tables")["error"]
    assert "Invalid arguments" in run_tool(b, "get_machine_telemetry")["error"]


def test_langchain_tool_sets_per_backend():
    for name, expected in AGENT_TOOLS.items():
        tools = build_tools(ToolDispatcher(name, "direct"))
        assert [t.name for t in tools] == expected


def test_langchain_tool_invocation_simulator():
    tools = {t.name: t for t in build_tools(ToolDispatcher("simulator", "direct"))}
    out = json.loads(tools["get_machine_status"].invoke({"machine_id": "M001"}))
    assert out["downtime_minutes"] == 379
    assert len(json.loads(tools["get_machine_status"].invoke({}))) == 10


def test_mcp_server_in_memory(make_of_backend):
    from mcp import Client

    b, _ = make_of_backend("cr01_pressure_fault")

    async def go():
        async with Client(create_server(b)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool("get_active_alarms", {})
            bad = await c.call_tool("get_machine_telemetry", {"machine_id": "nope"})
            return names, res, bad

    names, res, bad = asyncio.run(go())
    assert {"get_factory_overview", "get_active_alarms", "analyze_machine_downtime"} <= names
    assert "CR01_DP_LOW" in res.content[0].text
    assert "Unknown asset" in bad.content[0].text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_http_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server", "--backend", "simulator", "--transport", "streamable-http", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=5)


def test_agent_tools_over_real_mcp_http(mcp_http_server):
    """agent tool -> MCP client -> streamable HTTP -> MCP server process -> backend"""
    d = ToolDispatcher("simulator", "mcp", mcp_url=mcp_http_server)
    tools = {t.name: t for t in build_tools(d)}
    assert json.loads(tools["get_machine_status"].invoke({"machine_id": "M001"}))["downtime_minutes"] == 379
    assert len(json.loads(tools["get_machine_status"].invoke({}))) == 10
    assert json.loads(tools["get_order_details"].invoke({"order_id": "O0010"}))["id"] == "O0010"


def test_mcp_unreachable_returns_error_not_crash():
    out = json.loads(ToolDispatcher("simulator", "mcp", mcp_url="http://127.0.0.1:9/mcp").call("get_machine_status"))
    assert "MCP call 'get_machine_status' failed" in out["error"]


def test_single_element_list_keeps_its_shape_over_mcp(make_of_backend):
    """Regression: a lone active alarm must still arrive as a list, like over the direct transport."""
    from mcp import Client

    b, _ = make_of_backend("cr01_pressure_fault")  # exactly one active alarm

    async def go():
        async with Client(create_server(b)) as c:
            return (await c.call_tool("get_active_alarms", {})).content[0].text

    over_mcp = json.loads(asyncio.run(go()))
    direct = json.loads(ToolDispatcher(backend=b, transport="direct").call("get_active_alarms"))
    assert isinstance(over_mcp, list) and len(over_mcp) == 1
    assert [a["alarm_id"] for a in over_mcp] == [a["alarm_id"] for a in direct]
