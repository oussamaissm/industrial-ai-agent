"""MCP server exposing factory operations from any FactoryBackend.

    python -m src.mcp_server --backend openfactory                 # stdio
    python -m src.mcp_server --backend openfactory --transport streamable-http --port 8765

All tools are read-only.
"""
from __future__ import annotations

from typing import Optional

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from src.backends import FactoryBackend, get_backend
from src.tools.operations import run_tool_json


def create_server(backend: Optional[FactoryBackend] = None) -> "MCPServer":
    backend = backend or get_backend()
    # Tools return the JSON *text* (not a typed object): MCP would otherwise split
    # or unwrap lists, making a 1-element list indistinguishable from an object.
    mcp = MCPServer(
        name=f"industrial-factory-{backend.name}",
        instructions=(
            "Read-only access to a factory. Machines are called 'assets' in some systems "
            "(e.g. CR-01). Never invent data: if a tool returns an error, report it."
        ),
    )

    @mcp.tool()
    def get_factory_overview() -> str:
        """Factory-wide summary: site, areas, asset counts by status, active alarm count, running batches."""
        return run_tool_json(backend, "get_factory_overview")

    @mcp.tool()
    def get_machine_status(machine_id: Optional[str] = None) -> str:
        """Status of one machine/asset, or of ALL when machine_id is omitted."""
        return run_tool_json(backend, "get_machine_status", machine_id=machine_id)

    @mcp.tool()
    def get_machine_telemetry(machine_id: str) -> str:
        """Live readings (value, unit, quality, limits) and active alarms for one machine/asset."""
        return run_tool_json(backend, "get_machine_telemetry", machine_id=machine_id)

    @mcp.tool()
    def get_active_alarms(machine_id: Optional[str] = None) -> str:
        """Active alarms, most severe first. Optionally filter by machine/asset."""
        return run_tool_json(backend, "get_active_alarms", machine_id=machine_id)

    @mcp.tool()
    def get_production_metrics() -> str:
        """Production indicators (batches, throughput, rejects, pass rate or order totals)."""
        return run_tool_json(backend, "get_production_metrics")

    @mcp.tool()
    def analyze_machine_downtime() -> str:
        """Which machines/assets are degraded or down, since when, and why."""
        return run_tool_json(backend, "analyze_machine_downtime")

    @mcp.tool()
    def get_order_details(order_id: str) -> str:
        """Production order details (simulator backend only)."""
        return run_tool_json(backend, "get_order_details", order_id=order_id)

    @mcp.tool()
    def calculate_schedule(order_id: str, machine_id: str) -> str:
        """Validate a proposed order-to-machine schedule (simulator backend only). Read-only."""
        return run_tool_json(backend, "calculate_schedule", order_id=order_id, machine_id=machine_id)

    return mcp
