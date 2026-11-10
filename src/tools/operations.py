"""Single source of truth for the tool surface (names, args, error handling).

Used by BOTH the MCP server and the in-process LangChain tools, so the agent
sees identical behaviour whichever transport is configured.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.backends.base import BackendError, FactoryBackend, NotSupportedError

# Tools each backend offers to the *agent*. The MCP server registers all of
# them; unsupported ones answer with a structured error.
AGENT_TOOLS = {
    "simulator": ["get_machine_status", "get_order_details", "calculate_schedule"],
    "openfactory": [
        "get_factory_overview",
        "get_machine_status",
        "get_machine_telemetry",
        "get_active_alarms",
        "get_production_metrics",
        "analyze_machine_downtime",
    ],
}
ALL_TOOLS = [
    "get_factory_overview", "get_machine_status", "get_machine_telemetry", "get_active_alarms",
    "get_production_metrics", "analyze_machine_downtime", "get_order_details", "calculate_schedule",
]


def run_tool(backend: FactoryBackend, name: str, **kwargs: Any) -> Any:
    """Execute a tool; never raises. Errors become ``{"error": ..., ...}`` so the
    LLM can read them and recover (e.g. retry with a valid asset id)."""
    if name not in ALL_TOOLS:
        return {"error": f"Unknown tool '{name}'."}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        return getattr(backend, name)(**kwargs)
    except NotSupportedError as exc:
        return {"error": str(exc), "supported": False}
    except BackendError as exc:
        return {"error": str(exc)}
    except TypeError as exc:  # wrong/missing arguments from the LLM
        return {"error": f"Invalid arguments for {name}: {exc}"}


def run_tool_json(backend: FactoryBackend, name: str, **kwargs: Any) -> str:
    return json.dumps(run_tool(backend, name, **kwargs), default=str)
