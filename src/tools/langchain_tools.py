"""LangChain tool wrappers. Same names/descriptions regardless of transport."""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.tools.dispatch import ToolDispatcher
from src.tools.operations import AGENT_TOOLS


def build_tools(dispatcher: ToolDispatcher) -> list:
    """Return the LangChain tools the agent may use for ``dispatcher.backend_name``."""

    @tool
    def get_machine_status(machine_id: Optional[str] = None) -> str:
        """Get operational status for one machine/asset or for all of them.

        Args:
            machine_id: Optional identifier (e.g. "M001" or "CR-01"). Omit to get all.
        """
        return dispatcher.call("get_machine_status", machine_id=machine_id)

    @tool
    def get_order_details(order_id: str) -> str:
        """Retrieve quantity, priority, and deadline for a production order."""
        return dispatcher.call("get_order_details", order_id=order_id)

    @tool
    def calculate_schedule(order_id: str, machine_id: str) -> str:
        """Validate and calculate a production schedule for an order and machine."""
        return dispatcher.call("calculate_schedule", order_id=order_id, machine_id=machine_id)

    @tool
    def get_factory_overview() -> str:
        """Factory-wide summary: site, areas, asset counts by status, active alarm count, running batches."""
        return dispatcher.call("get_factory_overview")

    @tool
    def get_machine_telemetry(machine_id: str) -> str:
        """Live readings (value, unit, quality, alarm limits) and active alarms for ONE asset, e.g. "CR-01"."""
        return dispatcher.call("get_machine_telemetry", machine_id=machine_id)

    @tool
    def get_active_alarms(machine_id: Optional[str] = None) -> str:
        """List active alarms, most severe first. Optionally filter by asset id. Empty list = no active alarms."""
        return dispatcher.call("get_active_alarms", machine_id=machine_id)

    @tool
    def get_production_metrics() -> str:
        """Production indicators: running batches, filling-line throughput, reject count, vision pass rate."""
        return dispatcher.call("get_production_metrics")

    @tool
    def analyze_machine_downtime() -> str:
        """List assets that are degraded or in alarm right now: since when and which parameter is out of range."""
        return dispatcher.call("analyze_machine_downtime")

    registry = {t.name: t for t in (
        get_machine_status, get_order_details, calculate_schedule, get_factory_overview,
        get_machine_telemetry, get_active_alarms, get_production_metrics, analyze_machine_downtime,
    )}
    return [registry[name] for name in AGENT_TOOLS[dispatcher.backend_name]]
