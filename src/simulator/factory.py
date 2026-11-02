"""
Factory OpenEnv: A practice environment for industrial AI agents.

This is a safe, deterministic copy of a real factory MES/SCADA system.
- Same API schema as production tools (MCP protocol)
- Synthetic data with planted failures
- Reproducible via fixed random seed
"""
import json
from pathlib import Path

class FactorySimulator:
    def __init__(self, data_path: str = "data/factory_state.json"):
        with open(data_path, "r") as f:
            self.state = json.load(f)

    def get_machine_status(self, machine_id: str = None) -> dict:
        if machine_id:
            return next((m for m in self.state["machines"] if m["id"] == machine_id), {"error": "Not found"})
        return self.state["machines"]

    def get_order_details(self, order_id: str) -> dict:
        return next((o for o in self.state["orders"] if o["id"] == order_id), {"error": "Not found"})

    def calculate_schedule(self, order_id: str, machine_id: str) -> dict:
        order = self.get_order_details(order_id)
        machine = next((m for m in self.state["machines"] if m["id"] == machine_id), None)
        
        if not machine or machine.get("status") == "down":
            return {"success": False, "reason": "Machine unavailable"}
        
        return {"success": True, "scheduled_machine": machine_id, "estimated_hours": order["quantity"] / 10.0}
