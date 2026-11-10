"""Backend over the original synthetic simulator (the benchmark baseline).

``get_machine_status``, ``get_order_details`` and ``calculate_schedule`` return
exactly what ``FactorySimulator`` always returned, so earlier results stay
comparable. The other operations are derived from the same data.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Optional

from src.backends.base import FactoryBackend, canon_id
from src.simulator.factory import FactorySimulator


class SimulatorBackend(FactoryBackend):
    name = "simulator"

    def __init__(self, data_path: Optional[str] = None):
        self._data_path = data_path or os.environ.get("FACTORY_STATE_PATH", "data/factory_state.json")

    def _sim(self) -> FactorySimulator:
        # Re-read per call, like the original tools did.
        return FactorySimulator(self._data_path)

    def _resolve(self, machine_id: str) -> Optional[str]:
        wanted = canon_id(machine_id)
        for m in self._sim().state["machines"]:
            if canon_id(m["id"]) == wanted:
                return m["id"]
        return None

    # ---- original operations (unchanged behaviour) ----------------------
    def get_machine_status(self, machine_id: Optional[str] = None) -> Any:
        if machine_id:
            machine_id = self._resolve(machine_id) or machine_id
        return self._sim().get_machine_status(machine_id)

    def get_order_details(self, order_id: str) -> dict[str, Any]:
        return self._sim().get_order_details(order_id)

    def calculate_schedule(self, order_id: str, machine_id: str) -> dict[str, Any]:
        machine_id = self._resolve(machine_id) or machine_id
        order = self._sim().get_order_details(order_id)
        if "error" in order:
            return {"success": False, "reason": f"Order {order_id} not found"}
        return self._sim().calculate_schedule(order_id, machine_id)

    # ---- derived operations ----------------------------------------------
    def get_factory_overview(self) -> dict[str, Any]:
        state = self._sim().state
        by_status: dict[str, int] = defaultdict(int)
        for m in state["machines"]:
            by_status[m["status"]] += 1
        return {
            "backend": self.name,
            "site": "Synthetic factory",
            "machines": len(state["machines"]),
            "machines_by_status": dict(by_status),
            "orders": len(state["orders"]),
        }

    def get_machine_telemetry(self, machine_id: str) -> dict[str, Any]:
        m = self.get_machine_status(machine_id)
        if not isinstance(m, dict) or m.get("error"):
            return {"error": f"Machine {machine_id} not found"}
        return {
            "machine_id": m["id"],
            "status": m["status"],
            "readings": [
                {"name": "capacity_hours", "value": m["capacity_hours"], "unit": "h"},
                {"name": "downtime_minutes", "value": m["downtime_minutes"], "unit": "min"},
            ],
        }

    def get_active_alarms(self, machine_id: Optional[str] = None) -> list[dict[str, Any]]:
        wanted = canon_id(machine_id) if machine_id else None
        alarms = []
        for m in self._sim().state["machines"]:
            if m["status"] == "operational":
                continue
            if wanted and canon_id(m["id"]) != wanted:
                continue
            alarms.append({
                "asset_id": m["id"],
                "severity": "critical" if m["status"] == "down" else "warning",
                "message": f"Machine {m['id']} is {m['status']}",
            })
        return alarms

    def get_production_metrics(self) -> dict[str, Any]:
        orders = self._sim().state["orders"]
        per_product: dict[str, int] = defaultdict(int)
        for o in orders:
            per_product[o["product"]] += o["quantity"]
        return {
            "open_orders": len(orders),
            "total_quantity": sum(o["quantity"] for o in orders),
            "quantity_by_product": dict(per_product),
        }

    def analyze_machine_downtime(self) -> dict[str, Any]:
        machines = sorted(self._sim().state["machines"], key=lambda m: m["downtime_minutes"], reverse=True)
        return {
            "ranking": [{"machine_id": m["id"], "downtime_minutes": m["downtime_minutes"], "status": m["status"]} for m in machines],
            "highest_downtime_machine": machines[0]["id"] if machines else None,
            "total_downtime_minutes": sum(m["downtime_minutes"] for m in machines),
            "not_operational": [m["id"] for m in machines if m["status"] != "operational"],
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": self.name, "data_path": self._data_path}
