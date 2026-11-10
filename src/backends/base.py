"""Backend interface: one contract, several factory data sources.

The agent and the MCP server only talk to ``FactoryBackend``. Whether the data
comes from the synthetic simulator or from an external system (Open Factory
Demo Factory) is a deployment choice, not an agent change.

All methods are READ-ONLY and return JSON-serialisable values.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class BackendError(RuntimeError):
    """The backend could not answer (unreachable, timeout, bad payload)."""


class NotSupportedError(BackendError):
    """The operation does not exist for this backend (e.g. orders on a utility plant)."""


def canon_id(value: Optional[str]) -> str:
    """Canonical form of an asset id: 'BR-101', 'br101', 'BR 101' -> 'BR101'."""
    return re.sub(r"[^A-Za-z0-9]", "", value or "").upper()


class FactoryBackend(ABC):
    name: str = "base"

    # --- read-only operations shared by every backend -------------------
    @abstractmethod
    def get_factory_overview(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_machine_status(self, machine_id: Optional[str] = None) -> Any: ...

    @abstractmethod
    def get_machine_telemetry(self, machine_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_active_alarms(self, machine_id: Optional[str] = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_production_metrics(self) -> dict[str, Any]: ...

    @abstractmethod
    def analyze_machine_downtime(self) -> dict[str, Any]: ...

    @abstractmethod
    def health(self) -> dict[str, Any]: ...

    # --- optional: only some backends model orders/scheduling -----------
    def get_order_details(self, order_id: str) -> dict[str, Any]:
        raise NotSupportedError(f"'{self.name}' backend has no production orders.")

    def calculate_schedule(self, order_id: str, machine_id: str) -> dict[str, Any]:
        raise NotSupportedError(f"'{self.name}' backend has no order scheduling.")
