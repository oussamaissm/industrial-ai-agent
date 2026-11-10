"""Scenario control for the Open Factory Demo Factory (BENCHMARK HARNESS ONLY).

The agent's tools are read-only and never touch these endpoints. The harness
uses them to put the plant into a known state before each task, so the ground
truth is deterministic:

    reset -> enable drift(s) -> wait until the expected alarm is raised
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

from src.backends.base import BackendError


class OpenFactoryScenarioController:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0, client: Optional[httpx.Client] = None):
        self.base_url = (base_url or os.environ.get("OPENFACTORY_BASE_URL") or "http://localhost:8000").rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _req(self, method: str, path: str) -> Any:
        try:
            resp = self._client.request(method, path)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"Scenario control failed: {method} {self.base_url}{path}: {exc}") from exc

    def list_drifts(self) -> list[dict]:
        return self._req("GET", "/api/v1/simulation/drifts")

    def reset(self) -> None:
        self._req("POST", "/api/v1/simulation/reset")

    def enable_drift(self, drift_id: str) -> None:
        self._req("POST", f"/api/v1/simulation/drifts/{drift_id}/enable")

    def disable_all(self) -> None:
        self._req("POST", "/api/v1/simulation/drifts/disable-all")

    def active_alarm_ids(self) -> set[str]:
        alarms = self._req("GET", "/api/v1/alarms")
        return {a["alarm_id"] for a in alarms if a["state"] in ("active", "acknowledged")}

    def wait_for_alarm(self, alarm_id: str, timeout: float = 90.0, poll: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if alarm_id in self.active_alarm_ids():
                return
            time.sleep(poll)
        raise BackendError(f"Alarm {alarm_id} was not raised within {timeout}s.")

    def apply(self, scenario: Optional[dict]) -> None:
        """scenario = {"enable_drifts": [...], "wait_for_alarm": "ID"}; None/{} = healthy plant."""
        self.reset()
        scenario = scenario or {}
        for drift_id in scenario.get("enable_drifts", []):
            self.enable_drift(drift_id)
        if scenario.get("wait_for_alarm"):
            self.wait_for_alarm(scenario["wait_for_alarm"])
        else:
            time.sleep(2.0)  # let the healthy plant settle for a couple of ticks

    def teardown(self) -> None:
        self.disable_all()
        self.reset()
