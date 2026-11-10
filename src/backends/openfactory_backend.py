"""Connector for the Open Factory Demo Factory (REST API, read-only).

    https://github.com/Open-Factory-Initiative/Demo-Factory

The Demo Factory is a *simulated* cGMP-style site (bioreactor, WFI loop, clean
steam, purified water, cleanroom, filling line). It is NOT a discrete-parts
shop: it has telemetry, alarms and batches, but no production orders. The
connector therefore exposes monitoring operations only, and the agent never
calls the simulation-control endpoints (drifts, resets); only the benchmark
harness does, see ``src/evaluation/scenarios.py``.

"Machine" in the agent vocabulary == "asset" in the Demo Factory (e.g. CR-01).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pydantic import TypeAdapter, ValidationError

from src.backends.base import BackendError, FactoryBackend, canon_id
from src.backends.openfactory_models import (
    OFAlarm,
    OFAsset,
    OFBatch,
    OFSnapshot,
    OFTagDefinition,
    OFTagDetail,
    OFTagValue,
)

DEFAULT_BASE_URL = "http://localhost:8000"
ACTIVE_ALARM_STATES = {"active", "acknowledged"}
_SEVERITY_RANK = {"critical": 2, "warning": 1, "info": 0}


class OpenFactoryBackend(FactoryBackend):
    name = "openfactory"

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        retries: int = 2,
        backoff: float = 0.3,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = (base_url or os.environ.get("OPENFACTORY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._retries = retries
        self._backoff = backoff
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._definitions: dict[str, OFTagDefinition] = {}  # static -> cached forever
        self._assets: Optional[list[OFAsset]] = None  # static -> cached

    # ------------------------------------------------------------------ HTTP
    def _get(self, path: str) -> Any:
        last: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.get(path)
                if resp.status_code == 404:
                    raise BackendError(f"Not found: {path}")
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except BackendError:
                raise
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                last = exc
                if attempt < self._retries:
                    time.sleep(self._backoff * (2 ** attempt))
        raise BackendError(f"Open Factory API unreachable or failing at {self.base_url}{path}: {last}")

    def _parse(self, model: Any, payload: Any, what: str) -> Any:
        try:
            return TypeAdapter(model).validate_python(payload)
        except ValidationError as exc:
            raise BackendError(f"Unexpected {what} payload from Open Factory API: {exc.errors()[:2]}") from exc

    # ------------------------------------------------------------- resources
    def _snapshot(self) -> OFSnapshot:
        return self._parse(OFSnapshot, self._get("/api/v1/simulation/status"), "site snapshot")

    def _asset_list(self) -> list[OFAsset]:
        if self._assets is None:
            self._assets = self._parse(list[OFAsset], self._get("/api/v1/assets"), "asset list")
        return self._assets

    def _resolve_asset(self, machine_id: str) -> OFAsset:
        wanted = canon_id(machine_id)
        for asset in self._asset_list():
            if canon_id(asset.asset_id) == wanted:
                return asset
        known = ", ".join(a.asset_id for a in self._asset_list())
        raise BackendError(f"Unknown asset '{machine_id}'. Known assets: {known}")

    def _definition(self, tag_name: str) -> OFTagDefinition:
        if tag_name not in self._definitions:
            detail = self._parse(OFTagDetail, self._get(f"/api/v1/tags/{tag_name}"), "tag detail")
            self._definitions[tag_name] = detail.definition
        return self._definitions[tag_name]

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _active(alarms: list[OFAlarm]) -> list[OFAlarm]:
        return [a for a in alarms if a.state in ACTIVE_ALARM_STATES]

    @staticmethod
    def _alarm_dict(a: OFAlarm) -> dict[str, Any]:
        minutes = None
        if a.created_at:
            created = a.created_at if a.created_at.tzinfo else a.created_at.replace(tzinfo=timezone.utc)
            minutes = round((datetime.now(timezone.utc) - created).total_seconds() / 60, 1)
        return {
            "alarm_id": a.alarm_id,
            "asset_id": a.asset_id,
            "tag_name": a.tag_name,
            "severity": a.severity,
            "state": a.state,
            "message": a.message,
            "active_for_minutes": minutes,
        }

    @staticmethod
    def _asset_status(tags: list[OFTagValue], alarms: list[OFAlarm]) -> str:
        if any(a.severity == "critical" for a in alarms) or any(t.quality == "Bad" for t in tags):
            return "alarm"
        if alarms or any(t.quality in ("Warning", "Stale") for t in tags):
            return "warning"
        return "operational"

    def _status_rows(self, snap: OFSnapshot) -> list[dict[str, Any]]:
        active = self._active(snap.alarms)
        rows = []
        for asset in snap.assets:
            tags = [snap.tags[t] for t in asset.tags if t in snap.tags]
            alarms = [a for a in active if a.asset_id == asset.asset_id]
            rows.append({
                "machine_id": asset.asset_id,
                "name": asset.name,
                "area": asset.area,
                "equipment_type": asset.equipment_type,
                "status": self._asset_status(tags, alarms),
                "active_alarms": [a.alarm_id for a in alarms],
                "degraded_parameters": [t.tag_name for t in tags if t.quality in ("Warning", "Bad", "Stale")],
            })
        return rows

    # ------------------------------------------------------------- operations
    def health(self) -> dict[str, Any]:
        data = self._get("/health")
        return {"status": data.get("status", "unknown"), "backend": self.name, "base_url": self.base_url}

    def get_factory_overview(self) -> dict[str, Any]:
        snap = self._snapshot()
        rows = self._status_rows(snap)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {
            "backend": self.name,
            "site": snap.site.name,
            "site_status": snap.site.status,
            "areas": snap.site.areas,
            "assets": len(rows),
            "assets_by_status": counts,
            "active_alarm_count": len(self._active(snap.alarms)),
            "running_batches": [b.batch_id for b in snap.batches if b.state == "running"],
            "disclaimer": snap.site.disclaimer,
        }

    def get_machine_status(self, machine_id: Optional[str] = None) -> Any:
        rows = self._status_rows(self._snapshot())
        if machine_id is None:
            return rows
        asset = self._resolve_asset(machine_id)
        return next(r for r in rows if r["machine_id"] == asset.asset_id)

    def get_machine_telemetry(self, machine_id: str) -> dict[str, Any]:
        asset = self._resolve_asset(machine_id)
        snap = self._snapshot()
        readings = []
        for tag_name in asset.tags:
            tag = snap.tags.get(tag_name)
            if tag is None:
                continue
            d = self._definition(tag_name)
            limits = {k: v for k, v in {
                "normal_min": d.normal_min, "normal_max": d.normal_max,
                "warning_low": d.warning_low, "warning_high": d.warning_high,
                "alarm_low": d.alarm_low, "alarm_high": d.alarm_high,
            }.items() if v is not None}
            readings.append({
                "name": tag.tag_name,
                "value": tag.value,
                "unit": tag.unit,
                "quality": tag.quality,
                "limits": limits,
                "timestamp": tag.timestamp.isoformat() if tag.timestamp else None,
            })
        alarms = [a for a in self._active(snap.alarms) if a.asset_id == asset.asset_id]
        return {
            "machine_id": asset.asset_id,
            "name": asset.name,
            "readings": readings,
            "out_of_range": [r["name"] for r in readings if r["quality"] != "Good"],
            "active_alarms": [self._alarm_dict(a) for a in alarms],
        }

    def get_active_alarms(self, machine_id: Optional[str] = None) -> list[dict[str, Any]]:
        alarms = self._parse(list[OFAlarm], self._get("/api/v1/alarms"), "alarm list")
        active = self._active(alarms)
        if machine_id:
            asset = self._resolve_asset(machine_id)
            active = [a for a in active if a.asset_id == asset.asset_id]
        active.sort(key=lambda a: -_SEVERITY_RANK.get(a.severity, 0))
        return [self._alarm_dict(a) for a in active]

    def get_production_metrics(self) -> dict[str, Any]:
        snap = self._snapshot()

        def val(name: str) -> Any:
            t = snap.tags.get(name)
            return None if t is None else {"value": t.value, "unit": t.unit, "quality": t.quality}

        return {
            "batches": [
                {"batch_id": b.batch_id, "asset_id": b.asset_id, "lot_number": b.lot_number,
                 "recipe_phase": b.recipe_phase, "state": b.state}
                for b in snap.batches
            ],
            "filling_line": {
                "asset_id": "FL-401",
                "line_state": val("FL401.LineState"),
                "units_per_minute": val("FL401.UnitsPerMinute"),
                "reject_count": val("FL401.RejectCount"),
                "vision_inspection_pass_rate": val("FL401.VisionInspectionPassRate"),
                "label_verification": val("FL401.LabelVerificationStatus"),
            },
            "bioreactor": {"asset_id": "BR-101", "recipe_phase": val("BR101.RecipePhase")},
        }

    def analyze_machine_downtime(self) -> dict[str, Any]:
        """The Demo Factory has no downtime counters. The closest honest analogue
        is: which assets are degraded *right now*, since when, and on which
        parameter, ordered by severity."""
        snap = self._snapshot()
        active = self._active(snap.alarms)
        findings = []
        for row in self._status_rows(snap):
            if row["status"] == "operational":
                continue
            asset = next(a for a in snap.assets if a.asset_id == row["machine_id"])
            params = []
            for name in row["degraded_parameters"]:
                t = snap.tags[name]
                d = self._definition(name)
                params.append({"name": name, "value": t.value, "unit": t.unit, "quality": t.quality,
                               "warning_low": d.warning_low, "warning_high": d.warning_high,
                               "alarm_low": d.alarm_low, "alarm_high": d.alarm_high})
            findings.append({
                "machine_id": row["machine_id"], "name": asset.name, "status": row["status"],
                "alarms": [self._alarm_dict(a) for a in active if a.asset_id == row["machine_id"]],
                "out_of_range_parameters": params,
            })
        findings.sort(key=lambda f: 0 if f["status"] == "alarm" else 1)
        return {
            "note": "Demo Factory has no downtime counters; this lists assets currently in warning/alarm state.",
            "degraded_assets": findings,
            "all_clear": not findings,
        }
