"""Pydantic models for the Open Factory Demo Factory REST payloads.

Only the fields the agent needs are declared; unknown fields are ignored so a
newer Demo Factory does not break the connector, but a *missing or mistyped*
required field raises a validation error that surfaces as ``BackendError``
instead of silently feeding bad data to the LLM.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OFAsset(_Model):
    asset_id: str
    name: str
    area: str
    equipment_type: str
    description: str = ""
    tags: list[str] = []


class OFTagValue(_Model):
    tag_name: str
    asset_id: str
    value: Any
    unit: str = ""
    quality: str = "Good"  # Good | Warning | Bad | Stale
    timestamp: Optional[datetime] = None
    description: str = ""


class OFTagDefinition(_Model):
    tag_name: str
    unit: str = ""
    normal_min: Optional[float] = None
    normal_max: Optional[float] = None
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    alarm_low: Optional[float] = None
    alarm_high: Optional[float] = None


class OFTagDetail(_Model):
    value: OFTagValue
    definition: OFTagDefinition


class OFAlarm(_Model):
    alarm_id: str
    asset_id: str
    tag_name: Optional[str] = None
    severity: str  # info | warning | critical
    message: str
    state: str  # inactive | active | acknowledged | cleared
    created_at: Optional[datetime] = None


class OFBatch(_Model):
    batch_id: str
    asset_id: str
    lot_number: str = ""
    recipe_phase: str = ""
    state: str = ""
    started_at: Optional[datetime] = None


class OFSite(_Model):
    name: str
    status: str
    areas: dict[str, list[str]]
    disclaimer: str = ""


class OFSnapshot(_Model):
    """GET /api/v1/simulation/status (atomic view of the whole site)."""
    site: OFSite
    assets: list[OFAsset]
    tags: dict[str, OFTagValue]
    alarms: list[OFAlarm]
    batches: list[OFBatch] = []
    active_alarm_count: int = 0
    active_deviation_count: int = 0
