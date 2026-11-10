"""Factory data backends. Select with ``FACTORY_BACKEND=simulator|openfactory``."""
from __future__ import annotations

import os
from typing import Optional

from src.backends.base import BackendError, FactoryBackend, NotSupportedError, canon_id

BACKENDS = ("simulator", "openfactory")
DEFAULT_BACKEND = "simulator"


def resolve_backend_name(name: Optional[str] = None) -> str:
    name = (name or os.environ.get("FACTORY_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Use one of {list(BACKENDS)}.")
    return name


def get_backend(name: Optional[str] = None, **kwargs) -> FactoryBackend:
    name = resolve_backend_name(name)
    if name == "openfactory":
        from src.backends.openfactory_backend import OpenFactoryBackend
        return OpenFactoryBackend(**kwargs)
    from src.backends.simulator_backend import SimulatorBackend
    return SimulatorBackend(**kwargs)


__all__ = [
    "BACKENDS", "BackendError", "FactoryBackend", "NotSupportedError",
    "canon_id", "get_backend", "resolve_backend_name",
]
