"""Tool dispatch: call the factory either in-process or through an MCP server.

    FACTORY_TOOL_TRANSPORT=direct   (default) backend called in-process
    FACTORY_TOOL_TRANSPORT=mcp      agent -> MCP (streamable HTTP) -> backend
    MCP_SERVER_URL=http://localhost:8765/mcp
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Optional

from src.backends import FactoryBackend, get_backend, resolve_backend_name
from src.tools.operations import run_tool_json

DEFAULT_MCP_URL = "http://localhost:8765/mcp"
TRANSPORTS = ("direct", "mcp")


def resolve_transport(transport: Optional[str] = None) -> str:
    transport = (transport or os.environ.get("FACTORY_TOOL_TRANSPORT") or "direct").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError(f"Unknown tool transport '{transport}'. Use one of {list(TRANSPORTS)}.")
    return transport


def _run_coro(coro):
    """Run a coroutine from sync code, even if the caller already has an event loop
    (Streamlit / LangGraph worker threads)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _result_to_json(result: Any) -> str:
    """The factory MCP server returns one JSON text block per call."""
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    if not texts and getattr(result, "structured_content", None) is not None:
        return json.dumps(result.structured_content, default=str)
    return texts[0] if len(texts) == 1 else json.dumps([json.loads(t) for t in texts], default=str)


class ToolDispatcher:
    def __init__(
        self,
        backend_name: Optional[str] = None,
        transport: Optional[str] = None,
        mcp_url: Optional[str] = None,
        backend: Optional[FactoryBackend] = None,
    ):
        self.backend_name = resolve_backend_name(backend_name if backend is None else backend.name)
        self.transport = resolve_transport(transport)
        self.mcp_url = mcp_url or os.environ.get("MCP_SERVER_URL") or DEFAULT_MCP_URL
        self._backend = backend

    @property
    def backend(self) -> FactoryBackend:
        if self._backend is None:
            self._backend = get_backend(self.backend_name)
        return self._backend

    def call(self, name: str, **kwargs: Any) -> str:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if self.transport == "direct":
            return run_tool_json(self.backend, name, **kwargs)
        return self._call_mcp(name, kwargs)

    def _call_mcp(self, name: str, kwargs: dict[str, Any]) -> str:
        from mcp import Client

        async def go() -> str:
            async with Client(self.mcp_url) as client:
                result = await client.call_tool(name, kwargs)
                if getattr(result, "is_error", False):
                    return json.dumps({"error": _result_to_json(result)})
                return _result_to_json(result)

        try:
            return _run_coro(go())
        except Exception as exc:  # noqa: BLE001 - surface to the LLM, don't crash the graph
            return json.dumps({"error": f"MCP call '{name}' failed at {self.mcp_url}: {type(exc).__name__}: {exc}"})
