import json
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")  # workflow import requires it; no LLM call is made in tests
os.environ["FACTORY_STATE_PATH"] = str(ROOT / "data" / "factory_state.json")
os.environ.pop("FACTORY_BACKEND", None)
os.environ.pop("FACTORY_TOOL_TRANSPORT", None)

FIXTURES = ROOT / "tests" / "fixtures" / "openfactory"


def load_fixture(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FakeOpenFactory:
    """httpx.MockTransport handler replaying REAL payloads captured from the Demo Factory."""

    def __init__(self, state: str = "healthy"):
        self.state = load_fixture(state)
        self.tag_details = load_fixture("tag_details")
        self.calls: list[str] = []
        self.fail_first = 0  # number of initial 503 responses
        self.garbage = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(503, json={"detail": "starting"})
        if self.garbage and path == "/api/v1/simulation/status":
            return httpx.Response(200, json={"unexpected": True})
        if path in self.state:
            return httpx.Response(200, json=self.state[path])
        if path.startswith("/api/v1/tags/"):
            name = path.rsplit("/", 1)[1]
            if name in self.tag_details:
                return httpx.Response(200, json=self.tag_details[name])
        return httpx.Response(404, json={"detail": "Not found"})


@pytest.fixture
def make_of_backend():
    from src.backends.openfactory_backend import OpenFactoryBackend

    def factory(state: str = "healthy", **kw):
        handler = FakeOpenFactory(state)
        client = httpx.Client(base_url="http://of.test", transport=httpx.MockTransport(handler))
        backend = OpenFactoryBackend(base_url="http://of.test", client=client, backoff=0.0, **kw)
        return backend, handler

    return factory
