"""Live tests against a running Open Factory Demo Factory (skipped when unreachable).

    docker compose --profile openfactory up -d openfactory-api
    OPENFACTORY_BASE_URL=http://localhost:8000 pytest tests/test_live_openfactory.py -v

An 'oracle' agent (no LLM) answers each benchmark task using only the tool
outputs. It proves the whole chain (scenario control -> API -> connector ->
tools -> scoring) and that every task is solvable from the tools alone.
"""
import json
import os
from pathlib import Path

import httpx
import pytest

from src.evaluation.runner import evaluate_success, load_tasks
from src.evaluation.scenarios import OpenFactoryScenarioController
from src.tools.dispatch import ToolDispatcher

BASE = os.environ.get("OPENFACTORY_BASE_URL", "http://localhost:8000")


def _reachable() -> bool:
    try:
        return httpx.get(f"{BASE}/health", timeout=2).json().get("status") == "ok"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"Open Factory Demo Factory not reachable at {BASE}")

TASKS = load_tasks(Path("data/benchmark_tasks_openfactory.jsonl"))


@pytest.fixture(scope="module")
def controller():
    c = OpenFactoryScenarioController(BASE)
    yield c
    c.teardown()


def oracle_answer(d: ToolDispatcher) -> dict:
    alarms = json.loads(d.call("get_active_alarms"))
    if not alarms:
        return {"answer": "There are no active alarms on the site.", "structured_data": None}
    top = alarms[0]
    telemetry = json.loads(d.call("get_machine_telemetry", machine_id=top["asset_id"]))
    bad = ", ".join(telemetry["out_of_range"])
    return {
        "answer": f"{top['asset_id']}: {top['message']} Out of range: {bad}.",
        "structured_data": {"machine_id": top["asset_id"]},
    }


@pytest.mark.parametrize("transport", ["direct"])
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_oracle_solves_every_task(task, transport, controller):
    controller.apply(task.get("scenario"))
    answer = oracle_answer(ToolDispatcher("openfactory", transport))
    ok, why = evaluate_success(task, answer)
    assert ok, why


def test_scoring_rejects_a_plausible_but_wrong_answer(controller):
    task = next(t for t in TASKS if t["id"] == "OF01")
    controller.apply(task["scenario"])
    assert not evaluate_success(task, {"answer": "BR-101 temperature is high.", "structured_data": {"machine_id": "BR-101"}})[0]


def test_agent_tools_cannot_change_the_plant(controller):
    controller.apply({})
    d = ToolDispatcher("openfactory", "direct")
    before = d.call("get_factory_overview")
    for name in ("enable_drift", "reset", "ack_alarm"):
        assert "Unknown tool" in json.loads(d.call(name))["error"]
    assert json.loads(before)["active_alarm_count"] == json.loads(d.call("get_factory_overview"))["active_alarm_count"]
