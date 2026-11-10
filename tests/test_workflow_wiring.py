"""LangGraph -> tools -> backend wiring, with a scripted LLM (no network, no API key needed)."""
import json

from langchain_core.messages import AIMessage

from src.agents import workflow
from src.evaluation.runner import evaluate_final_message, load_tasks
from pathlib import Path


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.seen_tools = None

    def bind_tools(self, tools, tool_choice=None):
        self.seen_tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        return self.script.pop(0)


def _run(graph, request, mode="tool"):
    return graph.invoke({"request": request, "messages": [], "mode": mode, "retry_count": 0, "verification_result": {}, "metrics": {}})


def test_openfactory_graph_uses_openfactory_tools_and_prompt(monkeypatch, make_of_backend):
    backend, _ = make_of_backend("cr01_pressure_fault")
    monkeypatch.setattr("src.tools.dispatch.get_backend", lambda name=None, **kw: backend)
    final = {"answer": "CR-01 differential pressure is low (alarm CR01_DP_LOW).", "structured_data": {"machine_id": "CR-01"}, "confidence": 0.9}
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "get_active_alarms", "args": {}, "id": "c1"}]),
        AIMessage(content=json.dumps(final)),
    ])
    monkeypatch.setattr(workflow, "get_llm", lambda size="small": llm)

    state = _run(workflow.build_agent_graph("openfactory", "direct"), "What is wrong?")

    assert llm.seen_tools == ["get_factory_overview", "get_machine_status", "get_machine_telemetry",
                              "get_active_alarms", "get_production_metrics", "analyze_machine_downtime"]
    tool_msg = next(m for m in state["messages"] if m.type == "tool")
    assert "CR01_DP_LOW" in tool_msg.content  # the real alarm reached the LLM through the tool
    task = next(t for t in load_tasks(Path("data/benchmark_tasks_openfactory.jsonl")) if t["id"] == "OF01")
    assert evaluate_final_message(task, state["messages"])["success"] is True


def test_simulator_graph_unchanged_toolset(monkeypatch):
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "get_machine_status", "args": {}, "id": "c1"}]),
        AIMessage(content=json.dumps({"answer": "M001 has the most downtime.", "structured_data": {"machine_id": "M001"}, "confidence": 0.9})),
    ])
    monkeypatch.setattr(workflow, "get_llm", lambda size="small": llm)
    state = _run(workflow.build_agent_graph("simulator", "direct"), "Which machine has the most downtime?")
    assert llm.seen_tools == ["get_machine_status", "get_order_details", "calculate_schedule"]
    task = load_tasks(Path("data/benchmark_tasks.jsonl"))[0]
    assert evaluate_final_message(task, state["messages"])["success"] is True


def test_prompts_are_backend_specific():
    assert "NO production orders" in workflow.build_system_prompt("tool", "openfactory")
    assert "calculate_schedule" in workflow.build_system_prompt("tool", "simulator")
    assert "NO tools" in workflow.build_system_prompt("baseline", "openfactory")
