from __future__ import annotations

import json
import os
import time
from typing import Annotated, Literal, Optional, TypedDict

import psutil
from groq import BadRequestError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.backends import resolve_backend_name
from src.tools.dispatch import ToolDispatcher
from src.tools.langchain_tools import build_tools


# ============================================================
# Configuration
# ============================================================

MODELS = {
    "small": "openai/gpt-oss-20b",
    "large": "openai/gpt-oss-120b",
}
CONFIDENCE_THRESHOLD = 0.5
MAX_VERIFICATION_RETRIES = 2
MAX_LLM_API_ATTEMPTS = 2

if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Set it in your environment before running."
    )


# ============================================================
# Factory tools
# ============================================================
# Tools are built per backend (see src/tools). The default module-level list
# is the original simulator toolset, kept for backwards compatibility.

tools = build_tools(ToolDispatcher("simulator", "direct"))


def make_tools(backend: str | None = None, transport: str | None = None) -> list:
    """Tools for a backend ("simulator" | "openfactory") over a transport
    ("direct" | "mcp"). Defaults come from FACTORY_BACKEND / FACTORY_TOOL_TRANSPORT."""
    return build_tools(ToolDispatcher(backend, transport))


# ============================================================
# Agent state
# ============================================================

class AgentState(TypedDict):
    request: str
    messages: Annotated[list[BaseMessage], add_messages]
    mode: Literal["baseline", "tool", "verified", "hybrid"]
    retry_count: int
    verification_result: dict
    metrics: dict


# ============================================================
# LLM (one cached client per model size)
# ============================================================

_llm_cache: dict[str, ChatGroq] = {}


def get_llm(model_size: str = "small") -> ChatGroq:
    if model_size not in MODELS:
        raise ValueError(f"Unknown model_size '{model_size}'. Use one of {list(MODELS)}.")

    if model_size not in _llm_cache:
        _llm_cache[model_size] = ChatGroq(
            model=MODELS[model_size],
            temperature=0.1,
            max_tokens=1024,
            max_retries=2,
            timeout=60,
            api_key=os.environ["GROQ_API_KEY"],
        )

    return _llm_cache[model_size]


# ============================================================
# System prompts
# ============================================================

BASE_RULES = """
You are an Industrial Operations Assistant.
You help users analyze factory operations, production orders, and scheduling.
"""

SIMULATOR_TOOL_SECTION = """
Available tools:
- get_machine_status: retrieve machine status and downtime. Call with NO arguments to get ALL machines.
- get_order_details: retrieve production order information. Order IDs look like "O0010".
- calculate_schedule: validate a proposed order-machine schedule.

CRITICAL RULES FOR AUTONOMOUS AGENTS:
1. NEVER ask the user for clarification or more information. Solve the task autonomously using the tools.
2. Order IDs look like "O0010". NEVER pass product names (like "Prod_A") to get_order_details.
3. If asked about production issues for a product (e.g., "Why did Prod_A output fall?"), you MUST first call get_machine_status() with NO arguments to check ALL machines for downtime or maintenance.
4. If asked to schedule an order, you MUST first call get_machine_status() to find an operational machine, then call calculate_schedule.
"""

OPENFACTORY_TOOL_SECTION = """
You are connected to an external, read-only industrial site (Open Factory Demo Factory):
a simulated pharmaceutical plant with assets BR-101 (bioreactor), WFI-201 (water-for-injection
loop), CSG-301 (clean steam), PW-101 (purified water), CR-01 (cleanroom) and FL-401 (filling line).
It has telemetry, alarms and batches. It has NO production orders and NO scheduling.

Available tools:
- get_factory_overview: site summary, asset counts by status, active alarm count.
- get_machine_status: status of one asset, or ALL assets when called with NO arguments.
- get_machine_telemetry: live readings with units, quality and alarm limits for ONE asset (e.g. "CR-01").
- get_active_alarms: active alarms, most severe first. An empty list means NO active alarms.
- get_production_metrics: batches, filling-line throughput, rejects, vision pass rate.
- analyze_machine_downtime: assets currently degraded or in alarm, with the out-of-range parameters.

CRITICAL RULES FOR AUTONOMOUS AGENTS:
1. NEVER ask the user for clarification. Solve the task autonomously using the tools.
2. Asset IDs look like "CR-01" or "BR-101". Use only IDs returned by the tools.
3. For "what is wrong / any alarms / any problems" questions, FIRST call get_active_alarms(), then get_machine_telemetry for the affected asset to name the parameter, its value and its limit.
4. If get_active_alarms() returns an empty list and analyze_machine_downtime shows all_clear, say clearly that there are NO active alarms. Do not invent problems.
5. You are read-only. Recommend actions; never claim you changed or acknowledged anything.
6. In structured_data put machine_id (the asset ID) and, when relevant, the affected parameter.
"""

TOOL_SECTIONS = {
    "simulator": SIMULATOR_TOOL_SECTION,
    "openfactory": OPENFACTORY_TOOL_SECTION,
}

BASELINE_NOTE = """
You have NO tools in this mode. Never attempt to call a tool or function.
Answer from the question alone. If you lack the factual data needed, say so
and give a low confidence.
"""

OUTPUT_RULES = """
Never invent machine status, order details, or scheduling results.
When you have enough information, return ONLY a valid JSON object with:
{
  "answer": "your final answer",
  "structured_data": null,
  "confidence": 0.9
}
structured_data may contain relevant fields such as machine_id, order_id,
estimated_hours, and reasoning.
confidence must be a number between 0 and 1.
If information is missing, explain what is missing instead of inventing it.
"""


def build_system_prompt(mode: str, backend: str = "simulator") -> str:
    if mode == "baseline":
        return BASE_RULES + BASELINE_NOTE + OUTPUT_RULES
    return BASE_RULES + TOOL_SECTIONS[backend] + OUTPUT_RULES


# ============================================================
# Metrics helpers
# ============================================================

def _extract_token_usage(response: AIMessage) -> dict:
    usage = (response.response_metadata or {}).get("token_usage", {})
    if not usage:
        usage = response.usage_metadata or {}

    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _record_metrics(state, elapsed, memory_before, memory_after, response) -> dict:
    metrics = dict(state.get("metrics") or {})
    usage = _extract_token_usage(response)

    metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
    metrics["latency_seconds"] = metrics.get("latency_seconds", 0.0) + elapsed
    metrics["memory_delta_mb"] = metrics.get("memory_delta_mb", 0.0) + (
        (memory_after - memory_before) / (1024 * 1024)
    )
    metrics["input_tokens"] = metrics.get("input_tokens", 0) + usage["input_tokens"]
    metrics["output_tokens"] = metrics.get("output_tokens", 0) + usage["output_tokens"]
    metrics["tokens_estimated"] = metrics.get("tokens_estimated", 0) + usage["total_tokens"]
    metrics["cost_usd_estimated"] = metrics.get("cost_usd_estimated", 0.0)

    return metrics


# ============================================================
# LLM node
# ============================================================

def _invoke_with_retry(llm, messages) -> AIMessage:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_LLM_API_ATTEMPTS):
        try:
            return llm.invoke(messages)
        except BadRequestError as exc:
            if "tool_use_failed" not in str(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def call_llm(
    state: AgentState,
    model_size: str = "small",
    agent_tools: Optional[list] = None,
    backend: str = "simulator",
) -> dict:
    llm = get_llm(model_size)
    mode = state["mode"]

    if mode == "baseline":
        llm_to_use = llm
    else:
        llm_to_use = llm.bind_tools(agent_tools if agent_tools is not None else tools, tool_choice="auto")

    history = list(state.get("messages", []))

    request_present = any(
        isinstance(message, HumanMessage) and message.content == state["request"]
        for message in history
    )

    if not request_present:
        history.append(HumanMessage(content=state["request"]))

    messages = [SystemMessage(content=build_system_prompt(mode, backend)), *history]

    process = psutil.Process()
    memory_before = process.memory_info().rss
    start_time = time.perf_counter()

    response = _invoke_with_retry(llm_to_use, messages)

    elapsed = time.perf_counter() - start_time
    memory_after = process.memory_info().rss

    metrics = _record_metrics(
        state=state,
        elapsed=elapsed,
        memory_before=memory_before,
        memory_after=memory_after,
        response=response,
    )

    update: dict = {"messages": [response], "metrics": metrics}

    if not request_present:
        update["messages"] = [HumanMessage(content=state["request"]), response]

    return update


# ============================================================
# Output verification
# ============================================================

def _parse_final_json(content: object) -> tuple[Optional[dict], str]:
    if not isinstance(content, str):
        return None, "Response content is not a string."

    text = content.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc.msg}"

    if not isinstance(data, dict):
        return None, "The final response must be a JSON object."

    if not isinstance(data.get("answer"), str) or not data["answer"].strip():
        return None, "Missing or empty 'answer' field."

    confidence = data.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        return None, "Confidence must be a number between 0 and 1."

    if "structured_data" not in data:
        return None, "Missing 'structured_data' field."

    return data, "OK"


def verify_output(state: AgentState) -> dict:
    messages = state.get("messages", [])
    final_message = messages[-1] if messages else None

    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        return {
            "verification_result": {
                "valid": False,
                "reason": "No final AI response found.",
            }
        }

    data, reason = _parse_final_json(final_message.content)

    if data is None:
        result = {"valid": False, "reason": reason}
    elif data["confidence"] < CONFIDENCE_THRESHOLD:
        result = {"valid": False, "reason": "Confidence below threshold."}
    else:
        result = {"valid": True, "reason": "OK", "parsed_output": data}

    return {"verification_result": result}


# ============================================================
# Routing
# ============================================================

def route_after_llm(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    if state["mode"] in ("verified", "hybrid"):
        return "verify_node"
    return END


def route_after_strong_llm(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "strong_tools"
    return "verify_node"


def route_after_verification(state: AgentState) -> str:
    verification = state.get("verification_result", {})
    if verification.get("valid", False):
        return END

    retries = state.get("retry_count", 0)
    mode = state["mode"]

    if mode == "verified" and retries < MAX_VERIFICATION_RETRIES:
        return "retry_llm"
    if mode == "hybrid" and retries < 1:
        return "escalate"
    return END


def _feedback_message(state: AgentState) -> HumanMessage:
    reason = state.get("verification_result", {}).get("reason", "Output validation failed.")
    return HumanMessage(
        content=f"Your previous final response failed verification: {reason} Please correct it and return only the required JSON."
    )


def retry_llm(state: AgentState) -> dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "messages": [_feedback_message(state)],
    }


def escalate(state: AgentState) -> dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "messages": [_feedback_message(state)],
    }


# ============================================================
# Graph construction
# ============================================================

def build_agent_graph(backend: Optional[str] = None, transport: Optional[str] = None):
    """Build the LangGraph agent.

    backend:   "simulator" (default) or "openfactory"  (env: FACTORY_BACKEND)
    transport: "direct" (default) or "mcp"             (env: FACTORY_TOOL_TRANSPORT)
    """
    backend = resolve_backend_name(backend)
    agent_tools = make_tools(backend, transport)
    workflow = StateGraph(AgentState)

    workflow.add_node("llm_node", lambda state: call_llm(state, "small", agent_tools, backend))
    workflow.add_node("strong_llm_node", lambda state: call_llm(state, "large", agent_tools, backend))
    workflow.add_node("tools", ToolNode(agent_tools))
    workflow.add_node("strong_tools", ToolNode(agent_tools))
    workflow.add_node("verify_node", verify_output)
    workflow.add_node("retry_llm", retry_llm)
    workflow.add_node("escalate", escalate)

    workflow.set_entry_point("llm_node")

    workflow.add_conditional_edges(
        "llm_node",
        route_after_llm,
        {"tools": "tools", "verify_node": "verify_node", END: END},
    )
    workflow.add_edge("tools", "llm_node")

    workflow.add_conditional_edges(
        "verify_node",
        route_after_verification,
        {"retry_llm": "retry_llm", "escalate": "escalate", END: END},
    )
    workflow.add_edge("retry_llm", "llm_node")
    workflow.add_edge("escalate", "strong_llm_node")

    workflow.add_conditional_edges(
        "strong_llm_node",
        route_after_strong_llm,
        {"strong_tools": "strong_tools", "verify_node": "verify_node"},
    )
    workflow.add_edge("strong_tools", "strong_llm_node")

    return workflow.compile()
