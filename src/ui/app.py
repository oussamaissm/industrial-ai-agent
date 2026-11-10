import hmac
import json
import time
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.workflow import build_agent_graph
from src.backends import BACKENDS, resolve_backend_name
from src.evaluation.metrics import load_results

st.set_page_config(page_title="Industrial AI Agent", page_icon="🏭", layout="wide")

# --- optional guards for a public deployment (both off by default) -------------
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_RUNS = int(os.environ.get("MAX_RUNS_PER_SESSION", "0"))  # 0 = unlimited
if APP_PASSWORD and not st.session_state.get("auth"):
    st.title("🏭 Industrial AI Agent")
    code = st.text_input("Access code", type="password")
    if code and hmac.compare_digest(code, APP_PASSWORD):
        st.session_state["auth"] = True
        st.rerun()
    if code:
        st.error("Wrong access code.")
    st.stop()

st.title("🏭 Industrial AI Agent Benchmark")
st.markdown("*Interactive demo of verified and hybrid AI agent architectures*")

TASK_FILES = {
    "simulator": "data/benchmark_tasks.jsonl",
    "openfactory": "data/benchmark_tasks_openfactory.jsonl",
}


@st.cache_data
def load_tasks(backend: str = "simulator"):
    tasks_path = Path(TASK_FILES[backend])
    if not tasks_path.exists():
        return []
    with open(tasks_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

st.sidebar.header("⚙️ Configuration")

backend = st.sidebar.selectbox(
    "Factory backend:",
    list(BACKENDS),
    index=list(BACKENDS).index(resolve_backend_name()),
    help="simulator = synthetic baseline; openfactory = external Open Factory Demo Factory (REST)",
)
transport = st.sidebar.radio(
    "Tool transport:",
    ["direct", "mcp"],
    help="mcp = tools are called through the MCP server (MCP_SERVER_URL)",
)

if backend == "openfactory":
    from src.evaluation.scenarios import OpenFactoryScenarioController

    st.sidebar.caption("Demo scenario (injects a fault into the external plant)")
    drift_options = {
        "Healthy plant": {},
        "Cleanroom pressure drop": {"enable_drifts": ["CR01_DIFFERENTIAL_PRESSURE_DRIFT"], "wait_for_alarm": "CR01_DP_LOW"},
        "Bioreactor over-temperature": {"enable_drifts": ["BR101_TEMPERATURE_DRIFT"], "wait_for_alarm": "BR101_TEMP_HIGH"},
        "Filling-line rejects": {"enable_drifts": ["FL401_REJECT_RATE_DRIFT"], "wait_for_alarm": "FL401_REJECT_HIGH"},
    }
    scenario_label = st.sidebar.selectbox("Scenario:", list(drift_options))
    if st.sidebar.button("Apply scenario"):
        try:
            with st.spinner("Configuring the Demo Factory..."):
                OpenFactoryScenarioController().apply(drift_options[scenario_label])
            st.sidebar.success(f"Applied: {scenario_label}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not reach the Demo Factory: {exc}")

mode = st.sidebar.radio(
    "Agent Architecture:",
    ["baseline", "tool", "verified", "hybrid"],
    help="Choose which architecture to test"
)

mode_descriptions = {
    "baseline": "🔴 No tools, no verification. LLM answers from memory only.",
    "tool": "🟡 LLM can query factory data through tools.",
    "verified": "🟢 Tool + verification with retries.",
    "hybrid": "🔵 Starts small, escalates to large model when needed."
}

st.sidebar.info(mode_descriptions[mode])

tasks = load_tasks(backend)
task_options = {f"{t['id']}: {t['request'][:50]}...": t for t in tasks}

selected_task_label = st.sidebar.selectbox("Select a task:", options=list(task_options.keys()))
selected_task = task_options[selected_task_label] if task_options else {"request": "", "type": "custom"}

use_custom = st.sidebar.checkbox("Or use custom query", value=False)

if use_custom:
    task_request = st.sidebar.text_area("Enter your query:", value="Which machine has the highest downtime?", height=100)
else:
    task_request = selected_task["request"]

run_button = st.sidebar.button("🚀 Run Agent", type="primary", use_container_width=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Task")
    st.markdown(f"**Request:** {task_request}")
    st.markdown(f"**Type:** {selected_task.get('type', 'custom')}")

with col2:
    st.subheader("📈 Metrics")
    metrics_placeholder = st.empty()

if run_button and MAX_RUNS and st.session_state.get("runs", 0) >= MAX_RUNS:
    st.warning(f"Demo limit reached ({MAX_RUNS} runs per session). Reload the page to start a new session.")
    run_button = False
if run_button:
    st.session_state["runs"] = st.session_state.get("runs", 0) + 1
    st.markdown("---")
    st.subheader("🤖 Agent Response")
    
    graph = build_agent_graph(backend, transport)
    
    initial_state = {
        "request": task_request,
        "messages": [],
        "mode": mode,
        "retry_count": 0,
        "verification_result": {},
        "metrics": {},
    }
    
    with st.spinner(f"Running {mode} agent..."):
        start_time = time.perf_counter()
        
        try:
            final_state = graph.invoke(initial_state)
            elapsed = time.perf_counter() - start_time
            
            final_message = final_state["messages"][-1]
            
            try:
                content = final_message.content
                if isinstance(content, str) and content.strip().startswith("{"):
                    response_data = json.loads(content)
                    answer = response_data.get("answer", "No answer provided")
                    structured_data = response_data.get("structured_data")
                    confidence = response_data.get("confidence", 0)
                else:
                    answer = str(content)
                    structured_data = None
                    confidence = 0
            except json.JSONDecodeError:
                answer = str(final_message.content)
                structured_data = None
                confidence = 0
            
            st.markdown("### 💬 Answer")
            st.markdown(answer)
            
            if structured_data:
                with st.expander("📋 Structured Data"):
                    st.json(structured_data)
            
            if confidence > 0:
                st.metric("Confidence", f"{confidence:.0%}")
            
            metrics = final_state.get("metrics", {})
            
            with metrics_placeholder.container():
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("⏱️ Latency", f"{elapsed:.2f}s")
                with col_m2:
                    st.metric("🔢 LLM Calls", metrics.get("llm_calls", 0))
                with col_m3:
                    tokens = metrics.get("tokens_estimated", 0)
                    st.metric("📊 Tokens", f"{tokens:,}")
                with col_m4:
                    if mode in ["verified", "hybrid"]:
                        verification = final_state.get("verification_result", {})
                        passed = verification.get("valid", False)
                        status = "✅ Passed" if passed else "❌ Failed"
                        st.metric("✓ Verification", status)
                    else:
                        st.metric("✓ Verification", "N/A")
            
            retry_count = final_state.get("retry_count", 0)
            if retry_count > 0:
                if mode == "hybrid":
                    st.warning(f"⚡ Agent escalated to large model after {retry_count} attempt(s)")
                else:
                    st.info(f"🔄 Agent retried {retry_count} time(s)")
            
            st.success("✅ Agent completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.subheader("📊 Historical Benchmark Results")

df = load_results()
if not df.empty:
    summary = df.groupby(["backend", "mode"]).agg(
        success=("success", "mean"),
        verification=("verification_passed", lambda x: x.dropna().mean() if x.notna().any() else None),
        latency=("latency_total", "mean"),
        escalated=("escalated", "mean"),
        tasks=("task_id", "count"),
    ).round(3)
    summary.columns = ["Success Rate", "Verification Pass Rate", "Avg Latency (s)", "Escalation Rate", "Tasks"]
    st.dataframe(summary, use_container_width=True)
else:
    st.info("No benchmark results found. Run: `python -m src.evaluation.runner --backend both`")
