import json
import time
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.workflow import build_agent_graph

st.set_page_config(page_title="Industrial AI Agent", page_icon="🏭", layout="wide")

st.title("🏭 Industrial AI Agent Benchmark")
st.markdown("*Interactive demo of verified and hybrid AI agent architectures*")

@st.cache_data
def load_tasks():
    tasks_path = Path("data/benchmark_tasks.jsonl")
    if not tasks_path.exists():
        return []
    with open(tasks_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

st.sidebar.header("⚙️ Configuration")

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

tasks = load_tasks()
task_options = {f"{t['id']}: {t['request'][:50]}...": t for t in tasks}

selected_task_label = st.sidebar.selectbox("Select a task:", options=list(task_options.keys()))
selected_task = task_options[selected_task_label]

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

if run_button:
    st.markdown("---")
    st.subheader("🤖 Agent Response")
    
    graph = build_agent_graph()
    
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

results_path = Path("experiments/results.jsonl")
if results_path.exists():
    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]
    
    if results:
        df = pd.DataFrame(results)
        summary = df.groupby("mode").agg({
            "success": "mean",
            "verification_passed": lambda x: x.dropna().mean() if x.notna().any() else None,
            "latency_total": "mean",
            "escalated": "mean"
        }).round(3)
        
        summary.columns = ["Success Rate", "Verification Pass Rate", "Avg Latency (s)", "Escalation Rate"]
        
        st.dataframe(summary, use_container_width=True)
else:
    st.info("No benchmark results found. Run: `python -m src.evaluation.runner`")
