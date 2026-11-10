# Interview guide: Industrial AI Agent

Everything below matches the actual code. Read it once, then be able to explain it without it.

## 1. The pitch

**30 seconds.** "I built an AI assistant for factory operations that answers questions like 'which asset has a problem and why' by querying plant data through tools. It never controls anything. I connected it to an external simulated plant through an MCP server, and I benchmarked four ways of combining an LLM, tools and verification, measuring success, latency and escalation. The goal was to practice what a Forward Deployed Engineer does: integrate AI into a customer system and prove it works with numbers."

**2 minutes.** Add: the problem (operators need trustworthy, constraint-aware answers, not a chatbot); the architecture (LangGraph agent → tool layer → backend interface → synthetic or external plant, tools via direct call or MCP); the key finding (grounding in data is the first requirement; reliability costs latency); and what you'd do next with a real customer.

## 2. The problem in one paragraph

A production manager asks a question. A raw LLM has no plant data, so it guesses or refuses (the baseline scored 0.00). With tools it can read machine status, orders, alarms and telemetry. But a wrong answer on a factory floor costs money, so the real question is **"when can I trust it, and at what cost?"** That reframes the project from "is the LLM smart" to "how do I design, integrate and measure a system I can trust".

## 3. Code map

```
src/
  agents/workflow.py        LangGraph agent: nodes, routing, prompts, modes
  backends/
    base.py                 FactoryBackend interface + errors + id normalization
    simulator_backend.py    wraps the original synthetic factory (the baseline)
    openfactory_backend.py  REST connector: retries, timeouts, Pydantic validation
    openfactory_models.py   Pydantic models of the Demo Factory payloads
    __init__.py             get_backend(): picks backend from FACTORY_BACKEND
  tools/
    operations.py           ONE definition of tools + error handling (run_tool)
    dispatch.py             ToolDispatcher: direct vs MCP transport
    langchain_tools.py      LangChain tool wrappers for the agent
  mcp_server/               MCP server exposing the same tools (stdio or HTTP)
  evaluation/
    runner.py               benchmark loop + scoring per task type
    scenarios.py            harness-only fault injection for the Demo Factory
    metrics.py              result aggregation by backend and mode
  simulator/factory.py      original synthetic factory
  verification/             Pydantic schemas + capacity business rule (not wired in yet)
  ui/app.py                 Streamlit demo
data/                       synthetic data, task files (simulator + openfactory)
tests/                      unit, integration, live tests
docker-compose.yml          services: benchmark, ui, openfactory-api, MCP server
```

## 4. How a request flows (trace this out loud)

1. User asks: "Which asset has a pressure problem?" (UI or benchmark runner).
2. Runner (benchmark only) calls the **scenario controller**: reset plant, enable a fault, wait for the alarm. This is the harness, **not** the agent.
3. `build_agent_graph(backend, transport)` builds the LangGraph agent with that backend's tools and system prompt.
4. **llm_node** (small model, `gpt-oss-20b` via Groq) decides to call tools, e.g. `get_active_alarms`.
5. **tools node** runs the tool via `ToolDispatcher`: either in-process, or as an MCP client call over HTTP to the MCP server.
6. Either way it ends in `run_tool(backend, name, args)` → the backend → for Open Factory, an HTTP GET to the REST API, parsed with Pydantic.
7. Result (JSON) returns to the LLM, which may call more tools (e.g. `get_machine_telemetry("CR-01")`), then outputs a JSON answer: `answer`, `structured_data`, `confidence`.
8. In `verified`/`hybrid`, **verify_node** checks the JSON schema and confidence ≥ 0.5; on failure it retries (verified, up to 2) or escalates once to the large model (hybrid, `gpt-oss-120b`).
9. Runner scores the answer against ground truth and logs success, latency, tokens, retries, escalation.

## 5. The four modes

| Mode | What it is | Why it exists |
|---|---|---|
| `baseline` | LLM only, no tools | Shows what the LLM can't know. Scored 0.00 |
| `tool` | LLM + tools | Grounding in real data. 0.67 |
| `verified` | tool + output validation + retries | Does checking improve reliability, and at what latency? |
| `hybrid` | small model first, escalate to large on failed verification | Cost/latency vs accuracy trade-off. 1.00 but slowest |

Be precise: verification checks **output format and confidence**, not domain correctness. A capacity validator exists (`verification/business_rules.py`) but isn't wired into the main workflow. Say so before they ask.

## 6. The Open Factory integration (the FDE core)

- **What it is:** an open-source simulated pharma plant (bioreactor, WFI loop, clean steam, purified water, cleanroom, filling line) exposing REST, OPC UA, MQTT and BACnet. I use REST.
- **The mismatch I handled:** it has telemetry, alarms and batches, but no orders and no downtime counters. I didn't fake them. Scheduling tools return "not supported"; `analyze_machine_downtime` reports assets currently degraded with parameter, value and limit.
- **Backend interface:** the agent talks to `FactoryBackend`; swapping plants is `FACTORY_BACKEND=openfactory`. That is the "same agent, different customer system" story.
- **Robustness at the boundary:** timeouts, retries with backoff on 5xx/connection errors, Pydantic validation (a malformed payload raises a clear error rather than reaching the model), flexible asset IDs (`CR-01`, `cr01`, `CR 01` all resolve), errors returned as JSON the LLM can read and recover from.
- **Read-only:** agent tools only GET. Fault injection (POSTs to drift endpoints) lives in `scenarios.py`, used only by the benchmark harness.

## 7. MCP: what it is and why

MCP (Model Context Protocol) is a standard way to expose tools to AI clients. I wrote one server that exposes the factory as read-only tools; the agent can call them over streamable HTTP, or call the same code in-process. Why bother: a customer's tools should be a reusable, standard interface (any MCP client can use them), not code welded into one agent. Be honest: in this project MCP is the integration layer, and in-process is the faster default.

**A real bug I found and can tell as a story:** over MCP, a list with a single element (one active alarm) came back unwrapped as an object, so the result's shape depended on the alarm count. I found it by running the whole chain end to end, fixed it by having the server return the JSON text verbatim, and added a regression test. Good example of why integration testing matters.

## 8. The benchmark

- **Synthetic set (3 tasks):** lookup, analysis, scheduling.
- **Open Factory set (4 tasks):** 3 fault diagnoses (cleanroom pressure, bioreactor temperature, filling-line rejects) + 1 healthy "no alarms" check.
- **Deterministic ground truth:** the harness creates the fault and waits for the alarm before asking. So the right answer is known.
- **Strict scoring** for Open Factory: right asset AND right parameter. (The older `analysis` type is lenient: keywords + any structured data pass. Know that weakness.)
- **Metrics:** success rate, verification pass rate, median and p95 latency, escalation rate, tokens.
- **Validating the benchmark itself:** an LLM-free "oracle" solves all 4 tasks from tool output alone against a live plant. That proves tasks are solvable from the tools and that a wrong answer fails.

## 9. Testing

37 tests: backend behavior against recorded **real** Demo Factory payloads, retries/garbage-payload handling, MCP server in memory and over real HTTP, LangGraph wiring with a scripted LLM, scoring logic, and live tests (auto-skipped without a running plant). No real LLM calls in tests.

## 10. Honest limitations (say these first)

- Small benchmarks: 3 and 4 tasks, one run per mode; results move in coarse steps; p95 is thin.
- Simulated plants, not real production data.
- Verification checks format/confidence, not domain constraints.
- Escalation untested on the synthetic set (never triggered).
- Only REST used; OPC UA/MQTT not integrated.
- LLM output is non-deterministic; I'd need repeated runs for confidence intervals.
- Docker Compose was run by me on my machine for the Open Factory benchmark, but there is no CI.

## 11. Likely questions, with answers

**Why not just give the LLM all the data in the prompt?** Doesn't scale, goes stale, and costs tokens every call. Tools fetch only what's needed, current, and make the data access auditable.

**Why read-only?** Trust is earned. Recommendations with human approval first; write access only after the system proves reliable. It also removes a whole class of safety risk.

**Why LangGraph?** The workflow has branching and loops (tool calls, verify, retry, escalate). A graph makes those transitions explicit and testable instead of buried in prompt logic.

**Why MCP instead of just function calls?** Standard interface the customer or other clients can reuse; separates tool implementation from agent. Cost: an extra network hop, which is why direct stays the default.

**What would you do first at a real customer?** Shadow the users, list the daily questions, pick one high-value read-only use case, connect to their data read-only, and define success with them before building more.

**How do you know it works?** Deterministic scenarios with known answers, strict scoring, an oracle proving solvability, plus tests. And I state what's not proven.

**What surprised you?** That the integration, not the model, was where the hard bugs were: the plant had no orders, payload shapes mattered, and the MCP list unwrapping bug.

**What would you improve next?** Wire domain validation into the workflow, add repeated runs and more scenarios, add OPC UA, measure cost per answer, add CI.

**Latency vs accuracy?** `hybrid` was the most accurate and ~14× slower than baseline on the synthetic set. For an interactive question you'd want `tool`/`verified`; for an overnight planning job, `hybrid`. That's a customer decision, not a technical one.

## 12. Demo script (3 minutes)

1. `docker compose --profile openfactory up -d --build openfactory-api mcp-server-openfactory`
2. Open the UI. Backend `openfactory`, transport `mcp`, mode `tool`.
3. Apply scenario "Cleanroom pressure drop". Ask the OF01 question. Show the answer names CR-01 and the parameter.
4. Show `docker compose logs mcp-server-openfactory`: the tool calls arrived over MCP.
5. Apply "Healthy plant", ask "any active alarms?": it should say none. (Shows it doesn't invent problems.)
6. Switch to `baseline` mode on the same question: "I don't have the data". That is the grounding argument in one click.

Don't claim: production-ready, reliable, or results beyond the tables you actually ran.
