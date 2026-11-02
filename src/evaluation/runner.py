import json
import os
import sys
import time
import traceback
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.workflow import _parse_final_json, build_agent_graph

MODES = ["baseline", "tool", "verified", "hybrid"]
TASKS_PATH = Path("data/benchmark_tasks.jsonl")
RESULTS_PATH = Path("experiments/results.jsonl")

SLEEP_BETWEEN_RUNS = 5.0


def _norm(value) -> str:
    return "" if value is None else str(value).strip().lower()


def evaluate_success(task: dict, data: dict) -> tuple[bool, str]:
    gt = task.get("ground_truth") or {}
    answer = _norm(data.get("answer"))

    structured = data.get("structured_data")
    if not isinstance(structured, dict):
        structured = {}

    expected_machine = _norm(gt.get("machine_id"))
    expected_order = _norm(gt.get("order_id"))
    expected_reason = _norm(gt.get("reason"))

    machine_ok = bool(expected_machine) and (
        expected_machine in answer
        or _norm(structured.get("machine_id")) == expected_machine
    )

    task_type = task.get("type")

    if task_type == "lookup":
        if not expected_machine:
            return False, "No machine_id in ground truth."
        return (machine_ok, "OK" if machine_ok else "Wrong or missing machine_id.")

    if task_type == "analysis":
        has_downtime_mention = (
            "downtime" in answer or "down" in answer or "maintenance" in answer or "status" in answer
        )
        has_structured_hypothesis = bool(structured)
        if has_downtime_mention and has_structured_hypothesis:
            return True, "OK"

        checks = {
            "machine_id": (not expected_machine) or machine_ok,
            "reason": (not expected_reason) or expected_reason in answer,
        }
        if not (expected_machine or expected_reason):
            return False, "No machine_id or reason in ground truth."
        failed = [name for name, ok in checks.items() if not ok]
        return (not failed, "OK" if not failed else f"Failed: {', '.join(failed)}.")

    if task_type == "scheduling":
        has_machine = bool(structured.get("machine_id"))
        has_time = bool(structured.get("estimated_hours"))
        has_order = bool(structured.get("order_id")) or expected_order in answer

        if has_machine and has_time and has_order:
            return True, "OK"

        checks = {
            "machine_id": (not expected_machine) or machine_ok,
            "order_id": (not expected_order)
            or expected_order in answer
            or _norm(structured.get("order_id")) == expected_order,
        }
        if not (expected_machine or expected_order):
            return False, "No machine_id or order_id in ground truth."
        failed = [name for name, ok in checks.items() if not ok]
        return (not failed, "OK" if not failed else f"Failed: {', '.join(failed)}.")

    ok = len(answer) > 0
    return ok, "OK" if ok else "Empty answer."


def evaluate_final_message(task: dict, messages: list) -> dict:
    last = messages[-1] if messages else None

    if not isinstance(last, AIMessage) or last.tool_calls:
        return {
            "success": False,
            "valid_schema": False,
            "failure_reason": "Run ended without a final AI answer.",
            "final_answer": None,
        }

    raw = last.content if isinstance(last.content, str) else str(last.content)
    data, reason = _parse_final_json(raw)

    if data is None:
        return {
            "success": False,
            "valid_schema": False,
            "failure_reason": reason,
            "final_answer": raw,
        }

    success, why = evaluate_success(task, data)
    return {
        "success": success,
        "valid_schema": True,
        "failure_reason": None if success else why,
        "final_answer": data,
    }


def main():
    graph = build_agent_graph()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    with open(TASKS_PATH, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    for mode in MODES:
        print(f"Running benchmark for mode: {mode}")

        for task in tasks:
            initial_state = {
                "request": task["request"],
                "messages": [],
                "mode": mode,
                "retry_count": 0,
                "verification_result": {},
                "metrics": {},
            }

            start_total = time.perf_counter()
            error = None
            final_state = initial_state

            try:
                final_state = graph.invoke(initial_state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

            latency_total = time.perf_counter() - start_total

            if error:
                evaluation = {
                    "success": False,
                    "valid_schema": False,
                    "failure_reason": error,
                    "final_answer": None,
                }
            else:
                evaluation = evaluate_final_message(task, final_state["messages"])

            if mode in ("verified", "hybrid") and not error:
                verification_passed = final_state["verification_result"].get("valid", False)
            else:
                verification_passed = None

            result = {
                "task_id": task["id"],
                "task_type": task.get("type"),
                "mode": mode,
                "success": evaluation["success"],
                "valid_schema": evaluation["valid_schema"],
                "verification_passed": verification_passed,
                "latency_total": latency_total,
                "metrics": final_state.get("metrics", {}),
                "escalated": mode == "hybrid" and final_state.get("retry_count", 0) > 0,
                "retry_count": final_state.get("retry_count", 0),
                "failure_reason": evaluation["failure_reason"],
                "final_answer": evaluation["final_answer"],
                "error": error,
            }

            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

            time.sleep(SLEEP_BETWEEN_RUNS)

    print(f"\n✅ Benchmark complete. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
