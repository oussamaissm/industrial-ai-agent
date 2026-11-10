import argparse
import json
import os
import re
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

# Per-backend defaults. The simulator paths are the original ones, so existing
# results stay valid and comparable.
BACKEND_DEFAULTS = {
    "simulator": (Path("data/benchmark_tasks.jsonl"), Path("experiments/results.jsonl")),
    "openfactory": (Path("data/benchmark_tasks_openfactory.jsonl"), Path("experiments/results_openfactory.jsonl")),
}

SLEEP_BETWEEN_RUNS = 5.0


def _norm(value) -> str:
    return "" if value is None else str(value).strip().lower()


def _canon(value) -> str:
    """Alphanumeric-only lowercase: 'CR-01' == 'cr01' == 'CR 01'."""
    return re.sub(r"[^a-z0-9]", "", _norm(value))


NO_ALARM_PATTERN = re.compile(
    r"\b(no|zero|0|none|not any)\b[^.]{0,40}\balarms?\b|\balarms?\b[^.]{0,20}\b(none|0|zero|clear)\b|\ball clear\b",
    re.IGNORECASE,
)


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
        or _canon(expected_machine) in _canon(answer)
        or _canon(structured.get("machine_id")) == _canon(expected_machine)
    )

    task_type = task.get("type")

    # --- Open Factory task types (strict: no lenient keyword shortcuts) ---
    if task_type == "diagnosis":
        haystack = answer + " " + _norm(json.dumps(structured, default=str))
        params = [_norm(p) for p in gt.get("parameter_any", [])]
        parameter_ok = (not params) or any(p in haystack for p in params)
        failed = [n for n, ok in (("machine_id", machine_ok), ("parameter", parameter_ok)) if not ok]
        return (not failed, "OK" if not failed else f"Failed: {', '.join(failed)}.")

    if task_type == "healthy":
        says_none = bool(NO_ALARM_PATTERN.search(data.get("answer") or ""))
        claims_problem = bool(re.search(r"\b(critical|exceed|out of range|excursion)\b", answer)) and not says_none
        ok = says_none and not claims_problem
        return (ok, "OK" if ok else "Did not clearly report 'no active alarms'.")

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


def load_tasks(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_benchmark(
    backend: str,
    modes: list[str],
    tasks_path: Path,
    results_path: Path,
    transport: str | None = None,
    sleep_between: float = SLEEP_BETWEEN_RUNS,
) -> None:
    graph = build_agent_graph(backend, transport)

    controller = None
    if backend == "openfactory":
        from src.evaluation.scenarios import OpenFactoryScenarioController
        controller = OpenFactoryScenarioController()

    results_path.parent.mkdir(exist_ok=True)
    if results_path.exists():
        results_path.unlink()

    tasks = load_tasks(tasks_path)

    try:
        for mode in modes:
            print(f"[{backend}] Running benchmark for mode: {mode}")

            for task in tasks:
                scenario_error = None
                if controller is not None:
                    try:
                        controller.apply(task.get("scenario"))
                    except Exception as exc:  # noqa: BLE001
                        scenario_error = f"ScenarioError: {exc}"
                        traceback.print_exc()

                initial_state = {
                    "request": task["request"],
                    "messages": [],
                    "mode": mode,
                    "retry_count": 0,
                    "verification_result": {},
                    "metrics": {},
                }

                start_total = time.perf_counter()
                error = scenario_error
                final_state = initial_state

                if scenario_error is None:
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
                    "backend": backend,
                    "tool_transport": transport or os.environ.get("FACTORY_TOOL_TRANSPORT", "direct"),
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

                with open(results_path, "a") as f:
                    f.write(json.dumps(result, default=str) + "\n")

                time.sleep(sleep_between)
    finally:
        if controller is not None:
            try:
                controller.teardown()
            except Exception:  # noqa: BLE001
                pass

    print(f"\n✅ Benchmark complete. Results saved to {results_path}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Industrial AI agent benchmark")
    parser.add_argument("--backend", default=os.environ.get("FACTORY_BACKEND", "simulator"),
                        choices=["simulator", "openfactory", "both"])
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--transport", default=None, choices=["direct", "mcp"],
                        help="how tools reach the backend (default: $FACTORY_TOOL_TRANSPORT or direct)")
    parser.add_argument("--tasks", default=None, help="override the tasks file (single backend only)")
    parser.add_argument("--results", default=None, help="override the results file (single backend only)")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_RUNS, help="seconds between runs (rate limits)")
    args = parser.parse_args(argv)

    backends = ["simulator", "openfactory"] if args.backend == "both" else [args.backend]
    if args.backend == "both" and (args.tasks or args.results):
        parser.error("--tasks/--results can only be used with a single backend")

    for backend in backends:
        default_tasks, default_results = BACKEND_DEFAULTS[backend]
        run_benchmark(
            backend=backend,
            modes=args.modes,
            tasks_path=Path(args.tasks) if args.tasks else default_tasks,
            results_path=Path(args.results) if args.results else default_results,
            transport=args.transport,
            sleep_between=args.sleep,
        )


if __name__ == "__main__":
    main()
