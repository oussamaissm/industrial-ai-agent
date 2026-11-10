import pandas as pd
from pathlib import Path

DEFAULT_RESULT_FILES = [
    "experiments/results.jsonl",              # simulator (baseline backend)
    "experiments/results_openfactory.jsonl",  # Open Factory Demo Factory
]


def load_results(paths=None) -> pd.DataFrame:
    """Load one or more result files into a single frame with a ``backend`` column.
    Files written before the backend flag existed are treated as ``simulator``."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for path in paths or DEFAULT_RESULT_FILES:
        if Path(path).exists():
            frame = pd.read_json(path, lines=True)
            if "backend" not in frame.columns:
                frame["backend"] = "simulator"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calculate_metrics(results_path=None):
    df = load_results(results_path)
    if df.empty:
        print("No results found. Run the benchmark first.")
        return

    metrics = df.groupby(["backend", "mode"]).agg(
        task_success_rate=("success", "mean"),
        verification_pass_rate=("verification_passed", lambda x: x.dropna().mean() if x.notna().any() else None),
        median_latency=("latency_total", "median"),
        p95_latency=("latency_total", lambda x: x.quantile(0.95)),
        escalation_rate=("escalated", "mean"),
        n_tasks=("task_id", "count"),
    ).round(4)

    print("=== Benchmark Results (by backend and mode) ===")
    print(metrics)
    return metrics


if __name__ == "__main__":
    calculate_metrics()
