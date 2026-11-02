import pandas as pd
from pathlib import Path

def calculate_metrics(results_path: str = "experiments/results.jsonl"):
    if not Path(results_path).exists():
        print("No results found. Run the benchmark first.")
        return
    df = pd.read_json(results_path, lines=True)
    
    def safe_mean(series):
        filtered = series.dropna()
        return filtered.mean() if len(filtered) > 0 else None
    
    metrics = df.groupby("mode").agg(
        task_success_rate=("success", "mean"),
        verification_pass_rate=("verification_passed", lambda x: x.dropna().mean() if x.notna().any() else None),
        median_latency=("latency_total", "median"),
        p95_latency=("latency_total", lambda x: x.quantile(0.95)),
        escalation_rate=("escalated", "mean"),
    ).round(4)
    
    print("=== Benchmark Results ===")
    print(metrics)
    return metrics

if __name__ == "__main__":
    calculate_metrics()
