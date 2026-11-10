import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def main():
    random.seed(42)
    Path("data").mkdir(exist_ok=True)

    machines = [
        {"id": f"M{i:03d}", "capacity_hours": random.randint(8, 16), 
         "status": random.choice(["operational", "maintenance", "down"]), 
         "downtime_minutes": random.randint(0, 500)}
        for i in range(1, 11)
    ]
    orders = [
        {"id": f"O{i:04d}", "product": f"Prod_{random.choice(['A', 'B', 'C'])}", 
         "quantity": random.randint(50, 500), "priority": random.randint(1, 3), 
         "deadline": (datetime.now() + timedelta(days=random.randint(1, 10))).isoformat()}
        for i in range(1, 51)
    ]
    
    with open("data/factory_state.json", "w") as f:
        json.dump({"machines": machines, "orders": orders}, f, indent=2)

    max_downtime_machine = max(machines, key=lambda x: x["downtime_minutes"])
    tasks = [
        {
            "id": "T001", "type": "lookup", "complexity": "low",
            "request": "Which machine is responsible for the largest number of downtime minutes?",
            "expected_tools": ["get_machine_status"],
            "ground_truth": {"machine_id": max_downtime_machine["id"], "downtime": max_downtime_machine["downtime_minutes"]}
        },
        {
            "id": "T002", "type": "analysis", "complexity": "medium",
            "request": "Why did production output fall below the daily target for Prod_A?",
            "expected_tools": ["get_machine_status"],
            "ground_truth": {"reason": "machine_down", "machine_id": "M003"}
        },
        {
            "id": "T003", "type": "scheduling", "complexity": "high",
            "request": "Calculate the production schedule for tomorrow for order O0010, respecting machine capacity.",
            "expected_tools": ["get_order_details", "calculate_schedule"],
            "ground_truth": {"valid": True, "machine_id": "M001"}
        }
    ]
    
    with open("data/benchmark_tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print("✅ Generated data/factory_state.json and data/benchmark_tasks.jsonl")

if __name__ == "__main__":
    main()
