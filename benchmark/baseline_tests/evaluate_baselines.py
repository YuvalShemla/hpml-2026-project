import json
import numpy as np
import pandas as pd

FILES = {
    "llama4_maverick": "llama4_maverick_results.json",
    "qwen_2_5_7b": "qwen_2_5_7b_results.json"
}

def valid_plan(text):
    if not isinstance(text, str):
        return False
    return "# Task" in text and "# Agent" in text and "# Tool" in text

def evaluate_model(path):
    with open(path) as f:
        data = json.load(f)

    total = len(data)
    valid = 0
    missing = 0

    input_tokens = []
    output_tokens = []
    response_lengths = []

    for row in data:
        response = row.get("response")

        if not response:
            missing += 1
            continue

        if valid_plan(response):
            valid += 1

        response_lengths.append(len(response))

        if row.get("input_tokens") is not None:
            input_tokens.append(row["input_tokens"])

        if row.get("output_tokens") is not None:
            output_tokens.append(row["output_tokens"])

    metrics = {
        "total_scenarios": total,
        "valid_plan_rate": valid / total if total else 0,
        "missing_responses": missing,
        "avg_input_tokens": np.mean(input_tokens) if input_tokens else None,
        "avg_output_tokens": np.mean(output_tokens) if output_tokens else None,
        "avg_response_length": np.mean(response_lengths) if response_lengths else None
    }

    return metrics

results = []

for model, file in FILES.items():
    metrics = evaluate_model(file)
    metrics["model"] = model
    results.append(metrics)
    print("\n", model)
    print(metrics)

df = pd.DataFrame(results)

print("\nComparison Table")
print(df)

df.to_csv("baseline_metrics.csv", index=False)

print("\nSaved baseline_metrics.csv")