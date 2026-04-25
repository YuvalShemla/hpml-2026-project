#!/usr/bin/env python3
"""Validate all training datasets and produce a quality report.

Usage:
    uv run python benchmark/generate_data/validate_datasets.py
"""

import json
import re
from collections import Counter
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"

VALID_AGENTS = {
    "IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities",
    "WorkOrderAgent", "VibrationAgent", "none", "",
}

VALID_TOOLS = {
    "sites", "assets", "sensors", "history",
    "get_failure_modes", "get_failure_mode_sensor_mapping",
    "get_ai_tasks", "get_tsfm_models", "run_tsfm_forecasting",
    "run_tsfm_finetuning", "run_tsad", "run_integrated_tsad",
    "json_reader", "current_date_time", "current_time_english",
    "get_work_orders", "get_preventive_work_orders", "get_corrective_work_orders",
    "get_events", "get_failure_codes", "get_work_order_distribution",
    "predict_next_work_order", "analyze_alert_to_failure",
    "get_vibration_data", "list_vibration_sensors", "compute_fft_spectrum",
    "compute_envelope_spectrum", "assess_vibration_severity",
    "calculate_bearing_frequencies", "list_known_bearings", "diagnose_vibration",
    "none", "",
}


def validate_plan(text: str) -> list[str]:
    issues = []
    agents = re.findall(r"#Agent\d+:\s*(\S+)", text)
    tools = re.findall(r"#Tool\d+:\s*(\S+)", text)
    for a in agents:
        if a not in VALID_AGENTS:
            issues.append(f"unknown_agent:{a}")
    for t in tools:
        if t.rstrip("()") not in VALID_TOOLS:
            issues.append(f"unknown_tool:{t}")
    for m in re.finditer(r"#Args\d+:\s*(.+)", text):
        arg = m.group(1).strip()
        if arg and arg != "{}":
            try:
                json.loads(arg)
            except json.JSONDecodeError:
                issues.append("bad_json_args")
    return issues


def main():
    print("=" * 70)
    print("  DATASET VALIDATION REPORT")
    print("=" * 70)

    total_examples = 0
    total_issues = 0

    for fname in ["tool_knowledge.jsonl", "planning.jsonl", "execution.jsonl", "combined_all.jsonl"]:
        path = DATASETS_DIR / fname
        if not path.exists():
            print(f"\n  {fname}: NOT FOUND")
            continue

        with open(path) as f:
            data = [json.loads(line) for line in f]

        cats = Counter(d.get("metadata", {}).get("category", "?") for d in data)
        sources = Counter(d.get("metadata", {}).get("source", "?") for d in data)

        print(f"\n{'─'*70}")
        print(f"  {fname}: {len(data)} examples ({path.stat().st_size / 1024:.1f} KB)")
        print(f"  Categories: {dict(cats.most_common())}")
        print(f"  Sources:    {dict(sources.most_common())}")

        # Format check
        bad_format = sum(1 for d in data if "messages" not in d or len(d["messages"]) < 2)
        if bad_format:
            print(f"  BAD FORMAT: {bad_format} entries missing messages")

        # Empty check
        empty = sum(1 for d in data if not d.get("messages", [{}])[1].get("content", "").strip())
        if empty:
            print(f"  EMPTY RESPONSES: {empty}")

        # Duplicate check
        instructions = [d["messages"][0]["content"].strip().lower() for d in data]
        dupes = len(instructions) - len(set(instructions))
        if dupes:
            print(f"  DUPLICATES: {dupes}")

        # Plan validation for planning data
        plans = [d for d in data if "planning" == d.get("metadata", {}).get("category")]
        if plans:
            all_issues = []
            for p in plans:
                issues = validate_plan(p["messages"][1]["content"])
                all_issues.extend(issues)

            valid = sum(1 for p in plans if not validate_plan(p["messages"][1]["content"]))
            print(f"  Plan validity: {valid}/{len(plans)} ({100 * valid / len(plans):.1f}%)")
            if all_issues:
                issue_counts = Counter(all_issues)
                print(f"  Plan issues: {dict(issue_counts.most_common(10))}")
                total_issues += len(all_issues)

            # Step distribution
            step_counts = Counter()
            for p in plans:
                n = len(re.findall(r"#Task\d+:", p["messages"][1]["content"]))
                step_counts[n] += 1
            print(f"  Steps: {dict(sorted(step_counts.items()))}")

        # Length stats
        user_lens = [len(d["messages"][0]["content"]) for d in data]
        asst_lens = [len(d["messages"][1]["content"]) for d in data]
        print(f"  User len: min={min(user_lens)}, avg={sum(user_lens)//len(user_lens)}, max={max(user_lens)}")
        print(f"  Asst len: min={min(asst_lens)}, avg={sum(asst_lens)//len(asst_lens)}, max={max(asst_lens)}")

        if fname != "combined_all.jsonl":
            total_examples += len(data)

    print(f"\n{'='*70}")
    print(f"  TOTAL: {total_examples} examples across 3 datasets")
    print(f"  Issues found: {total_issues}")
    print(f"  Output dir: {DATASETS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
