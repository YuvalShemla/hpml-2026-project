#!/usr/bin/env python3
"""Run Gemini 2.5 Flash on AssetOpsBench Track 1 (Planning).

Two evaluation modes:
  Mode A ("blind"):    Raw question only, no tool context. Comparable to existing baselines.
  Mode B ("informed"): Full planner prompt with agent descriptions. Tests actual planning ability.

Usage:
    cd /Users/yuvalshemla/Desktop/HPML_PROJECT/AssetOpsBenchGroup20
    uv run python benchmark/baseline_tests/run_gemini_track1.py
"""

import asyncio
import json
import os
import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on the path for workflow/llm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import litellm
from workflow.planner import _PLAN_PROMPT, parse_plan
from workflow.executor import Executor
from llm.litellm import LiteLLMBackend

MODEL = "gemini/gemini-2.5-flash"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ── Load scenarios ──────────────────────────────────────────────────────────

def load_scenarios():
    """Load scenarios from existing results file (prompts already captured)."""
    path = os.path.join(os.path.dirname(__file__), "llama4_maverick_results.json")
    with open(path) as f:
        data = json.load(f)
    return [{"id": d["id"], "type": "", "text": d["prompt"], "category": ""} for d in data]


# ── Get agent descriptions via MCP discovery ─────────────────────────────────

async def get_agent_descriptions():
    """Discover tools from all MCP servers, returns {agent_name: tool_signatures}."""
    llm = LiteLLMBackend(MODEL)
    executor = Executor(llm)
    descs = await executor.get_agent_descriptions()
    return descs


# ── Run Mode A: Blind (raw question only) ───────────────────────────────────

def run_blind(scenarios):
    """Send raw questions to Gemini without any tool context."""
    results = []
    print(f"\n{'='*60}")
    print(f"  Mode A: BLIND ({len(scenarios)} scenarios)")
    print(f"{'='*60}\n")

    for i, sc in enumerate(scenarios):
        prompt = sc["text"]
        try:
            response = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
                api_key=GEMINI_API_KEY,
            )
            answer = response.choices[0].message.content
            usage = response.usage
            results.append({
                "id": sc["id"],
                "type": sc["type"],
                "prompt": prompt,
                "response": answer,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            })
        except Exception as e:
            print(f"  ERROR on ID {sc['id']}: {e}")
            results.append({
                "id": sc["id"],
                "type": sc["type"],
                "prompt": prompt,
                "response": None,
                "error": str(e),
            })
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(scenarios)}")
        time.sleep(0.1)

    return results


# ── Run Mode B: Informed (with agent descriptions) ──────────────────────────

def run_informed(scenarios, agent_descriptions):
    """Send full planner prompt with agent descriptions to Gemini."""
    results = []
    print(f"\n{'='*60}")
    print(f"  Mode B: INFORMED ({len(scenarios)} scenarios)")
    print(f"{'='*60}\n")

    agents_text = "\n\n".join(
        f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
    )

    for i, sc in enumerate(scenarios):
        full_prompt = _PLAN_PROMPT.format(agents=agents_text, question=sc["text"])
        try:
            response = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=4096,
                api_key=GEMINI_API_KEY,
            )
            answer = response.choices[0].message.content
            usage = response.usage

            # Parse the plan
            plan = parse_plan(answer)
            plan_steps = len(plan.steps)
            agents_used = list(set(s.agent for s in plan.steps))
            tools_used = list(set(s.tool for s in plan.steps))

            results.append({
                "id": sc["id"],
                "type": sc["type"],
                "prompt": sc["text"],
                "response": answer,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "plan_steps": plan_steps,
                "agents_used": agents_used,
                "tools_used": tools_used,
            })
        except Exception as e:
            print(f"  ERROR on ID {sc['id']}: {e}")
            results.append({
                "id": sc["id"],
                "type": sc["type"],
                "prompt": sc["text"],
                "response": None,
                "error": str(e),
            })
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(scenarios)}")
        time.sleep(0.1)

    return results


# ── Evaluation ──────────────────────────────────────────────────────────────

VALID_AGENTS = {"IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities"}
VALID_TOOLS = {
    "IoTAgent": {"sites", "assets", "sensors", "history"},
    "FMSRAgent": {"get_failure_modes", "get_failure_mode_sensor_mapping"},
    "TSFMAgent": {"get_ai_tasks", "get_tsfm_models", "run_tsfm_forecasting",
                  "run_tsfm_finetuning", "run_tsad", "run_integrated_tsad"},
    "Utilities": {"json_reader", "current_date_time", "current_time_english"},
}


def valid_plan(text):
    """Check if response has the expected plan structure."""
    if not isinstance(text, str):
        return False
    return "# Task" in text and "# Agent" in text and "# Tool" in text


def evaluate(results, mode_name):
    """Evaluate results and print metrics."""
    total = len(results)
    valid = 0
    parsable = 0
    agent_correct = 0
    tool_correct = 0
    missing = 0

    input_tokens = []
    output_tokens = []
    response_lengths = []

    type_counts = {}
    type_valid = {}

    for row in results:
        sc_type = row.get("type", "unknown")
        type_counts[sc_type] = type_counts.get(sc_type, 0) + 1

        response = row.get("response")
        if not response:
            missing += 1
            continue

        response_lengths.append(len(response))
        if row.get("input_tokens") is not None:
            input_tokens.append(row["input_tokens"])
        if row.get("output_tokens") is not None:
            output_tokens.append(row["output_tokens"])

        is_valid = valid_plan(response)
        if is_valid:
            valid += 1
            type_valid[sc_type] = type_valid.get(sc_type, 0) + 1

        # For informed mode, check structural parsing
        if "plan_steps" in row and row["plan_steps"] and row["plan_steps"] > 0:
            parsable += 1

            # Check agent correctness
            agents_ok = all(a in VALID_AGENTS for a in row.get("agents_used", []))
            if agents_ok:
                agent_correct += 1

            # Check tool correctness
            tools_ok = True
            for step_agent, step_tool in zip(row.get("agents_used", []), row.get("tools_used", [])):
                if step_tool.lower() not in ("none", "null", ""):
                    if step_agent in VALID_TOOLS and step_tool not in VALID_TOOLS.get(step_agent, set()):
                        tools_ok = False
            if tools_ok:
                tool_correct += 1

    avg_input = sum(input_tokens) / len(input_tokens) if input_tokens else 0
    avg_output = sum(output_tokens) / len(output_tokens) if output_tokens else 0
    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    total_tokens = sum(input_tokens) + sum(output_tokens)

    print(f"\n{'─'*60}")
    print(f"  {mode_name} Evaluation Results")
    print(f"{'─'*60}")
    print(f"  Total scenarios:      {total}")
    print(f"  Missing responses:    {missing}")
    print(f"  Valid plan rate:      {valid}/{total} ({100*valid/total:.1f}%)")
    if parsable > 0:
        print(f"  Parsable plans:       {parsable}/{total} ({100*parsable/total:.1f}%)")
        print(f"  Agent correct:        {agent_correct}/{parsable} ({100*agent_correct/parsable:.1f}%)")
        print(f"  Tool correct:         {tool_correct}/{parsable} ({100*tool_correct/parsable:.1f}%)")
    print(f"  Avg input tokens:     {avg_input:.1f}")
    print(f"  Avg output tokens:    {avg_output:.1f}")
    print(f"  Total tokens:         {total_tokens:,}")
    print(f"  Avg response length:  {avg_len:.1f} chars")

    # Estimate cost (Gemini 2.5 Flash: $0.15/1M input, $0.60/1M output)
    cost = sum(input_tokens) * 0.15 / 1_000_000 + sum(output_tokens) * 0.60 / 1_000_000
    print(f"  Est. cost:            ${cost:.4f}")

    if type_counts:
        print(f"\n  Per-type breakdown:")
        for t in sorted(type_counts):
            v = type_valid.get(t, 0)
            c = type_counts[t]
            print(f"    {t:20s}: {v}/{c} valid ({100*v/c:.0f}%)")

    return {
        "total": total,
        "valid_plan_rate": valid / total if total else 0,
        "parsable": parsable,
        "agent_correct": agent_correct,
        "tool_correct": tool_correct,
        "missing": missing,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "total_tokens": total_tokens,
        "avg_response_length": avg_len,
        "est_cost": cost,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    outdir = os.path.dirname(__file__)

    print("Loading scenarios...")
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    # Mode A: Blind
    print("\nRunning Mode A (blind)...")
    blind_results = run_blind(scenarios)
    blind_path = os.path.join(outdir, "gemini_flash_blind_results.json")
    with open(blind_path, "w") as f:
        json.dump(blind_results, f, indent=2)
    print(f"Saved: {blind_path}")
    blind_metrics = evaluate(blind_results, "Mode A: BLIND")

    # Mode B: Informed (needs MCP discovery)
    print("\nDiscovering agent capabilities for Mode B...")
    agent_descs = asyncio.run(get_agent_descriptions())
    print(f"Discovered {len(agent_descs)} agents: {list(agent_descs.keys())}")

    print("\nRunning Mode B (informed)...")
    informed_results = run_informed(scenarios, agent_descs)
    informed_path = os.path.join(outdir, "gemini_flash_informed_results.json")
    with open(informed_path, "w") as f:
        json.dump(informed_results, f, indent=2)
    print(f"Saved: {informed_path}")
    informed_metrics = evaluate(informed_results, "Mode B: INFORMED")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s} {'Blind':>10s} {'Informed':>10s}")
    print(f"  {'─'*45}")
    print(f"  {'Valid plan rate':<25s} {100*blind_metrics['valid_plan_rate']:>9.1f}% {100*informed_metrics['valid_plan_rate']:>9.1f}%")
    print(f"  {'Avg input tokens':<25s} {blind_metrics['avg_input_tokens']:>10.1f} {informed_metrics['avg_input_tokens']:>10.1f}")
    print(f"  {'Avg output tokens':<25s} {blind_metrics['avg_output_tokens']:>10.1f} {informed_metrics['avg_output_tokens']:>10.1f}")
    print(f"  {'Total tokens':<25s} {blind_metrics['total_tokens']:>10,} {informed_metrics['total_tokens']:>10,}")
    print(f"  {'Est. cost':<25s} ${blind_metrics['est_cost']:>9.4f} ${informed_metrics['est_cost']:>9.4f}")


if __name__ == "__main__":
    main()
