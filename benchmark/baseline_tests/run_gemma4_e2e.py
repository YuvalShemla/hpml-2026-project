#!/usr/bin/env python3
"""Run Gemma 4 26B-A4B end-to-end through plan-execute pipeline.

Runs selected scenarios through the full pipeline:
  question → plan → execute tools via MCP → summarize answer

Captures per-call token usage for dashboard visualization.
"""

import asyncio
import json
import os
import sys
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm.litellm import LiteLLMBackend
from workflow.runner import PlanExecuteRunner

MODEL_ID = "gemini/gemma-4-26b-a4b-it"
SCENARIOS_FILE = os.path.join(os.path.dirname(__file__), "e2e_10_scenarios.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "gemma4_e2e_results.json")


def label_calls(call_log, plan_steps):
    """Label each LLM call: plan, arg_resolve, or summarize."""
    labeled = []
    if not call_log:
        return labeled

    # First call is always PLAN
    labeled.append({"label": "plan", **call_log[0]})

    # Middle calls are ARG_RESOLVE (one per step with {step_N} placeholders)
    for i, entry in enumerate(call_log[1:-1], start=1):
        labeled.append({"label": f"arg_resolve", **entry})

    # Last call is SUMMARIZE (if there are at least 2 calls)
    if len(call_log) >= 2:
        labeled.append({"label": "summarize", **call_log[-1]})

    return labeled


async def run_scenario(runner, llm, scenario):
    """Run a single scenario and return structured result with token tracking."""
    llm.reset_call_log()
    t0 = time.time()
    try:
        result = await runner.run(scenario["prompt"])
        elapsed = time.time() - t0

        plan_data = []
        for s in result.plan.steps:
            plan_data.append({
                "step": s.step_number,
                "task": s.task,
                "agent": s.agent,
                "tool": s.tool,
                "tool_args": s.tool_args,
                "dependencies": s.dependencies,
                "expected_output": s.expected_output,
            })

        history_data = []
        for r in result.history:
            response = r.response
            if response and len(response) > 2000:
                response = response[:2000] + f"\n... [truncated, {len(r.response)} chars total]"
            history_data.append({
                "step": r.step_number,
                "task": r.task,
                "agent": r.agent,
                "tool": r.tool,
                "tool_args": r.tool_args,
                "response": response,
                "error": r.error,
                "success": r.success,
            })

        # Label LLM calls
        llm_calls = label_calls(llm.call_log, len(result.plan.steps))
        total_input = sum(c.get("input_tokens", 0) for c in llm_calls)
        total_output = sum(c.get("output_tokens", 0) for c in llm_calls)

        # Estimate tool description tokens (from the plan call's input)
        plan_call = llm_calls[0] if llm_calls else {}
        plan_input_tokens = plan_call.get("input_tokens", 0)
        # The question itself is short (~20-50 tokens); the rest is tool descriptions + prompt template
        question_tokens_est = len(scenario["prompt"].split()) * 1.3  # rough estimate
        tool_desc_tokens_est = max(0, plan_input_tokens - question_tokens_est - 100)  # 100 for template

        return {
            "id": scenario["id"],
            "prompt": scenario["prompt"],
            "gold_steps": scenario.get("gold_steps", 0),
            "gold_response": scenario.get("gold_response", ""),
            "answer": result.answer,
            "plan": plan_data,
            "history": history_data,
            "plan_steps": len(result.plan.steps),
            "steps_succeeded": sum(1 for r in result.history if r.success),
            "steps_failed": sum(1 for r in result.history if not r.success),
            "all_succeeded": all(r.success for r in result.history),
            "elapsed_seconds": round(elapsed, 1),
            "error": None,
            "llm_calls": llm_calls,
            "token_summary": {
                "total_input": total_input,
                "total_output": total_output,
                "total": total_input + total_output,
                "num_llm_calls": len(llm_calls),
                "tool_desc_tokens_est": round(tool_desc_tokens_est),
                "plan_input_tokens": plan_input_tokens,
            },
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "id": scenario["id"],
            "prompt": scenario["prompt"],
            "gold_steps": scenario.get("gold_steps", 0),
            "gold_response": scenario.get("gold_response", ""),
            "answer": None,
            "plan": [],
            "history": [],
            "plan_steps": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
            "all_succeeded": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": traceback.format_exc(),
            "llm_calls": label_calls(llm.call_log, 0),
            "token_summary": {},
        }


async def main():
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)

    print(f"Running {len(scenarios)} scenarios with {MODEL_ID}")
    print(f"Output: {OUTPUT_FILE}\n")

    llm = LiteLLMBackend(MODEL_ID)
    runner = PlanExecuteRunner(llm=llm)

    results = []
    for i, sc in enumerate(scenarios):
        # Rate limit: wait 45s between requests (16K tok/min limit)
        if i > 0:
            wait = 45
            print(f"  (waiting {wait}s for rate limit...)")
            await asyncio.sleep(wait)

        print(f"[{i+1}/{len(scenarios)}] ID {sc['id']}: {sc['prompt'][:70]}...")

        # Retry with backoff on rate limit errors
        result = None
        for attempt in range(3):
            result = await run_scenario(runner, llm, sc)
            if result["error"] and "429" in result["error"]:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, retry {attempt+1}/3 in {wait}s...")
                await asyncio.sleep(wait)
            else:
                break

        status = "OK" if result["all_succeeded"] else "FAIL"
        tok = result.get("token_summary", {})
        print(f"  {status} — {result['plan_steps']} steps, "
              f"{result['steps_succeeded']}/{result['plan_steps']} ok, "
              f"{result['elapsed_seconds']}s, "
              f"{tok.get('total', 0)} tokens ({tok.get('num_llm_calls', 0)} LLM calls)")

        results.append(result)

        # Save after each scenario
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    total = len(results)
    full_success = sum(1 for r in results if r["all_succeeded"])
    ran = [r for r in results if r.get("plan_steps", 0) > 0]
    total_tokens = sum(r.get("token_summary", {}).get("total", 0) for r in ran)
    total_input = sum(r.get("token_summary", {}).get("total_input", 0) for r in ran)
    total_tool_desc = sum(r.get("token_summary", {}).get("tool_desc_tokens_est", 0) for r in ran)

    # Median tokens per scenario
    scenario_totals = sorted(r.get("token_summary", {}).get("total", 0) for r in ran)
    if scenario_totals:
        mid = len(scenario_totals) // 2
        if len(scenario_totals) % 2 == 0:
            median_tokens = (scenario_totals[mid - 1] + scenario_totals[mid]) / 2
        else:
            median_tokens = scenario_totals[mid]
    else:
        median_tokens = 0

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: Gemma 4 26B-A4B End-to-End Results")
    print(f"{'=' * 60}")
    print(f"  Scenarios ran:     {len(ran)}/{total}")
    print(f"  Full success:      {full_success}/{len(ran)}")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Total input tokens:{total_input:,}")
    print(f"  Median tokens/sc:  {median_tokens:,.0f}")
    print(f"  Tool desc tokens:  {total_tool_desc:,} ({100*total_tool_desc/total_input:.0f}% of input)" if total_input else "")
    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
