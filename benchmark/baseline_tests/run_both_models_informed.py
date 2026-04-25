#!/usr/bin/env python3
"""Run small models on AssetOpsBench Track 1 (Informed Mode) via OpenRouter.

Same evaluation as run_gemini_track1.py Mode B: full planner prompt with
agent/tool descriptions discovered from MCP servers.

Usage:
    cd /Users/yuvalshemla/Desktop/HPML_PROJECT/AssetOpsBenchGroup20

    # Run all models:
    uv run python benchmark/baseline_tests/run_both_models_informed.py

    # Run a specific model:
    uv run python benchmark/baseline_tests/run_both_models_informed.py --model gemma_3_4b
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import litellm
from workflow.planner import _PLAN_PROMPT, parse_plan
from workflow.executor import Executor
from llm.litellm import LiteLLMBackend

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Free-tier models (no per-request credit cap)
MODELS = {
    "gemma_3_4b": {
        "id": "openrouter/google/gemma-3-4b-it:free",
        "max_tokens": 2048,
        "delay": 6,  # seconds between requests (free model rate limit)
        "params": "4B",
    },
    "nemotron_nano_9b": {
        "id": "openrouter/nvidia/nemotron-nano-9b-v2:free",
        "max_tokens": 2048,
        "delay": 6,
        "params": "9B",
    },
    "llama_3_1_8b": {
        "id": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "max_tokens": 350,  # free-tier per-request credit cap
        "delay": 1,
        "params": "8B",
    },
}

MAX_RETRIES = 4
RETRY_DELAY = 10

VALID_AGENTS = {"IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities"}
VALID_TOOLS = {
    "IoTAgent": {"sites", "assets", "sensors", "history"},
    "FMSRAgent": {"get_failure_modes", "get_failure_mode_sensor_mapping"},
    "TSFMAgent": {"get_ai_tasks", "get_tsfm_models", "run_tsfm_forecasting",
                  "run_tsfm_finetuning", "run_tsad", "run_integrated_tsad"},
    "Utilities": {"json_reader", "current_date_time", "current_time_english"},
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_scenarios():
    path = os.path.join(os.path.dirname(__file__), "llama4_maverick_results.json")
    with open(path) as f:
        data = json.load(f)
    return [{"id": d["id"], "text": d["prompt"]} for d in data]


async def get_agent_descriptions():
    llm = LiteLLMBackend("gemini/gemini-2.5-flash")
    executor = Executor(llm)
    return await executor.get_agent_descriptions()


def strip_thinking(text: str) -> str:
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_model(model_id: str, prompt: str, max_tokens: int = 2048) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                api_key=OPENROUTER_API_KEY,
            )
            raw = response.choices[0].message.content or ""
            usage = response.usage
            return {
                "response": strip_thinking(raw),
                "raw_response": raw,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            }
        except Exception as e:
            last_err = e
            is_rate = "429" in str(e) or "rate" in str(e).lower()
            is_credit = "402" in str(e) or "credits" in str(e).lower()
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (attempt + 1) * (3 if is_rate else 1)
                tag = "rate-limit" if is_rate else ("credit-cap" if is_credit else "error")
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} [{tag}] (wait {wait}s)")
                time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("Unexpected")


# ── Run ─────────────────────────────────────────────────────────────────────


def run_informed(scenarios, agent_descriptions, model_cfg, model_name):
    model_id = model_cfg["id"]
    max_tokens = model_cfg["max_tokens"]
    delay = model_cfg["delay"]

    results = []
    print(f"\n{'=' * 60}")
    print(f"  {model_name} ({model_cfg['params']}) — INFORMED MODE")
    print(f"  Model: {model_id}")
    print(f"  Scenarios: {len(scenarios)} | max_tokens: {max_tokens} | delay: {delay}s")
    print(f"{'=' * 60}\n")

    agents_text = "\n\n".join(
        f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
    )

    errors = 0
    t0 = time.time()
    for i, sc in enumerate(scenarios):
        full_prompt = _PLAN_PROMPT.format(agents=agents_text, question=sc["text"])
        try:
            result = call_model(model_id, full_prompt, max_tokens=max_tokens)
            answer = result["response"]
            plan = parse_plan(answer)

            results.append({
                "id": sc["id"],
                "prompt": sc["text"],
                "response": answer,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
                "plan_steps": len(plan.steps),
                "agents_used": list(set(s.agent for s in plan.steps)),
                "tools_used": list(set(s.tool for s in plan.steps)),
                "agents_per_step": [s.agent for s in plan.steps],
                "tools_per_step": [s.tool for s in plan.steps],
            })
        except Exception as e:
            errors += 1
            err = str(e).split("OpenrouterException - ")[-1][:100] if "Openrouter" in str(e) else str(e)[:100]
            print(f"  ERROR [{errors}] on ID {sc['id']}: {err}")
            results.append({
                "id": sc["id"],
                "prompt": sc["text"],
                "response": None,
                "error": str(e)[:200],
            })

        if (i + 1) % 10 == 0:
            valid = sum(1 for r in results if r.get("plan_steps", 0) > 0)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(scenarios) - i - 1)
            print(f"  [{i+1}/{len(scenarios)}] Valid: {valid}/{i+1} ({100*valid/(i+1):.0f}%) | Errors: {errors} | ETA: {eta:.0f}s")

        time.sleep(delay)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed/len(scenarios):.1f}s/scenario)")
    return results


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate(results, model_name):
    total = len(results)
    valid = parsable = agent_correct = tool_correct = missing = 0
    input_tokens, output_tokens, response_lengths, step_counts = [], [], [], []
    all_agents, all_tools = [], []

    for row in results:
        response = row.get("response")
        if not response:
            missing += 1
            continue

        response_lengths.append(len(response))
        if row.get("input_tokens") is not None:
            input_tokens.append(row["input_tokens"])
        if row.get("output_tokens") is not None:
            output_tokens.append(row["output_tokens"])

        if re.search(r"#Task\d+:", response) and re.search(r"#Agent\d+:", response) and re.search(r"#Tool\d+:", response):
            valid += 1

        if row.get("plan_steps", 0) > 0:
            parsable += 1
            step_counts.append(row["plan_steps"])
            aps = row.get("agents_per_step", row.get("agents_used", []))
            tps = row.get("tools_per_step", row.get("tools_used", []))
            all_agents.extend(aps)
            all_tools.extend(tps)

            if all(a in VALID_AGENTS or a.lower() in ("none", "null", "") for a in aps):
                agent_correct += 1

            tok = True
            for a, t in zip(aps, tps):
                if t.lower() not in ("none", "null", ""):
                    if a in VALID_TOOLS and t not in VALID_TOOLS.get(a, set()):
                        tok = False
                        break
            if tok:
                tool_correct += 1

    avg_in = sum(input_tokens) / len(input_tokens) if input_tokens else 0
    avg_out = sum(output_tokens) / len(output_tokens) if output_tokens else 0
    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    tok_total = sum(input_tokens) + sum(output_tokens)
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0

    print(f"\n{'─' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─' * 60}")
    print(f"  Total scenarios:      {total}")
    print(f"  Missing responses:    {missing}")
    print(f"  Valid plan rate:      {valid}/{total} ({100 * valid / total:.1f}%)")
    print(f"  Parsable plans:       {parsable}/{total} ({100 * parsable / total:.1f}%)")
    if parsable > 0:
        print(f"  Agent correct:        {agent_correct}/{parsable} ({100 * agent_correct / parsable:.1f}%)")
        print(f"  Tool correct:         {tool_correct}/{parsable} ({100 * tool_correct / parsable:.1f}%)")
        print(f"  Avg steps per plan:   {avg_steps:.1f}")
    print(f"  Avg input tokens:     {avg_in:.1f}")
    print(f"  Avg output tokens:    {avg_out:.1f}")
    print(f"  Total tokens:         {tok_total:,}")
    print(f"  Avg response length:  {avg_len:.1f} chars")

    if all_agents:
        ac = Counter(all_agents)
        print(f"\n  Agent distribution ({sum(ac.values())} steps):")
        for a, c in ac.most_common():
            print(f"    {a:30s}: {c:4d} ({100 * c / sum(ac.values()):.1f}%)")

    if all_tools:
        tc = Counter(all_tools)
        print(f"\n  Tool distribution (top 10):")
        for t, c in tc.most_common(10):
            print(f"    {t:35s}: {c:4d}")

    return {
        "total": total, "valid": valid,
        "valid_plan_rate": valid / total if total else 0,
        "parsable": parsable, "agent_correct": agent_correct,
        "tool_correct": tool_correct, "missing": missing,
        "avg_input_tokens": avg_in, "avg_output_tokens": avg_out,
        "total_tokens": tok_total, "avg_response_length": avg_len,
        "avg_steps": avg_steps,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run a single model")
    args = parser.parse_args()

    outdir = os.path.dirname(__file__)
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS

    print("Loading scenarios...")
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    print("\nDiscovering agent capabilities via MCP servers...")
    agent_descs = asyncio.run(get_agent_descriptions())
    print(f"Discovered {len(agent_descs)} agents: {list(agent_descs.keys())}")

    # Quick connectivity test
    print("\n--- Quick connectivity test ---")
    agents_text = "\n\n".join(f"{n}:\n{d}" for n, d in agent_descs.items())
    test_prompt = _PLAN_PROMPT.format(agents=agents_text, question=scenarios[0]["text"])

    working = {}
    for name, cfg in models_to_run.items():
        try:
            result = call_model(cfg["id"], test_prompt, max_tokens=cfg["max_tokens"])
            plan = parse_plan(result["response"])
            print(f"  {name} ({cfg['params']}): OK — {len(plan.steps)} steps, {result['total_tokens']} tok")
            working[name] = cfg
        except Exception as e:
            err = str(e).split("OpenrouterException - ")[-1][:100] if "Openrouter" in str(e) else str(e)[:100]
            print(f"  {name}: FAILED — {err}")

    if not working:
        print("\nNo models available. Exiting.")
        return

    # Run benchmark
    all_metrics = {}
    for name, cfg in working.items():
        results = run_informed(scenarios, agent_descs, cfg, name)

        filename = os.path.join(outdir, f"{name}_informed_results.json")
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {filename}")

        metrics = evaluate(results, name)
        all_metrics[name] = metrics

    # Final comparison
    print(f"\n{'=' * 75}")
    print(f"  COMPARISON: Small Models vs Gemini 2.5 Flash (Informed Mode)")
    print(f"{'=' * 75}")

    col_w = 20
    headers = ["Gemini 2.5 Flash"] + [f"{n} ({MODELS[n]['params']})" for n in working]
    print(f"  {'Metric':<22s}", end="")
    for h in headers:
        print(f" {h:>{col_w}s}", end="")
    print()
    print(f"  {'─' * 22}", end="")
    for _ in headers:
        print(f" {'─' * col_w}", end="")
    print()

    gemini = {"valid": "148/152 (97.4%)", "parsable": "148/152",
              "agent_ok": "93/148 (62.8%)", "tool_ok": "148/148 (100%)",
              "steps": "3.0", "tokens": "762,623"}

    def fmt_rate(m):
        return f"{m['valid']}/{m['total']} ({100*m['valid_plan_rate']:.1f}%)"
    def fmt_parsable(m):
        return f"{m['parsable']}/{m['total']}"
    def fmt_agent(m):
        return f"{m['agent_correct']}/{m['parsable']} ({100*m['agent_correct']/m['parsable']:.1f}%)" if m['parsable'] > 0 else "N/A"
    def fmt_tool(m):
        return f"{m['tool_correct']}/{m['parsable']} ({100*m['tool_correct']/m['parsable']:.1f}%)" if m['parsable'] > 0 else "N/A"

    rows = [
        ("Valid plan rate", gemini["valid"], fmt_rate),
        ("Parsable plans", gemini["parsable"], fmt_parsable),
        ("Agent correct", gemini["agent_ok"], fmt_agent),
        ("Tool correct", gemini["tool_ok"], fmt_tool),
        ("Avg steps/plan", gemini["steps"], lambda m: f"{m['avg_steps']:.1f}"),
        ("Total tokens", gemini["tokens"], lambda m: f"{m['total_tokens']:,}"),
    ]

    for label, gv, fmt in rows:
        print(f"  {label:<22s} {gv:>{col_w}s}", end="")
        for name in working:
            print(f" {fmt(all_metrics[name]):>{col_w}s}", end="")
        print()

    print(f"\n{'=' * 75}")
    print("Done.")


if __name__ == "__main__":
    main()
