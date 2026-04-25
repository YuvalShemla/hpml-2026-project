#!/usr/bin/env python3
"""Proper plan quality evaluation for AssetOpsBench.

Three evaluation levels:
  1. structural:  Compare agent/tool sequences against Gemini gold plans (fast, no API)
  2. judge:       LLM-as-judge using Gemini to rate plan correctness (semantic, costs ~$0.10)
  3. all:         Both structural + judge

Usage:
    cd /Users/yuvalshemla/Desktop/HPML_PROJECT/AssetOpsBenchGroup20

    # Structural eval only (fast, free)
    uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
        --candidate benchmark/baseline_tests/llama_3_1_8b_informed_sample_results.json

    # LLM judge eval (uses Gemini, ~$0.10 for 152 scenarios)
    uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
        --candidate benchmark/baseline_tests/llama_3_1_8b_informed_sample_results.json \
        --mode judge

    # Use Gemini 3 Flash as judge
    uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
        --candidate benchmark/baseline_tests/llama_3_1_8b_informed_sample_results.json \
        --mode all --judge-model gemini/gemini-3-flash-preview

    # Run small models informed mode + evaluate in one go
    uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
        --run-model gemma_3_4b --mode all
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import litellm
from workflow.planner import _PLAN_PROMPT, parse_plan

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

BASEDIR = os.path.dirname(__file__)
GOLD_PATH = os.path.join(BASEDIR, "gemini_flash_informed_results.json")

# Valid agents and their tools (ground truth from MCP servers)
VALID_AGENTS = {"IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities"}
VALID_TOOLS = {
    "IoTAgent": {"sites", "assets", "sensors", "history"},
    "FMSRAgent": {"get_failure_modes", "get_failure_mode_sensor_mapping"},
    "TSFMAgent": {"get_ai_tasks", "get_tsfm_models", "run_tsfm_forecasting",
                  "run_tsfm_finetuning", "run_tsad", "run_integrated_tsad"},
    "Utilities": {"json_reader", "current_date_time", "current_time_english"},
}
ALL_TOOLS = set()
for tools in VALID_TOOLS.values():
    ALL_TOOLS.update(tools)

# Models available for --run-model
RUN_MODELS = {
    "gemma_3_4b": {
        "id": "openrouter/google/gemma-3-4b-it:free",
        "max_tokens": 2048, "delay": 6, "params": "4B",
    },
    "nemotron_nano_9b": {
        "id": "openrouter/nvidia/nemotron-nano-9b-v2:free",
        "max_tokens": 2048, "delay": 6, "params": "9B",
    },
    "llama_3_1_8b": {
        "id": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "max_tokens": 350, "delay": 1, "params": "8B",
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_json(path):
    with open(path) as f:
        return json.load(f)


def strip_thinking(text: str) -> str:
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def normalize_tool(tool: str) -> str:
    """Normalize tool names: strip () suffix, lowercase."""
    t = tool.strip().rstrip("()")
    return t if t.lower() not in ("none", "null", "") else "none"


def normalize_agent(agent: str) -> str:
    a = agent.strip()
    return a if a.lower() not in ("none", "null", "") else "none"


def extract_plan_steps(response: str):
    """Parse a response into a list of (agent, tool, args_dict) tuples."""
    if not response:
        return []
    plan = parse_plan(response)
    steps = []
    for s in plan.steps:
        agent = normalize_agent(s.agent)
        tool = normalize_tool(s.tool)
        steps.append({
            "agent": agent,
            "tool": tool,
            "args": s.tool_args,
            "task": s.task,
            "deps": s.dependencies,
            "expected_output": s.expected_output,
        })
    return steps


def extract_actionable_steps(steps):
    """Filter out 'none' agent/tool steps (summary/reasoning steps)."""
    return [s for s in steps if s["agent"] != "none" and s["tool"] != "none"]


# ── Level 1: Structural Evaluation ─────────────────────────────────────────


def compute_set_f1(gold_set, cand_set):
    """Compute precision, recall, F1 between two sets."""
    if not gold_set and not cand_set:
        return 1.0, 1.0, 1.0
    if not gold_set or not cand_set:
        return 0.0, 0.0, 0.0
    tp = len(gold_set & cand_set)
    precision = tp / len(cand_set) if cand_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def compute_sequence_match(gold_seq, cand_seq):
    """Compute ordered sequence similarity (longest common subsequence ratio)."""
    if not gold_seq and not cand_seq:
        return 1.0
    if not gold_seq or not cand_seq:
        return 0.0
    n, m = len(gold_seq), len(cand_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gold_seq[i - 1] == cand_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    return 2 * lcs / (n + m)


def structural_eval_single(gold_response, cand_response):
    """Evaluate a single candidate plan against the gold reference structurally."""
    gold_steps = extract_plan_steps(gold_response)
    cand_steps = extract_plan_steps(cand_response)

    gold_action = extract_actionable_steps(gold_steps)
    cand_action = extract_actionable_steps(cand_steps)

    if not gold_action:
        # Gold plan has no actionable steps — skip
        return None

    result = {}

    # 1. Valid format
    result["valid_format"] = len(cand_steps) > 0

    if not cand_action:
        # Candidate produced no actionable steps
        result.update({
            "agent_set_f1": 0, "tool_set_f1": 0,
            "agent_seq_match": 0, "tool_seq_match": 0,
            "agent_tool_pair_f1": 0,
            "step_count_gold": len(gold_action),
            "step_count_cand": 0,
            "over_decomposition": 0,
            "arg_key_overlap": 0,
            "structural_score": 0,
        })
        return result

    # 2. Agent set F1
    gold_agents = set(s["agent"] for s in gold_action)
    cand_agents = set(s["agent"] for s in cand_action)
    _, _, agent_f1 = compute_set_f1(gold_agents, cand_agents)
    result["agent_set_f1"] = agent_f1

    # 3. Tool set F1
    gold_tools = set(s["tool"] for s in gold_action)
    cand_tools = set(s["tool"] for s in cand_action)
    _, _, tool_f1 = compute_set_f1(gold_tools, cand_tools)
    result["tool_set_f1"] = tool_f1

    # 4. Agent sequence match (LCS-based)
    gold_agent_seq = [s["agent"] for s in gold_action]
    cand_agent_seq = [s["agent"] for s in cand_action]
    result["agent_seq_match"] = compute_sequence_match(gold_agent_seq, cand_agent_seq)

    # 5. Tool sequence match
    gold_tool_seq = [s["tool"] for s in gold_action]
    cand_tool_seq = [s["tool"] for s in cand_action]
    result["tool_seq_match"] = compute_sequence_match(gold_tool_seq, cand_tool_seq)

    # 6. (Agent, Tool) pair F1 — the core metric
    gold_pairs = set((s["agent"], s["tool"]) for s in gold_action)
    cand_pairs = set((s["agent"], s["tool"]) for s in cand_action)
    _, _, pair_f1 = compute_set_f1(gold_pairs, cand_pairs)
    result["agent_tool_pair_f1"] = pair_f1

    # 7. Step counts
    result["step_count_gold"] = len(gold_action)
    result["step_count_cand"] = len(cand_action)
    ratio = len(cand_action) / len(gold_action)
    result["over_decomposition"] = max(0, ratio - 1)  # 0 = same or fewer, >0 = more steps

    # 8. Argument key overlap (for matching agent-tool pairs)
    arg_overlaps = []
    for gs in gold_action:
        for cs in cand_action:
            if gs["agent"] == cs["agent"] and gs["tool"] == cs["tool"]:
                gk = set(gs["args"].keys()) if gs["args"] else set()
                ck = set(cs["args"].keys()) if cs["args"] else set()
                if gk or ck:
                    _, _, kf1 = compute_set_f1(gk, ck)
                    arg_overlaps.append(kf1)
                else:
                    arg_overlaps.append(1.0)  # Both empty — match
                break
    result["arg_key_overlap"] = sum(arg_overlaps) / len(arg_overlaps) if arg_overlaps else 0

    # 9. Composite structural score (weighted)
    result["structural_score"] = (
        0.30 * pair_f1 +
        0.20 * tool_f1 +
        0.15 * agent_f1 +
        0.15 * result["tool_seq_match"] +
        0.10 * result["arg_key_overlap"] +
        0.10 * max(0, 1 - result["over_decomposition"])  # Penalize over-decomposition
    )

    return result


def run_structural_eval(gold_data, cand_data):
    """Run structural evaluation across all scenarios."""
    # Build lookup by ID
    gold_by_id = {d["id"]: d for d in gold_data}
    cand_by_id = {d["id"]: d for d in cand_data}

    common_ids = sorted(set(gold_by_id) & set(cand_by_id))
    print(f"\n  Structural eval: {len(common_ids)} common scenarios")

    results = []
    skipped = 0
    no_response = 0

    for sid in common_ids:
        gold_resp = gold_by_id[sid].get("response", "")
        cand_resp = cand_by_id[sid].get("response", "")

        if not cand_resp:
            no_response += 1
            continue

        if not gold_resp:
            skipped += 1
            continue

        r = structural_eval_single(gold_resp, cand_resp)
        if r is None:
            skipped += 1
            continue

        r["id"] = sid
        r["prompt"] = gold_by_id[sid].get("prompt", "")
        results.append(r)

    print(f"  Evaluated: {len(results)} | Skipped (no gold): {skipped} | No response: {no_response}")

    if not results:
        print("  No evaluable scenarios!")
        return results, {}

    # Aggregate metrics
    metrics = {}
    for key in ["valid_format", "agent_set_f1", "tool_set_f1", "agent_seq_match",
                 "tool_seq_match", "agent_tool_pair_f1", "arg_key_overlap",
                 "structural_score", "over_decomposition"]:
        vals = [r[key] for r in results if key in r]
        if key == "valid_format":
            metrics[key] = sum(vals) / len(vals) if vals else 0
        else:
            metrics[key] = sum(vals) / len(vals) if vals else 0

    metrics["avg_gold_steps"] = sum(r["step_count_gold"] for r in results) / len(results)
    metrics["avg_cand_steps"] = sum(r["step_count_cand"] for r in results) / len(results)
    metrics["evaluated"] = len(results)

    print(f"\n{'─' * 65}")
    print(f"  STRUCTURAL EVALUATION RESULTS")
    print(f"{'─' * 65}")
    print(f"  Scenarios evaluated:         {metrics['evaluated']}")
    print(f"  Valid format rate:           {metrics['valid_format']:.1%}")
    print(f"  ── Plan Quality (vs Gemini gold) ──")
    print(f"  Agent-Tool pair F1:          {metrics['agent_tool_pair_f1']:.3f}")
    print(f"  Tool set F1:                 {metrics['tool_set_f1']:.3f}")
    print(f"  Agent set F1:                {metrics['agent_set_f1']:.3f}")
    print(f"  Tool sequence match:         {metrics['tool_seq_match']:.3f}")
    print(f"  Agent sequence match:        {metrics['agent_seq_match']:.3f}")
    print(f"  Arg key overlap:             {metrics['arg_key_overlap']:.3f}")
    print(f"  Over-decomposition:          {metrics['over_decomposition']:.2f} (0=same, >0=more steps)")
    print(f"  Avg steps (gold / cand):     {metrics['avg_gold_steps']:.1f} / {metrics['avg_cand_steps']:.1f}")
    print(f"  ── Composite ──")
    print(f"  Structural score:            {metrics['structural_score']:.3f}")

    return results, metrics


# ── Level 2: LLM Judge Evaluation ──────────────────────────────────────────


_JUDGE_PROMPT = """\
You are evaluating the quality of an AI-generated plan for industrial asset operations.

## Question
{question}

## Gold Reference Plan (from Gemini 2.5 Flash — near-optimal)
{gold_plan}

## Candidate Plan (from a smaller model being evaluated)
{candidate_plan}

## Available Agents & Tools
- IoTAgent: sites(), assets(site_name), sensors(site_name, asset_id), history(site_name, asset_id, start, final)
- FMSRAgent: get_failure_modes(asset_name), get_failure_mode_sensor_mapping(asset_name, failure_modes, sensors)
- TSFMAgent: get_ai_tasks(), get_tsfm_models(), run_tsfm_forecasting(...), run_tsfm_finetuning(...), run_tsad(...), run_integrated_tsad(...)
- Utilities: json_reader(json_string), current_date_time(), current_time_english()

## Evaluation Criteria
Rate the candidate plan on each dimension (1-5 scale):

1. **correctness** (1-5): Would this plan, if executed, correctly answer the question?
   - 5: Plan would fully answer the question
   - 3: Partially correct — right direction but missing key steps or wrong approach
   - 1: Would not answer the question at all

2. **agent_routing** (1-5): Are the correct agents assigned?
   - 5: All agents match the gold reference
   - 3: Most agents correct but some wrong
   - 1: Major agent misassignments

3. **tool_selection** (1-5): Are the correct tools selected for each agent?
   - 5: All tools match gold reference
   - 3: Most tools correct but some wrong/missing
   - 1: Major tool errors

4. **argument_quality** (1-5): Are tool arguments reasonable and correct?
   - 5: Args match gold reference closely
   - 3: Partially correct args
   - 1: Major arg errors or missing required args

5. **efficiency** (1-5): Is the plan appropriately sized? (not over/under-decomposed)
   - 5: Same number of actionable steps as gold
   - 3: 1-2 extra or missing steps
   - 1: Severely over-decomposed (3+ unnecessary steps) or missing critical steps

6. **dependency_correctness** (1-5): Are step dependencies correctly specified?
   - 5: Dependencies match logical flow
   - 3: Minor dependency errors
   - 1: Major dependency errors that would break execution

Respond with ONLY a JSON object (no markdown fences):
{{"correctness": <int>, "agent_routing": <int>, "tool_selection": <int>, "argument_quality": <int>, "efficiency": <int>, "dependency_correctness": <int>, "explanation": "<one sentence summary>"}}
"""


def call_judge(question, gold_plan, candidate_plan, model, api_key):
    """Call LLM judge to evaluate a single plan."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        gold_plan=gold_plan or "(no gold plan available)",
        candidate_plan=candidate_plan or "(empty — no plan generated)",
    )

    # Gemini 2.5 Flash uses ~300 thinking tokens, so needs higher max_tokens
    max_tok = 1024 if "2.5" in model else 300

    for attempt in range(3):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tok,
                api_key=api_key,
            )
            raw = response.choices[0].message.content or ""
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                raw = raw.lstrip("json").strip()

            result = json.loads(raw)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            elif attempt < 2:
                time.sleep(3)
            else:
                return {"error": str(e)[:100]}

    return {"error": "Failed after 3 attempts"}


def run_judge_eval(gold_data, cand_data, judge_model="gemini/gemini-2.5-flash", api_key=None):
    """Run LLM-as-judge evaluation."""
    if api_key is None:
        api_key = GEMINI_API_KEY

    gold_by_id = {d["id"]: d for d in gold_data}
    cand_by_id = {d["id"]: d for d in cand_data}
    common_ids = sorted(set(gold_by_id) & set(cand_by_id))

    # Only evaluate scenarios where candidate has a response
    evaluable = [sid for sid in common_ids if cand_by_id[sid].get("response")]

    print(f"\n  LLM Judge eval: {len(evaluable)} scenarios with {judge_model}")
    print(f"  Est. cost: ~${len(evaluable) * 0.001:.3f}")

    results = []
    errors = 0
    t0 = time.time()

    for i, sid in enumerate(evaluable):
        question = gold_by_id[sid].get("prompt", "")
        gold_resp = gold_by_id[sid].get("response", "")
        cand_resp = cand_by_id[sid].get("response", "")

        verdict = call_judge(question, gold_resp, cand_resp, judge_model, api_key)

        if "error" in verdict:
            errors += 1
            print(f"  ERROR [{errors}] ID {sid}: {verdict['error']}")

        verdict["id"] = sid
        verdict["prompt"] = question
        results.append(verdict)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(evaluable) - i - 1)
            ok = sum(1 for r in results if "correctness" in r)
            print(f"  [{i+1}/{len(evaluable)}] Judged: {ok} | Errors: {errors} | ETA: {eta:.0f}s")

        # Rate limiting
        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s ({elapsed/max(len(evaluable),1):.1f}s/scenario)")

    # Aggregate
    scored = [r for r in results if "correctness" in r]
    if not scored:
        print("  No successfully judged scenarios!")
        return results, {}

    dimensions = ["correctness", "agent_routing", "tool_selection",
                   "argument_quality", "efficiency", "dependency_correctness"]
    metrics = {}
    for dim in dimensions:
        vals = [r[dim] for r in scored if dim in r]
        metrics[f"avg_{dim}"] = sum(vals) / len(vals) if vals else 0
        metrics[f"{dim}_dist"] = dict(Counter(vals))

    # Overall score (average of all dimensions)
    all_scores = []
    for r in scored:
        dims = [r.get(d, 0) for d in dimensions]
        all_scores.append(sum(dims) / len(dims))
    metrics["avg_overall"] = sum(all_scores) / len(all_scores) if all_scores else 0
    metrics["judged"] = len(scored)
    metrics["errors"] = errors

    # Score distribution
    perfect = sum(1 for s in all_scores if s >= 4.5)
    good = sum(1 for s in all_scores if 3.5 <= s < 4.5)
    fair = sum(1 for s in all_scores if 2.5 <= s < 3.5)
    poor = sum(1 for s in all_scores if s < 2.5)

    print(f"\n{'─' * 65}")
    print(f"  LLM JUDGE EVALUATION RESULTS (judge: {judge_model})")
    print(f"{'─' * 65}")
    print(f"  Scenarios judged:            {metrics['judged']}")
    print(f"  Judge errors:                {metrics['errors']}")
    print(f"  ── Dimension Scores (1-5 scale) ──")
    for dim in dimensions:
        avg = metrics[f"avg_{dim}"]
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {dim:28s} {avg:.2f}  {bar}")
    print(f"  {'─' * 45}")
    avg_o = metrics["avg_overall"]
    bar = "█" * int(avg_o) + "░" * (5 - int(avg_o))
    print(f"  {'OVERALL':28s} {avg_o:.2f}  {bar}")
    print(f"  ── Quality Distribution ──")
    print(f"  Perfect (≥4.5):              {perfect}/{len(scored)} ({100*perfect/len(scored):.0f}%)")
    print(f"  Good (3.5-4.5):              {good}/{len(scored)} ({100*good/len(scored):.0f}%)")
    print(f"  Fair (2.5-3.5):              {fair}/{len(scored)} ({100*fair/len(scored):.0f}%)")
    print(f"  Poor (<2.5):                 {poor}/{len(scored)} ({100*poor/len(scored):.0f}%)")

    return results, metrics


# ── Model Runner (optional: generate plans then evaluate) ──────────────────


def run_model_informed(scenarios, agent_descriptions, model_cfg):
    """Generate plans using a small model via OpenRouter."""
    model_id = model_cfg["id"]
    max_tokens = model_cfg["max_tokens"]
    delay = model_cfg["delay"]

    agents_text = "\n\n".join(
        f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
    )

    results = []
    errors = 0
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        full_prompt = _PLAN_PROMPT.format(agents=agents_text, question=sc["text"])
        try:
            for attempt in range(4):
                try:
                    response = litellm.completion(
                        model=model_id,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0,
                        max_tokens=max_tokens,
                        api_key=OPENROUTER_API_KEY,
                    )
                    raw = response.choices[0].message.content or ""
                    answer = strip_thinking(raw)
                    plan = parse_plan(answer)
                    results.append({
                        "id": sc["id"],
                        "prompt": sc["text"],
                        "response": answer,
                        "plan_steps": len(plan.steps),
                        "agents_per_step": [s.agent for s in plan.steps],
                        "tools_per_step": [s.tool for s in plan.steps],
                    })
                    break
                except Exception as e:
                    if attempt < 3 and ("429" in str(e) or "rate" in str(e).lower()):
                        time.sleep(10 * (attempt + 1))
                    else:
                        raise
        except Exception as e:
            errors += 1
            results.append({
                "id": sc["id"], "prompt": sc["text"],
                "response": None, "error": str(e)[:200],
            })

        if (i + 1) % 10 == 0:
            valid = sum(1 for r in results if r.get("plan_steps", 0) > 0)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(scenarios)}] Valid: {valid}/{i+1} | Errors: {errors}")

        time.sleep(delay)

    return results


# ── Gemini self-eval (evaluate Gemini against itself as sanity check) ──────


def gemini_self_eval_structural(gold_data):
    """Run structural eval of Gemini against itself (should score ~1.0)."""
    print("\n  === Gemini Self-Eval (sanity check) ===")
    results, metrics = run_structural_eval(gold_data, gold_data)
    return metrics


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Proper plan quality evaluation for AssetOpsBench"
    )
    parser.add_argument(
        "--candidate", type=str,
        help="Path to candidate results JSON file"
    )
    parser.add_argument(
        "--gold", type=str, default=GOLD_PATH,
        help="Path to gold reference results JSON (default: Gemini informed)"
    )
    parser.add_argument(
        "--mode", choices=["structural", "judge", "all"], default="all",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--judge-model", type=str, default="gemini/gemini-3-flash-preview",
        help="Model to use as LLM judge (default: gemini-3-flash-preview, "
             "which doesn't waste tokens on thinking like 2.5-flash)"
    )
    parser.add_argument(
        "--run-model", type=str, choices=list(RUN_MODELS.keys()),
        help="Run a model first, then evaluate (generates plans via OpenRouter)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of scenarios (0=all)"
    )
    parser.add_argument(
        "--self-eval", action="store_true",
        help="Run Gemini self-evaluation as sanity check"
    )
    parser.add_argument(
        "--output", type=str,
        help="Save detailed results to JSON file"
    )
    args = parser.parse_args()

    # Load gold data
    print(f"Loading gold reference: {args.gold}")
    gold_data = load_json(args.gold)
    gold_valid = sum(1 for d in gold_data if d.get("response"))
    print(f"  {len(gold_data)} scenarios, {gold_valid} with responses")

    # Sanity check
    if args.self_eval:
        gemini_self_eval_structural(gold_data)
        return

    # Load or generate candidate data
    if args.run_model:
        print(f"\nGenerating plans with {args.run_model}...")
        from workflow.executor import Executor
        from llm.litellm import LiteLLMBackend

        scenarios = [{"id": d["id"], "text": d["prompt"]} for d in gold_data]
        if args.limit:
            scenarios = scenarios[:args.limit]

        print("Discovering agent capabilities via MCP servers...")
        llm = LiteLLMBackend("gemini/gemini-2.5-flash")
        executor = Executor(llm)
        agent_descs = asyncio.run(executor.get_agent_descriptions())
        print(f"  Discovered {len(agent_descs)} agents")

        cfg = RUN_MODELS[args.run_model]
        print(f"\nRunning {args.run_model} ({cfg['params']}) on {len(scenarios)} scenarios...")
        cand_data = run_model_informed(scenarios, agent_descs, cfg)

        # Save results
        outfile = os.path.join(BASEDIR, f"{args.run_model}_informed_results.json")
        with open(outfile, "w") as f:
            json.dump(cand_data, f, indent=2)
        print(f"  Saved: {outfile}")

    elif args.candidate:
        print(f"\nLoading candidate: {args.candidate}")
        cand_data = load_json(args.candidate)
        cand_valid = sum(1 for d in cand_data if d.get("response"))
        print(f"  {len(cand_data)} scenarios, {cand_valid} with responses")
    else:
        parser.error("Must specify --candidate or --run-model")

    if args.limit and not args.run_model:
        cand_data = cand_data[:args.limit]

    # Run evaluations
    all_results = {"candidate": args.candidate or args.run_model}
    all_metrics = {}

    if args.mode in ("structural", "all"):
        struct_results, struct_metrics = run_structural_eval(gold_data, cand_data)
        all_results["structural"] = struct_results
        all_metrics["structural"] = struct_metrics

    if args.mode in ("judge", "all"):
        judge_results, judge_metrics = run_judge_eval(
            gold_data, cand_data,
            judge_model=args.judge_model,
            api_key=GEMINI_API_KEY,
        )
        all_results["judge"] = judge_results
        all_metrics["judge"] = judge_metrics

    # Final summary
    print(f"\n{'=' * 65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 65}")

    if "structural" in all_metrics:
        sm = all_metrics["structural"]
        print(f"  Structural (vs Gemini gold):")
        print(f"    Agent-Tool pair F1:  {sm.get('agent_tool_pair_f1', 0):.3f}")
        print(f"    Structural score:    {sm.get('structural_score', 0):.3f}")

    if "judge" in all_metrics:
        jm = all_metrics["judge"]
        print(f"  LLM Judge ({args.judge_model}):")
        print(f"    Overall:             {jm.get('avg_overall', 0):.2f}/5.0")
        print(f"    Correctness:         {jm.get('avg_correctness', 0):.2f}/5.0")
        print(f"    Tool selection:      {jm.get('avg_tool_selection', 0):.2f}/5.0")

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": all_metrics, "details": all_results}, f, indent=2)
        print(f"\n  Detailed results saved: {args.output}")

    print(f"\n{'=' * 65}")
    print("Done.")


if __name__ == "__main__":
    main()
