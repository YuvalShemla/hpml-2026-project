#!/usr/bin/env python3
"""Generate additional training data to fill quality gaps.

Targets ~1,800 new examples to bring total to ~2,500:
  - Tool knowledge with varied phrasings (300 new)
  - Hard negatives (54 new)
  - Clarification/abstention (45 new)
  - Better paraphrases with domain context (440 new)
  - WorkOrder-specific plans (80 new from 47 HF scenarios)
  - Vibration-specific plans (35 new)

Usage:
    uv run python benchmark/generate_data/generate_additional_data.py
"""

import json
import os
import random
import re
import sys
import time
import hashlib
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import litellm
litellm.suppress_debug_info = True

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini/gemini-2.5-flash"
SEED = 42
random.seed(SEED)

DATASETS_DIR = Path(__file__).parent / "datasets"

# ─── Tool catalog (same as generate_all_datasets.py) ─────────────────────────

TOOL_CATALOG = {
    "IoTAgent": {
        "desc": "Handles IoT telemetry data — sites, assets, sensors, historical readings from industrial equipment.",
        "tools": {
            "sites": ("List all available IoT monitoring sites", {}),
            "assets": ("List all asset IDs for a given site", {"site_name": "str"}),
            "sensors": ("List sensor names for an asset at a site", {"site_name": "str", "asset_id": "str"}),
            "history": ("Fetch historical sensor readings for a time range", {"site_name": "str", "asset_id": "str", "start": "str ISO8601", "final": "str ISO8601 (optional)"}),
        },
    },
    "FMSRAgent": {
        "desc": "Failure mode analysis and sensor relevance reasoning for industrial assets.",
        "tools": {
            "get_failure_modes": ("Return known failure modes for an asset type", {"asset_name": "str"}),
            "get_failure_mode_sensor_mapping": ("Determine sensor relevancy for failure modes", {"asset_name": "str", "failure_modes": "list[str]", "sensors": "list[str]"}),
        },
    },
    "TSFMAgent": {
        "desc": "Time-series forecasting and anomaly detection using IBM Granite TinyTimeMixer.",
        "tools": {
            "get_ai_tasks": ("List supported AI task types", {}),
            "get_tsfm_models": ("List available model checkpoints", {}),
            "run_tsfm_forecasting": ("Zero-shot time-series forecasting", {"dataset_path": "str", "timestamp_column": "str", "target_columns": "list[str]"}),
            "run_tsfm_finetuning": ("Few-shot fine-tune a model", {"dataset_path": "str", "timestamp_column": "str", "target_columns": "list[str]"}),
            "run_tsad": ("Conformal anomaly detection on forecasting output", {"dataset_path": "str", "tsfm_output_json": "str", "timestamp_column": "str", "target_columns": "list[str]"}),
            "run_integrated_tsad": ("End-to-end forecasting + anomaly detection", {"dataset_path": "str", "timestamp_column": "str", "target_columns": "list[str]"}),
        },
    },
    "Utilities": {
        "desc": "General utility functions — file reading, date/time.",
        "tools": {
            "json_reader": ("Read and parse a JSON file", {"file_name": "str"}),
            "current_date_time": ("Return current UTC date/time as JSON", {}),
            "current_time_english": ("Return current UTC time as readable string", {}),
        },
    },
    "WorkOrderAgent": {
        "desc": "Work order and maintenance event reasoning — queries maintenance records, failure codes, predicts future work orders.",
        "tools": {
            "get_work_orders": ("Retrieve work orders for an asset", {"asset_name": "str", "start_date": "str (optional)", "end_date": "str (optional)"}),
            "get_preventive_work_orders": ("Retrieve preventive maintenance work orders", {"asset_name": "str"}),
            "get_corrective_work_orders": ("Retrieve corrective/reactive work orders", {"asset_name": "str"}),
            "get_events": ("Retrieve maintenance events and alerts", {"asset_name": "str"}),
            "get_failure_codes": ("Retrieve failure codes from work orders", {"asset_name": "str"}),
            "get_work_order_distribution": ("Get work order distribution stats", {"asset_name": "str", "group_by": "str (optional)"}),
            "predict_next_work_order": ("Predict when next work order is needed", {"asset_name": "str"}),
            "analyze_alert_to_failure": ("Analyze alert-to-failure correlations", {"asset_name": "str"}),
        },
    },
    "VibrationAgent": {
        "desc": "Vibration analysis — FFT spectra, envelope analysis, bearing diagnostics, severity assessment.",
        "tools": {
            "get_vibration_data": ("Retrieve raw vibration data", {"sensor_id": "str"}),
            "list_vibration_sensors": ("List vibration sensors for an asset", {"asset_name": "str"}),
            "compute_fft_spectrum": ("Compute FFT frequency spectrum", {"sensor_id": "str"}),
            "compute_envelope_spectrum": ("Compute envelope spectrum for bearing faults", {"sensor_id": "str"}),
            "assess_vibration_severity": ("Assess severity against ISO standards", {"sensor_id": "str"}),
            "calculate_bearing_frequencies": ("Calculate characteristic bearing fault frequencies", {"bearing_model": "str", "shaft_rpm": "float"}),
            "list_known_bearings": ("List known bearing models", {}),
            "diagnose_vibration": ("Run comprehensive vibration diagnosis", {"asset_name": "str"}),
        },
    },
}

ALL_AGENTS = list(TOOL_CATALOG.keys())
FLAT_TOOLS = [(agent, tool) for agent, info in TOOL_CATALOG.items() for tool in info["tools"]]


def call_gemini(prompt, temperature=0.3, max_tokens=2048, retries=3):
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=GEMINI_API_KEY,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error: {err[:80]}")
                if attempt < retries - 1:
                    time.sleep(3)
    return None


def make_example(instruction, response, category, source="synthetic"):
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"category": category, "source": source},
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. VARIED TOOL KNOWLEDGE (~300 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_varied_tool_knowledge():
    """Generate tool knowledge QA with varied phrasings via Gemini."""
    examples = []

    # 1a. Batch: For each agent+tool, generate 3 varied Q&A pairs
    print("  Generating varied tool knowledge Q&A...")
    batch_count = 0
    for agent, info in TOOL_CATALOG.items():
        for tool, (desc, args) in info["tools"].items():
            args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if args else "none"

            prompt = f"""You are generating training data for a model that needs to learn about industrial asset operations tools.

Tool: {agent}.{tool}
Description: {desc}
Arguments: {args_str}
Agent description: {info['desc']}

Generate exactly 3 question-answer pairs about this tool. Each pair should test a DIFFERENT aspect:
1. A natural question asking what tool to use for a task (routing)
2. A question about the tool's arguments or usage
3. A question phrased as a user request (e.g., "I need to..." or "Help me...")

Use industrial domain language. Vary formality. Keep answers concise (1-2 sentences).

Format each as:
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>
Q3: <question>
A3: <answer>"""

            resp = call_gemini(prompt, temperature=0.5, max_tokens=1024)
            if not resp:
                continue

            for m in re.finditer(r'Q\d+:\s*(.+?)\nA\d+:\s*(.+?)(?=\nQ\d+:|\Z)', resp, re.DOTALL):
                q = m.group(1).strip()
                a = m.group(2).strip()
                if len(q) > 10 and len(a) > 10:
                    examples.append(make_example(q, a, "tool_knowledge_varied", "synthetic_varied"))

            batch_count += 1
            if batch_count % 10 == 0:
                print(f"    {batch_count}/{len(FLAT_TOOLS)} tools done, {len(examples)} examples")
            time.sleep(0.5)

    print(f"  Generated {len(examples)} varied tool knowledge examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 2. HARD NEGATIVES (~54 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_hard_negatives():
    """Generate hard negative examples — wrong routing with corrections."""
    examples = []

    # For each agent pair, generate confusable scenarios
    agent_pairs = [
        ("IoTAgent", "FMSRAgent", "sensor data vs failure analysis"),
        ("IoTAgent", "WorkOrderAgent", "sensor readings vs maintenance records"),
        ("IoTAgent", "TSFMAgent", "raw data vs forecasting/anomaly detection"),
        ("FMSRAgent", "WorkOrderAgent", "failure modes vs work order history"),
        ("FMSRAgent", "VibrationAgent", "failure analysis vs vibration diagnostics"),
        ("TSFMAgent", "WorkOrderAgent", "time-series prediction vs maintenance prediction"),
        ("TSFMAgent", "VibrationAgent", "anomaly detection vs vibration analysis"),
        ("WorkOrderAgent", "VibrationAgent", "maintenance events vs vibration data"),
        ("Utilities", "IoTAgent", "reading files vs reading sensor data"),
    ]

    print("  Generating hard negatives...")
    for agent1, agent2, confusion in agent_pairs:
        tools1 = list(TOOL_CATALOG[agent1]["tools"].keys())
        tools2 = list(TOOL_CATALOG[agent2]["tools"].keys())

        prompt = f"""You are generating training data about INCORRECT tool routing in industrial asset operations.

Agent 1: {agent1} — {TOOL_CATALOG[agent1]['desc']}
  Tools: {', '.join(tools1)}

Agent 2: {agent2} — {TOOL_CATALOG[agent2]['desc']}
  Tools: {', '.join(tools2)}

Common confusion: {confusion}

Generate 6 examples where someone might confuse these agents. For each:
- Write a user query
- State the WRONG agent.tool someone might pick
- State the CORRECT agent.tool
- Explain why in one sentence

Format:
QUERY: <question>
WRONG: <Agent.tool>
CORRECT: <Agent.tool>
WHY: <explanation>
---"""

        resp = call_gemini(prompt, temperature=0.5, max_tokens=2048)
        if not resp:
            continue

        blocks = resp.split("---")
        for block in blocks:
            q_m = re.search(r'QUERY:\s*(.+)', block)
            w_m = re.search(r'WRONG:\s*(.+)', block)
            c_m = re.search(r'CORRECT:\s*(.+)', block)
            why_m = re.search(r'WHY:\s*(.+)', block)

            if q_m and w_m and c_m and why_m:
                query = q_m.group(1).strip()
                wrong = w_m.group(1).strip()
                correct = c_m.group(1).strip()
                why = why_m.group(1).strip()

                instruction = f'A system routed the query "{query}" to {wrong}. What is the correct routing and why?'
                response = f"Incorrect: {wrong}. Correct: {correct}. {why}"
                examples.append(make_example(instruction, response, "hard_negative", "synthetic_hn"))

        time.sleep(0.5)

    print(f"  Generated {len(examples)} hard negatives")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 3. CLARIFICATION & ABSTENTION (~45 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_clarification_examples():
    """Generate examples where the correct response is to ask for clarification."""
    examples = []

    prompt = """You are generating training data for an industrial AI assistant. Generate 45 examples where the user's query is AMBIGUOUS, INCOMPLETE, or IMPOSSIBLE to answer with the available tools. The assistant should ask for clarification rather than hallucinate.

Available agents: IoTAgent (IoT sensor data), FMSRAgent (failure mode analysis), TSFMAgent (time-series forecasting/anomaly detection), Utilities (file reading, datetime), WorkOrderAgent (work order/maintenance records), VibrationAgent (vibration analysis).

Categories to cover:
- Missing asset/site name (10 examples)
- Ambiguous tool choice (10 examples)
- Task outside agent capabilities (10 examples)
- Insufficient parameters for the tool (10 examples)
- Vague/underspecified intent (5 examples)

Format each as:
USER: <ambiguous query>
ASSISTANT: <clarification response explaining what info is needed and which agents/tools could help>
---

Make the queries realistic — things a facility manager or engineer would actually ask. Keep assistant responses helpful, not dismissive. Mention specific agents/tools that COULD help once the user clarifies."""

    print("  Generating clarification examples...")
    resp = call_gemini(prompt, temperature=0.6, max_tokens=8192)
    if resp:
        blocks = resp.split("---")
        for block in blocks:
            u_m = re.search(r'USER:\s*(.+?)(?=\nASSISTANT:)', block, re.DOTALL)
            a_m = re.search(r'ASSISTANT:\s*(.+)', block, re.DOTALL)
            if u_m and a_m:
                user = u_m.group(1).strip()
                asst = a_m.group(1).strip()
                if len(user) > 5 and len(asst) > 20:
                    examples.append(make_example(user, asst, "clarification", "synthetic_clarification"))

    print(f"  Generated {len(examples)} clarification examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 4. BETTER PARAPHRASES (~440 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_better_paraphrases():
    """Generate paraphrases with domain context to avoid off-topic rewrites."""
    examples = []

    # Load gold plans
    gold_path = Path(__file__).parent.parent / "baseline_tests" / "gemini_flash_informed_results.json"
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold_plans = [g for g in gold_raw if g.get("plan_steps", 0) > 0]

    print(f"  Generating paraphrases for {len(gold_plans)} gold plans (3 each)...")

    for i, plan in enumerate(gold_plans):
        if i % 20 == 0:
            print(f"    {i}/{len(gold_plans)}...")

        prompt = f"""Rewrite this industrial asset operations question in 3 different ways.
IMPORTANT CONTEXT: This is about industrial IoT monitoring, chillers, AHUs, sensors, and maintenance — NOT about websites or general IT.
- "sites" means physical monitoring sites/facilities (like "MAIN site"), not websites
- "assets" means industrial equipment (chillers, AHUs, pumps), not financial assets
- Keep the same meaning, same specific entities (site names, asset names, sensor names)
- Vary phrasing, formality, and structure

Original: {plan['prompt']}

Return exactly 3 lines, numbered 1-3, nothing else:"""

        resp = call_gemini(prompt, temperature=0.7, max_tokens=512)
        if not resp:
            continue

        for line in resp.strip().split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"')
            if len(line) > 10 and line.lower() != plan["prompt"].lower():
                examples.append(make_example(
                    line, plan["response"], "planning", "paraphrase_v2",
                ))

        time.sleep(0.5)

    print(f"  Generated {len(examples)} paraphrases")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 5. WORKORDER-SPECIFIC PLANS (~80 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_workorder_plans():
    """Generate plans for the 47 WorkOrder scenarios from HuggingFace."""
    examples = []

    from datasets import load_dataset
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    wo_scenarios = [dict(r) for r in ds["train"] if r["type"] == "Workorder"]

    tool_catalog_str = "\n".join(
        f"{agent}: {info['desc']}\n" +
        "\n".join(f"  - {tool}({', '.join(args.keys()) if args else ''}): {desc}"
                  for tool, (desc, args) in info["tools"].items())
        for agent, info in TOOL_CATALOG.items()
    )

    print(f"  Generating plans for {len(wo_scenarios)} WorkOrder scenarios...")
    for i, sc in enumerate(wo_scenarios):
        if i % 10 == 0:
            print(f"    {i}/{len(wo_scenarios)}...")

        prompt = f"""You are an expert planner for industrial asset operations. Generate a structured plan for this question.

AVAILABLE TOOLS:
{tool_catalog_str}

OUTPUT FORMAT (exactly this structure):
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Use ONLY agents and tools from the list above
- WorkOrder questions should primarily use WorkOrderAgent tools
- Keep plans concise (minimum steps)
- Use {{step_N}} for inter-step dependencies

QUESTION: {sc['text']}"""

        resp = call_gemini(prompt, temperature=0.1, max_tokens=2048)
        if not resp:
            continue

        if "#Task1:" in resp and "#Agent1:" in resp:
            num_steps = len(re.findall(r"#Task\d+:", resp))
            examples.append(make_example(
                sc["text"], resp.strip(), "planning", "synthetic_workorder",
            ))
            examples[-1]["metadata"]["num_steps"] = num_steps
            examples[-1]["metadata"]["scenario_id"] = sc["id"]

        time.sleep(0.5)

    print(f"  Generated {len(examples)} WorkOrder plans")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 6. VIBRATION-SPECIFIC PLANS (~35 examples)
# ═══════════════════════════════════════════════════════════════════════

def generate_vibration_plans():
    """Generate plans involving VibrationAgent tools."""
    examples = []

    vibration_questions = [
        "Check the vibration levels on Compressor 1 and tell me if they exceed ISO limits.",
        "I suspect a bearing fault on Pump 2. Can you run a vibration diagnosis?",
        "Compute the FFT spectrum for vibration sensor V-101 on the main compressor.",
        "List all vibration sensors installed on the cooling tower.",
        "What bearing models are available in the system database?",
        "Assess vibration severity for all sensors on AHU-3.",
        "Calculate the expected bearing fault frequencies for a 6205 bearing at 1800 RPM.",
        "Run envelope spectrum analysis on sensor V-203 to check for inner race defects.",
        "Compare vibration data from sensors V-101 and V-102 on Motor 1 for the last month.",
        "Is there abnormal vibration on any equipment at the MAIN site? Check all vibration sensors.",
        "Get raw vibration data for sensor V-301 from January 2021.",
        "The pump is making unusual noise. Run a full vibration diagnosis and check bearing frequencies.",
        "List vibration sensors on Chiller 6 and assess their severity levels.",
        "Compute FFT spectrum for the drive-end sensor on Motor 2 and identify dominant frequencies.",
        "What are the BPFO and BPFI frequencies for a 6308 bearing running at 3600 RPM?",
        "Check if Compressor 2 shows signs of misalignment using vibration data.",
        "Get vibration data for all sensors on Pump 1 and flag any that exceed 7.1 mm/s RMS.",
        "Run envelope analysis on the fan motor bearing sensor to detect early-stage spalling.",
        "I need a comprehensive vibration health report for all rotating equipment at MAIN site.",
        "Compare vibration severity of Pump 1 vs Pump 2 — which needs attention first?",
    ]

    tool_catalog_str = "\n".join(
        f"{agent}: {info['desc']}\n" +
        "\n".join(f"  - {tool}({', '.join(args.keys()) if args else ''}): {desc}"
                  for tool, (desc, args) in info["tools"].items())
        for agent, info in TOOL_CATALOG.items()
    )

    print(f"  Generating plans for {len(vibration_questions)} vibration scenarios...")
    for i, question in enumerate(vibration_questions):
        prompt = f"""You are an expert planner for industrial asset operations. Generate a structured plan.

AVAILABLE TOOLS:
{tool_catalog_str}

OUTPUT FORMAT:
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Keep plans concise
- Use VibrationAgent tools for vibration analysis tasks
- Use {{step_N}} for dependencies

QUESTION: {question}"""

        resp = call_gemini(prompt, temperature=0.1, max_tokens=2048)
        if not resp:
            continue

        if "#Task1:" in resp and "#Agent1:" in resp:
            examples.append(make_example(question, resp.strip(), "planning", "synthetic_vibration"))

        time.sleep(0.5)

    print(f"  Generated {len(examples)} vibration plans")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def deduplicate(examples):
    seen = set()
    unique = []
    for ex in examples:
        h = hashlib.md5(ex["messages"][0]["content"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def save_jsonl(examples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} -> {path}")


def main():
    print("=" * 60)
    print("  ADDITIONAL DATA GENERATION")
    print("=" * 60)

    all_new = []

    # 1. Varied tool knowledge
    print("\n[1/6] Varied tool knowledge...")
    tk = generate_varied_tool_knowledge()
    all_new.extend(tk)

    # 2. Hard negatives
    print("\n[2/6] Hard negatives...")
    hn = generate_hard_negatives()
    all_new.extend(hn)

    # 3. Clarification
    print("\n[3/6] Clarification examples...")
    cl = generate_clarification_examples()
    all_new.extend(cl)

    # 4. Better paraphrases
    print("\n[4/6] Better paraphrases...")
    pp = generate_better_paraphrases()
    all_new.extend(pp)

    # 5. WorkOrder plans
    print("\n[5/6] WorkOrder plans...")
    wo = generate_workorder_plans()
    all_new.extend(wo)

    # 6. Vibration plans
    print("\n[6/6] Vibration plans...")
    vb = generate_vibration_plans()
    all_new.extend(vb)

    # Deduplicate new data
    all_new = deduplicate(all_new)

    # Save new data separately
    save_jsonl(all_new, DATASETS_DIR / "additional_data.jsonl")

    # Now merge with existing datasets
    print("\n[MERGE] Combining with existing data...")
    existing = {}
    for name in ["tool_knowledge", "planning", "execution"]:
        path = DATASETS_DIR / f"{name}.jsonl"
        if path.exists():
            with open(path) as f:
                existing[name] = [json.loads(line) for line in f]
            print(f"  Existing {name}: {len(existing[name])}")

    # Merge by category
    merged_tk = existing.get("tool_knowledge", []) + [e for e in all_new if "knowledge" in e["metadata"]["category"] or "hard_negative" in e["metadata"]["category"] or "clarification" in e["metadata"]["category"]]
    merged_plan = existing.get("planning", []) + [e for e in all_new if "planning" in e["metadata"]["category"] or e["metadata"]["source"].startswith("paraphrase")]
    merged_exec = existing.get("execution", [])  # no new execution data

    merged_tk = deduplicate(merged_tk)
    merged_plan = deduplicate(merged_plan)

    save_jsonl(merged_tk, DATASETS_DIR / "tool_knowledge.jsonl")
    save_jsonl(merged_plan, DATASETS_DIR / "planning.jsonl")

    # Combined
    combined = merged_tk + merged_plan + merged_exec
    combined = deduplicate(combined)
    save_jsonl(combined, DATASETS_DIR / "combined_all.jsonl")

    # Summary
    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"  New examples generated:  {len(all_new)}")
    print(f"  Tool knowledge (merged): {len(merged_tk)}")
    print(f"  Planning (merged):       {len(merged_plan)}")
    print(f"  Execution (unchanged):   {len(merged_exec)}")
    print(f"  Combined total:          {len(combined)}")
    print()

    # Breakdown of new data
    cats = Counter(e["metadata"]["source"] for e in all_new)
    print("  New data by source:")
    for src, count in cats.most_common():
        print(f"    {src}: {count}")


if __name__ == "__main__":
    main()
