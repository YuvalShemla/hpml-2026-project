#!/usr/bin/env python3
"""Generate the end-to-end Colab notebook as .ipynb.

v2: Proper train/test split, specialist models, ROUGE evaluation, token context comparison.

Usage:
    python notebook/build_notebook.py
    # Produces: notebook/AssetOpsBench_Gemma_FineTuning.ipynb
"""

import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n") if "\n" in source else [source],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n") if "\n" in source else [source],
        "outputs": [],
        "execution_count": None,
    }


def build_notebook():
    cells = []

    # ══════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""# Internalizing MCP Tool Knowledge in Small LLMs
## End-to-End: Baseline → Generalist Fine-Tuning → Specialist Models → Comparison

**HPML Group 20 — Columbia University, Spring 2026**

Pipeline on **1x H100 (80 GB)**:

1. **Setup & data** — clone repo, load datasets, create proper train/test split (no leakage)
2. **Baseline evaluation** — test the base model in informed (with tool descriptions) and blind (no descriptions) mode
3. **Generalist fine-tuning** — train one model on ALL tool knowledge + plans
4. **Specialist fine-tuning** — train per-domain models (IoT, FMSR, TSFM, WorkOrder)
5. **Evaluation** — compare generalist vs specialists on held-out scenarios, measure token savings

**Research question:** *Can Gemma internalize tool descriptions to plan without them in the prompt? Do domain specialists outperform a generalist?*"""))

    # ══════════════════════════════════════════════════════════════════
    # 0. SETUP
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 0. Setup & Installation"))

    cells.append(code("""!pip install -U transformers>=5.5.0 peft>=0.13.0 trl>=0.12.0
!pip install -U bitsandbytes>=0.44.0 accelerate>=1.0.0
!pip install -q datasets evaluate rouge-score
!pip install -q pandas matplotlib seaborn tqdm
!pip install -q sentencepiece protobuf

import transformers
print(f"transformers version: {transformers.__version__}")
print("All packages installed.")"""))

    cells.append(code("""import os, json, re, time, random, warnings, inspect
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")"""))

    # ══════════════════════════════════════════════════════════════════
    # 1. CONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 1. Configuration"))

    cells.append(code("""# ══════════════════════════════════════════════════════════════
# LIGHT MODE: set True for a quick test run (~30-60 min total)
# Set False for the full experiment (~4-5 hours)
# ══════════════════════════════════════════════════════════════
LIGHT_MODE = True

# ── Model ─────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-E4B-it"  # 4.5B params (2.3B effective)
LOAD_IN_8BIT = True  # 8-bit for Gemma 4 (4-bit has known bnb bug)

HF_TOKEN = ""
if not HF_TOKEN:
    from google.colab import userdata
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
        print(f"Loaded HF_TOKEN from Colab secrets")
    except Exception:
        print(f"WARNING: Set HF_TOKEN. Accept license at https://huggingface.co/{MODEL_ID}")

# ── Repo ──────────────────────────────────────────────────────
REPO_URL = "https://github.com/YuvalShemla/hpml-2026-project.git"
REPO_DIR = "/content/hpml-2026-project"

# ── QLoRA ─────────────────────────────────────────────────────
LORA_R = 32 if not LIGHT_MODE else 16
LORA_ALPHA = 64 if not LIGHT_MODE else 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# ── Training ──────────────────────────────────────────────────
# Auto-adjust for GPU memory (A100 40GB vs 80GB)
_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 40
if _gpu_mem_gb > 60:  # 80GB GPU
    MAX_SEQ_LENGTH = 2048 if not LIGHT_MODE else 1024
    PER_DEVICE_BATCH_SIZE = 4 if not LIGHT_MODE else 2
    GRADIENT_ACCUMULATION_STEPS = 4 if not LIGHT_MODE else 4
else:  # 40GB GPU
    MAX_SEQ_LENGTH = 1024 if not LIGHT_MODE else 512
    PER_DEVICE_BATCH_SIZE = 2 if not LIGHT_MODE else 1
    GRADIENT_ACCUMULATION_STEPS = 8 if not LIGHT_MODE else 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3 if not LIGHT_MODE else 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
BF16 = True

# ── Evaluation ────────────────────────────────────────────────
NUM_HELD_OUT = 50 if not LIGHT_MODE else 15
MAX_NEW_TOKENS = 1024 if not LIGHT_MODE else 512
TEMPERATURE = 0.1
TOP_P = 0.9

# ── Light mode: cap training data ────────────────────────────
MAX_TRAIN_EXAMPLES = None if not LIGHT_MODE else 300  # None = use all

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mode_str = "LIGHT (quick test)" if LIGHT_MODE else "FULL (production)"
print(f"Mode: {mode_str}")
print(f"Model: {MODEL_ID}")
print(f"QLoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Training: lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, eff_batch={PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Eval: {NUM_HELD_OUT} held-out scenarios, max_tokens={MAX_NEW_TOKENS}")
if LIGHT_MODE:
    print(f"Training cap: {MAX_TRAIN_EXAMPLES} examples per model")
    print(f"Estimated runtime: ~30-60 min")"""))

    # ══════════════════════════════════════════════════════════════════
    # 2. CLONE & LOAD
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 2. Clone Repository & Load Data"))

    cells.append(code("""if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"Cloned to {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

datasets_dir = Path(REPO_DIR) / "benchmark" / "generate_data" / "datasets"
ds_tool = load_jsonl(datasets_dir / "tool_knowledge.jsonl")
ds_plan = load_jsonl(datasets_dir / "planning.jsonl")
ds_exec = load_jsonl(datasets_dir / "execution.jsonl")

gold_path = Path(REPO_DIR) / "benchmark" / "baseline_tests" / "gemini_flash_informed_results.json"
with open(gold_path) as f:
    gold_plans_raw = json.load(f)
gold_plans = {g["id"]: g for g in gold_plans_raw if g.get("plan_steps", 0) > 0}

print(f"Tool knowledge: {len(ds_tool)}, Planning: {len(ds_plan)}, Execution: {len(ds_exec)}")
print(f"Gold plans: {len(gold_plans)}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 3. TRAIN/TEST SPLIT (NO LEAKAGE)
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 3. Train/Test Split (No Data Leakage)

We hold out scenarios for evaluation and remove them AND their paraphrases from training.
The split is stratified by scenario type (IoT, FMSA, TSFM, Workorder, multiagent)."""))

    cells.append(code("""from datasets import load_dataset

hf_ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
hf_scenarios = [dict(row) for row in hf_ds["train"]]

# Build full eval candidate list (scenarios with gold plans)
eval_candidates = []
for sc in hf_scenarios:
    if sc["id"] in gold_plans:
        eval_candidates.append({
            "id": sc["id"],
            "question": sc["text"],
            "type": sc["type"],
            "category": sc["category"],
            "gold_plan": gold_plans[sc["id"]]["response"],
            "gold_steps": gold_plans[sc["id"]]["plan_steps"],
            "gold_agents": gold_plans[sc["id"]].get("agents_used", []),
            "gold_tools": gold_plans[sc["id"]].get("tools_used", []),
        })

# Stratified split: hold out NUM_HELD_OUT scenarios, balanced by type
random.shuffle(eval_candidates)
by_type = defaultdict(list)
for sc in eval_candidates:
    by_type[sc["type"]].append(sc)

eval_scenarios = []
per_type_quota = max(1, NUM_HELD_OUT // len(by_type))
for t, scenarios in by_type.items():
    n = min(per_type_quota, len(scenarios))
    eval_scenarios.extend(scenarios[:n])

# Fill remaining quota from largest types
remaining = NUM_HELD_OUT - len(eval_scenarios)
eval_ids = {s["id"] for s in eval_scenarios}
for t in sorted(by_type.keys(), key=lambda t: -len(by_type[t])):
    for sc in by_type[t]:
        if remaining <= 0:
            break
        if sc["id"] not in eval_ids:
            eval_scenarios.append(sc)
            eval_ids.add(sc["id"])
            remaining -= 1

eval_questions_lower = {sc["question"].strip().lower() for sc in eval_scenarios}

print(f"Held-out eval scenarios: {len(eval_scenarios)}")
print(f"Type distribution: {dict(Counter(s['type'] for s in eval_scenarios))}")

# Remove held-out scenarios AND their paraphrases from training
# A paraphrase shares the same plan response as the gold scenario it paraphrases from
held_out_plans = set()
for sc in eval_scenarios:
    held_out_plans.add(sc["gold_plan"].strip())

def is_leaked(example):
    q = example["messages"][0]["content"].strip().lower()
    # Direct question match
    if q in eval_questions_lower:
        return True
    return False

clean_tool = [d for d in ds_tool if not is_leaked(d)]
clean_plan = [d for d in ds_plan if not is_leaked(d)]
clean_exec = [d for d in ds_exec if not is_leaked(d)]

print(f"\\nTraining data after removing held-out:")
print(f"  Tool knowledge: {len(clean_tool)} (was {len(ds_tool)})")
print(f"  Planning: {len(clean_plan)} (was {len(ds_plan)})")
print(f"  Execution: {len(clean_exec)} (was {len(ds_exec)})")

# Also remove paraphrases that share the exact same plan as held-out scenarios
before_para = len(clean_plan)
clean_plan = [d for d in clean_plan if d["messages"][1]["content"].strip() not in held_out_plans]
print(f"  Planning after removing shared-plan paraphrases: {len(clean_plan)} (removed {before_para - len(clean_plan)} more)")

total_train = len(clean_tool) + len(clean_plan) + len(clean_exec)
print(f"  Total clean training: {total_train}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 4. EVALUATION FRAMEWORK
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 4. Evaluation Framework

Metrics: plan format validity, agent/tool correctness, Agent-Tool pair F1, ROUGE-L."""))

    cells.append(code("""from evaluate import load as load_metric
rouge_metric = load_metric("rouge")

VALID_AGENTS = {"IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities", "WorkOrderAgent", "VibrationAgent", "none"}
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
    "none",
}

TOOL_DESCRIPTIONS = \"\"\"Available Agents and Tools:

IoTAgent: IoT telemetry data access.
  - sites(): List all available IoT sites.
  - assets(site_name): List all asset IDs for a given site.
  - sensors(site_name, asset_id): List sensor names for an asset.
  - history(site_name, asset_id, start, final?): Fetch historical sensor readings.

FMSRAgent: Failure mode analysis and sensor relevance.
  - get_failure_modes(asset_name): Return known failure modes.
  - get_failure_mode_sensor_mapping(asset_name, failure_modes, sensors): Map sensor relevancy.

TSFMAgent: Time-series forecasting and anomaly detection.
  - get_ai_tasks(): List AI task types.
  - get_tsfm_models(): List model checkpoints.
  - run_tsfm_forecasting(dataset_path, timestamp_column, target_columns): Forecast.
  - run_tsfm_finetuning(dataset_path, timestamp_column, target_columns): Fine-tune.
  - run_tsad(dataset_path, tsfm_output_json, timestamp_column, target_columns): Anomaly detection.
  - run_integrated_tsad(dataset_path, timestamp_column, target_columns): E2E forecast + anomaly.

Utilities:
  - json_reader(file_name): Read JSON file.
  - current_date_time(): Current UTC date/time.
  - current_time_english(): Current UTC time readable.

WorkOrderAgent: Work order and maintenance reasoning.
  - get_work_orders(asset_name): Retrieve work orders.
  - get_preventive_work_orders(asset_name): Preventive WOs.
  - get_corrective_work_orders(asset_name): Corrective WOs.
  - get_events(asset_name): Maintenance events/alerts.
  - get_failure_codes(asset_name): Failure codes.
  - get_work_order_distribution(asset_name): WO statistics.
  - predict_next_work_order(asset_name): Predict next WO.
  - analyze_alert_to_failure(asset_name): Alert-failure correlation.
\"\"\"

INFORMED_PROMPT = \"\"\"You are an expert planner for industrial asset operations. Given a question and available tools, produce a structured plan.

{tool_descriptions}

OUTPUT FORMAT:
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules: Use ONLY listed agents/tools. Keep plans concise. Use {{step_N}} for dependencies.

QUESTION: {question}
\"\"\"

BLIND_PROMPT = \"\"\"You are an expert planner for industrial asset operations. Produce a structured plan.

OUTPUT FORMAT:
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules: Keep plans concise. Use {{step_N}} for dependencies.

QUESTION: {question}
\"\"\"


def parse_plan(text):
    \"\"\"Parse a plan into structured steps with agent, tool, and args.\"\"\"
    steps = []
    if not text: return steps
    for block in re.split(r'(?=#Task\\d+:)', text):
        block = block.strip()
        task_m = re.search(r'#Task(\\d+):\\s*(.*)', block)
        agent_m = re.search(r'#Agent\\d+:\\s*(\\S+)', block)
        tool_m = re.search(r'#Tool\\d+:\\s*(\\S+)', block)
        args_m = re.search(r'#Args\\d+:\\s*(.*)', block)
        if task_m:
            # Try to parse args as JSON
            args_raw = args_m.group(1).strip() if args_m else "{}"
            try:
                args_parsed = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args_parsed = {}
            steps.append({
                "step": int(task_m.group(1)),
                "task": task_m.group(2).strip(),
                "agent": agent_m.group(1).strip() if agent_m else "",
                "tool": tool_m.group(1).strip().rstrip("()") if tool_m else "",
                "args": args_parsed,
                "args_raw": args_raw,
            })
    return steps


def evaluate_plan(gen_text, gold_text):
    \"\"\"Evaluate generated plan vs gold on routing, args, and text similarity.\"\"\"
    result = {"has_plan_format": False, "num_steps": 0, "valid_agents": 0,
              "total_agents": 0, "valid_tools": 0, "total_tools": 0,
              "agent_tool_f1": 0.0, "gold_steps": 0, "rouge_l": 0.0,
              "arg_key_f1": 0.0, "arg_value_match": 0.0}
    if "#Task" in gen_text and "#Agent" in gen_text:
        result["has_plan_format"] = True
    gen_steps = parse_plan(gen_text)
    gold_steps = parse_plan(gold_text)
    result["num_steps"] = len(gen_steps)
    result["gold_steps"] = len(gold_steps)

    # Agent & tool validity
    for s in gen_steps:
        result["total_agents"] += 1
        if s["agent"] in VALID_AGENTS: result["valid_agents"] += 1
        result["total_tools"] += 1
        if s["tool"] in VALID_TOOLS: result["valid_tools"] += 1

    # ── Tool-level set comparison ────────────────────────────────────
    gen_tools = [s["tool"] for s in gen_steps if s["tool"] and s["tool"] != "none"]
    gold_tools = [s["tool"] for s in gold_steps if s["tool"] and s["tool"] != "none"]
    gen_tool_set = set(gen_tools)
    gold_tool_set = set(gold_tools)

    result["tools_correct"] = list(gen_tool_set & gold_tool_set)   # in both
    result["tools_missing"] = list(gold_tool_set - gen_tool_set)   # in gold but not gen
    result["tools_extra"] = list(gen_tool_set - gold_tool_set)     # in gen but not gold
    result["tools_correct_n"] = len(result["tools_correct"])
    result["tools_missing_n"] = len(result["tools_missing"])
    result["tools_extra_n"] = len(result["tools_extra"])

    # Agent-Tool pair F1 (Jaccard)
    gen_pairs = {(s["agent"], s["tool"]) for s in gen_steps if s["agent"] and s["tool"]}
    gold_pairs = {(s["agent"], s["tool"]) for s in gold_steps if s["agent"] and s["tool"]}
    if gen_pairs or gold_pairs:
        result["agent_tool_f1"] = len(gen_pairs & gold_pairs) / len(gen_pairs | gold_pairs)

    # ── Argument evaluation ───────────────────────────────────────────
    gold_by_at = {}
    for s in gold_steps:
        key = (s["agent"], s["tool"])
        if key not in gold_by_at and s["args"]:
            gold_by_at[key] = s["args"]

    total_key_matches, total_keys = 0, 0
    total_val_matches, total_vals = 0, 0
    args_correct, args_missing, args_extra = [], [], []

    for s in gen_steps:
        key = (s["agent"], s["tool"])
        if key in gold_by_at and s["args"]:
            gold_args = gold_by_at[key]
            gen_args = s["args"]
            gold_keys = set(gold_args.keys())
            gen_keys = set(gen_args.keys())
            if gold_keys or gen_keys:
                matched = gold_keys & gen_keys
                total_key_matches += len(matched)
                total_keys += len(gold_keys | gen_keys)
                args_correct.extend(matched)
                args_missing.extend(gold_keys - gen_keys)
                args_extra.extend(gen_keys - gold_keys)
            shared_keys = gold_keys & gen_keys
            for k in shared_keys:
                total_vals += 1
                gv = str(gold_args[k]).strip().lower()
                ev = str(gen_args.get(k, "")).strip().lower()
                if gv == ev or ("{step_" in gv and "{step_" in ev):
                    total_val_matches += 1

    result["arg_key_f1"] = total_key_matches / total_keys if total_keys else 0.0
    result["arg_value_match"] = total_val_matches / total_vals if total_vals else 0.0
    result["args_correct"] = args_correct
    result["args_missing"] = args_missing   # in gold but model forgot
    result["args_extra"] = args_extra       # model hallucinated
    result["args_correct_n"] = len(args_correct)
    result["args_missing_n"] = len(args_missing)
    result["args_extra_n"] = len(args_extra)

    # ── Step count ────────────────────────────────────────────────────
    if result["gold_steps"] > 0:
        result["step_match"] = 1.0 if result["num_steps"] == result["gold_steps"] else 0.0
        result["step_ratio"] = result["num_steps"] / result["gold_steps"]
    else:
        result["step_match"] = 0.0
        result["step_ratio"] = 0.0

    # ROUGE-L
    try:
        r = rouge_metric.compute(predictions=[gen_text], references=[gold_text])
        result["rouge_l"] = r["rougeL"]
    except Exception:
        pass
    return result


def run_evaluation(model, tokenizer, scenarios, prompt_template, mode_name, tool_descriptions=""):
    results = []
    for sc in tqdm(scenarios, desc=f"Eval ({mode_name})"):
        if "{tool_descriptions}" in prompt_template:
            prompt = prompt_template.format(tool_descriptions=tool_descriptions, question=sc["question"])
        else:
            prompt = prompt_template.format(question=sc["question"])
        chat = [{"role": "user", "content": prompt}]
        tokenized = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True, return_dict=True)
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                        top_p=TOP_P, do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        metrics = evaluate_plan(generated, sc["gold_plan"])
        metrics.update({"id": sc["id"], "question": sc["question"], "type": sc.get("type", ""),
                        "generated": generated[:2000], "input_tokens": input_len,
                        "output_tokens": output_ids.shape[1] - input_len, "mode": mode_name})
        results.append(metrics)
    return results, summarize_results(results, mode_name)


def summarize_results(results, mode_name):
    n = len(results)
    if n == 0: return {"mode": mode_name, "total": 0}
    fmt_ok = sum(1 for r in results if r["has_plan_format"])
    return {
        "mode": mode_name, "total": n,
        "format_valid": fmt_ok, "format_valid_pct": 100 * fmt_ok / n,
        "avg_agent_tool_f1": np.mean([r["agent_tool_f1"] for r in results]),
        "avg_arg_key_f1": np.mean([r.get("arg_key_f1", 0) for r in results]),
        "avg_arg_value_match": np.mean([r.get("arg_value_match", 0) for r in results]),
        "avg_rouge_l": np.mean([r["rouge_l"] for r in results]),
        "avg_steps": np.mean([r["num_steps"] for r in results]),
        "avg_input_tokens": np.mean([r.get("input_tokens", 0) for r in results]),
        "avg_output_tokens": np.mean([r.get("output_tokens", 0) for r in results]),
        "agent_correctness": sum(r["valid_agents"] for r in results) / max(sum(r["total_agents"] for r in results), 1),
        "tool_correctness": sum(r["valid_tools"] for r in results) / max(sum(r["total_tools"] for r in results), 1),
        "step_exact_match": np.mean([r.get("step_match", 0) for r in results]),
        "avg_step_ratio": np.mean([r.get("step_ratio", 0) for r in results]),
        # Counts
        "total_tools_correct": sum(r.get("tools_correct_n", 0) for r in results),
        "total_tools_missing": sum(r.get("tools_missing_n", 0) for r in results),
        "total_tools_extra": sum(r.get("tools_extra_n", 0) for r in results),
        "total_args_correct": sum(r.get("args_correct_n", 0) for r in results),
        "total_args_missing": sum(r.get("args_missing_n", 0) for r in results),
        "total_args_extra": sum(r.get("args_extra_n", 0) for r in results),
    }


def print_summary(s):
    print(f"\\n{'='*55}")
    print(f"  {s['mode']} — {s['total']} scenarios")
    print(f"{'='*55}")
    print(f"  Format valid:    {s.get('format_valid',0)}/{s['total']} ({s.get('format_valid_pct',0):.1f}%)")
    print(f"  Agent-Tool F1:   {s.get('avg_agent_tool_f1',0):.3f}")
    print(f"  Arg key F1:      {s.get('avg_arg_key_f1',0):.3f}")
    print(f"  Arg value match: {s.get('avg_arg_value_match',0):.3f}")
    print(f"  ROUGE-L:         {s.get('avg_rouge_l',0):.3f}")
    print(f"  Agent correct:   {s.get('agent_correctness',0):.1%}")
    print(f"  Tool correct:    {s.get('tool_correctness',0):.1%}")
    print(f"  Step exact match: {s.get('step_exact_match',0):.1%}")
    print(f"  Avg step ratio:  {s.get('avg_step_ratio',0):.2f}x (1.0=perfect)")
    print(f"  Avg steps:       {s.get('avg_steps',0):.1f}")
    print(f"  Avg tokens in:   {s.get('avg_input_tokens',0):.0f}")
    tc = s.get('total_tools_correct',0)
    tm = s.get('total_tools_missing',0)
    te = s.get('total_tools_extra',0)
    print(f"  Tools:  {tc} correct, {tm} missing, {te} extra (hallucinated)")
    ac = s.get('total_args_correct',0)
    am = s.get('total_args_missing',0)
    ae = s.get('total_args_extra',0)
    print(f"  Args:   {ac} correct, {am} missing, {ae} extra (hallucinated)")"""))

    # ══════════════════════════════════════════════════════════════════
    # 5. LOAD MODEL
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 5. Load Base Model"))

    cells.append(code("""from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if LOAD_IN_8BIT:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    quant_label = "8-bit"
else:
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    quant_label = "4-bit NF4"

print(f"Loading {MODEL_ID} in {quant_label}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN, attn_implementation="eager")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
print(f"Loaded in {time.time()-t0:.1f}s, {torch.cuda.max_memory_allocated()/1e9:.1f} GB, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")"""))

    # ══════════════════════════════════════════════════════════════════
    # 6. BASELINE EVAL
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 6. Baseline Evaluation (No Fine-Tuning)

Test the base model on held-out scenarios in two modes:
- **Informed** — full tool descriptions in the prompt (the "easy" mode)
- **Blind** — no tool descriptions (the "hard" mode, what fine-tuning must fix)"""))

    cells.append(code("""print(f"Running baseline evaluation on {len(eval_scenarios)} scenarios...")

baseline_informed_results, baseline_informed_summary = run_evaluation(
    model, tokenizer, eval_scenarios, INFORMED_PROMPT, "Baseline: Informed",
    tool_descriptions=TOOL_DESCRIPTIONS)
print_summary(baseline_informed_summary)

baseline_blind_results, baseline_blind_summary = run_evaluation(
    model, tokenizer, eval_scenarios, BLIND_PROMPT, "Baseline: Blind")
print_summary(baseline_blind_summary)

token_overhead = baseline_informed_summary["avg_input_tokens"] - baseline_blind_summary["avg_input_tokens"]
print(f"\\nToken overhead from tool descriptions: {token_overhead:.0f} tokens ({100*token_overhead/baseline_informed_summary['avg_input_tokens']:.0f}% of informed prompt)")

# Show a few blind examples to see the hallucinated agents
print("\\nBlind mode examples (expect hallucinated agents):")
for r in baseline_blind_results[:3]:
    print(f"  ID {r['id']}: AT-F1={r['agent_tool_f1']:.2f}, {r['generated'][:150]}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 7. QLORA SETUP + TRAIN HELPER
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 7. QLoRA Setup & Training Helper

We define a reusable `train_model()` function so we can train both the generalist and specialist models."""))

    cells.append(code("""from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

def setup_lora(base_model):
    \"\"\"Prepare a model for QLoRA training. Returns a new PEFT model.\"\"\"
    m = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                              target_modules=LORA_TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM)
    m = get_peft_model(m, lora_config)
    t, total = m.get_nb_trainable_parameters()
    print(f"Trainable: {t:,} / {total:,} ({100*t/total:.2f}%)")
    return m


def train_model(peft_model, train_data, eval_data, output_dir, epochs=NUM_EPOCHS):
    \"\"\"Train a PEFT model on given data. Returns trainer.\"\"\"
    os.makedirs(output_dir, exist_ok=True)
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None

    total_steps = len(train_ds) * epochs // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    sft_kwargs = dict(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE, lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=warmup_steps, weight_decay=WEIGHT_DECAY, bf16=BF16,
        logging_steps=10, eval_strategy="steps" if eval_ds else "no",
        eval_steps=50 if eval_ds else None,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        load_best_model_at_end=bool(eval_ds), report_to="none", seed=SEED,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    # Handle max_seq_length version differences
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(model=peft_model, args=sft_config, train_dataset=train_ds,
                          processing_class=tokenizer)
    if eval_ds: trainer_kwargs["eval_dataset"] = eval_ds
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "max_seq_length" in trainer_params and "max_seq_length" not in sft_kwargs:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    trainer = SFTTrainer(**trainer_kwargs)
    print(f"Training: {len(train_ds)} examples, {epochs} epochs, ~{total_steps} steps")
    t0 = time.time()
    trainer.train()
    print(f"Done in {(time.time()-t0)/60:.1f} min, final loss: {trainer.state.log_history[-1].get('train_loss', trainer.state.log_history[-1].get('loss', '?'))}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer

print("Training helper ready.")"""))

    # ══════════════════════════════════════════════════════════════════
    # 8. PREPARE TRAINING DATA (3-TIER: GENERALIST / PLANNER / SPECIALISTS)
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 8. Prepare Training Data — 3-Tier Architecture

We train three types of models:

1. **Generalist** — one model trained on everything (baseline for comparison)
2. **Planner** — specialized in routing: given a question, output which agents/tools/dependencies to use (no args)
3. **Specialists** (per agent) — given a step's task + agent + tool, output the correct arguments

```
Question → [Planner] → routing plan → [Specialist per step] → full plan with args
```

This way, the planner focuses on *what to do* and specialists focus on *how to do it*."""))

    cells.append(code("""def get_agents_in_plan(plan_text):
    return [a for a in re.findall(r'#Agent\\d+:\\s*(\\S+)', plan_text) if a not in ('none', '')]

def strip_plan_to_routing(plan_text):
    \"\"\"Strip a plan to routing-only (Task/Agent/Tool/Dependency). Remove Args/ExpectedOutput.\"\"\"
    lines = []
    for line in plan_text.split("\\n"):
        stripped = line.strip()
        if any(stripped.startswith(f"#{tag}") for tag in ["Task", "Agent", "Tool", "Dependency"]):
            if not stripped.startswith("#Args") and not stripped.startswith("#ExpectedOutput"):
                lines.append(line)
        elif stripped == "":
            lines.append("")
    return "\\n".join(lines).strip()

def extract_specialist_steps(plan_text):
    \"\"\"Extract per-step specialist examples: input=(task, agent, tool) -> output=(args, expected).\"\"\"
    steps = []
    for block in re.split(r'(?=#Task\\d+:)', plan_text):
        block = block.strip()
        if not block: continue
        task_m = re.search(r'#Task\\d+:\\s*(.*)', block)
        agent_m = re.search(r'#Agent\\d+:\\s*(\\S+)', block)
        tool_m = re.search(r'#Tool\\d+:\\s*(\\S+)', block)
        args_m = re.search(r'#Args\\d+:\\s*(.*)', block)
        exp_m = re.search(r'#ExpectedOutput\\d+:\\s*(.*)', block)
        dep_m = re.search(r'#Dependency\\d+:\\s*(.*)', block)
        if task_m and agent_m and tool_m:
            agent = agent_m.group(1).strip()
            if agent in ('none', ''): continue
            instruction = f"Generate the arguments for this tool call:\\nTask: {task_m.group(1).strip()}\\nAgent: {agent}\\nTool: {tool_m.group(1).strip()}\\nDependency: {dep_m.group(1).strip() if dep_m else 'None'}"
            response = f"#Args: {args_m.group(1).strip() if args_m else '{}'}\\n#ExpectedOutput: {exp_m.group(1).strip() if exp_m else 'Result'}"
            steps.append({"agent": agent, "instruction": instruction, "response": response})
    return steps

# ── 1. Generalist data (everything) ──────────────────────────────
all_train = [{"messages": d["messages"]} for d in clean_tool + clean_plan + clean_exec]
random.shuffle(all_train)
if MAX_TRAIN_EXAMPLES:
    all_train = all_train[:MAX_TRAIN_EXAMPLES]
sp = int(len(all_train) * 0.95)
generalist_train, generalist_eval = all_train[:sp], all_train[sp:]

# ── 2. Planner data (routing-only plans) ─────────────────────────
planner_data = []
for d in clean_plan:
    if d.get("metadata", {}).get("category") != "planning": continue
    routing = strip_plan_to_routing(d["messages"][1]["content"])
    if "#Task" in routing and "#Agent" in routing:
        planner_data.append({"messages": [
            {"role": "user", "content": d["messages"][0]["content"]},
            {"role": "assistant", "content": routing},
        ]})
# Also add tool knowledge (helps planner learn agent/tool associations)
for d in clean_tool:
    planner_data.append({"messages": d["messages"]})

random.shuffle(planner_data)
if MAX_TRAIN_EXAMPLES:
    planner_data = planner_data[:MAX_TRAIN_EXAMPLES]
sp = int(len(planner_data) * 0.95)
planner_train, planner_eval = planner_data[:sp], planner_data[sp:]

# ── 3. Specialist data (per-agent step-level arg generation) ─────
specialist_data = defaultdict(list)
for d in clean_plan:
    if d.get("metadata", {}).get("category") != "planning": continue
    for step in extract_specialist_steps(d["messages"][1]["content"]):
        specialist_data[step["agent"]].append({"messages": [
            {"role": "user", "content": step["instruction"]},
            {"role": "assistant", "content": step["response"]},
        ]})

print(f"Training data prepared:")
print(f"  Generalist:  {len(generalist_train)} train, {len(generalist_eval)} eval")
print(f"  Planner:     {len(planner_train)} train, {len(planner_eval)} eval")
print(f"  Specialists:")
for agent, data in sorted(specialist_data.items()):
    print(f"    {agent}: {len(data)} step-level examples")"""))

    # ══════════════════════════════════════════════════════════════════
    # 9. TRAIN GENERALIST
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 9. Train Generalist Model\n\nOne model trained on ALL data — the baseline for comparison."))

    cells.append(code("""generalist_dir = f"{OUTPUT_DIR}/generalist"
peft_generalist = setup_lora(model)
gen_trainer = train_model(peft_generalist, generalist_train, generalist_eval, generalist_dir)

# Plot loss
log_h = gen_trainer.state.log_history
fig, ax = plt.subplots(figsize=(10, 4))
tl = [(h["step"], h["loss"]) for h in log_h if "loss" in h]
el = [(h["step"], h["eval_loss"]) for h in log_h if "eval_loss" in h]
if tl: ax.plot(*zip(*tl), label="Train", alpha=0.7)
if el: ax.plot(*zip(*el), label="Eval", lw=2)
ax.set(xlabel="Step", ylabel="Loss", title="Generalist Training"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

    # ══════════════════════════════════════════════════════════════════
    # 10. EVALUATE GENERALIST
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 10. Evaluate Generalist"))

    cells.append(code("""gen_blind_results, gen_blind_summary = run_evaluation(
    peft_generalist, tokenizer, eval_scenarios, BLIND_PROMPT, "Generalist: Blind")
print_summary(gen_blind_summary)

gen_informed_results, gen_informed_summary = run_evaluation(
    peft_generalist, tokenizer, eval_scenarios, INFORMED_PROMPT, "Generalist: Informed",
    tool_descriptions=TOOL_DESCRIPTIONS)
print_summary(gen_informed_summary)

for r in gen_blind_results[:3]:
    print(f"\\n--- ID {r['id']} ({r['type']}): {r['question'][:55]}... ---")
    print(f"  AT-F1={r['agent_tool_f1']:.2f} ROUGE={r['rouge_l']:.2f} Steps={r['num_steps']}(gold:{r['gold_steps']})")
    print(f"  {r['generated'][:200]}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 11. TRAIN PLANNER
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 11. Train Planner Model

The planner is specialized in **routing**: given a question, it outputs which agents, tools, and dependencies to use — but NOT the arguments. This is a simpler task than full plan generation.

The planner's output looks like:
```
#Task1: Retrieve assets for MAIN site
#Agent1: IoTAgent
#Tool1: assets
#Dependency1: None
```
No `#Args` or `#ExpectedOutput` — those come from the specialist."""))

    cells.append(code("""# Free generalist from GPU
del peft_generalist; torch.cuda.empty_cache()

planner_dir = f"{OUTPUT_DIR}/planner"
peft_planner = setup_lora(model)
planner_trainer = train_model(peft_planner, planner_train, planner_eval, planner_dir)"""))

    # ══════════════════════════════════════════════════════════════════
    # 12. EVALUATE PLANNER
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 12. Evaluate Planner (Routing Accuracy)"))

    cells.append(code("""# Evaluate planner on routing quality (AT-F1, agent/tool correctness)
# The planner generates routing-only plans, so we compare agent-tool pairs
planner_blind_results, planner_blind_summary = run_evaluation(
    peft_planner, tokenizer, eval_scenarios, BLIND_PROMPT, "Planner: Blind")
print_summary(planner_blind_summary)

# Show planner outputs (should be shorter, no args)
for r in planner_blind_results[:3]:
    print(f"\\n--- ID {r['id']}: {r['question'][:55]}... ---")
    print(f"  AT-F1={r['agent_tool_f1']:.2f}, Steps={r['num_steps']}(gold:{r['gold_steps']})")
    print(f"  {r['generated'][:250]}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 13. TRAIN SPECIALISTS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 13. Train Specialist Models (Per-Agent Arg Generation)

Each specialist is trained on step-level examples: given a task description + agent + tool name, it generates the correct arguments.

In production: Planner outputs the routing → each step is sent to the matching specialist → specialist fills in args."""))

    cells.append(code("""del peft_planner; torch.cuda.empty_cache()

specialist_models = {}
for agent_name, data in sorted(specialist_data.items()):
    if len(data) < 20:
        print(f"Skipping {agent_name}: {len(data)} examples (need >=20)")
        continue
    print(f"\\n{'='*50}")
    print(f"  Specialist: {agent_name} ({len(data)} examples)")
    print(f"{'='*50}")
    spec_model = setup_lora(model)
    random.shuffle(data)
    if MAX_TRAIN_EXAMPLES:
        data = data[:min(len(data), MAX_TRAIN_EXAMPLES)]
    sp = int(len(data) * 0.9)
    spec_dir = f"{OUTPUT_DIR}/specialist_{agent_name}"
    train_model(spec_model, data[:sp], data[sp:] if sp < len(data) else None, spec_dir)
    specialist_models[agent_name] = spec_model

print(f"\\nTrained {len(specialist_models)} specialists: {list(specialist_models.keys())}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 14. EVALUATE PLANNER + SPECIALIST PIPELINE
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 14. Evaluate Planner + Specialist Pipeline

The full pipeline: Planner generates routing → each step is sent to the matching specialist for arg generation → combine into final plan.

We evaluate the *combined output* against the gold plan."""))

    cells.append(code("""# For each eval scenario, run the planner then specialists
pipeline_results = []

for sc in tqdm(eval_scenarios, desc="Eval (Planner+Specialist pipeline)"):
    # Step 1: Planner generates routing
    planner_r = [r for r in planner_blind_results if r["id"] == sc["id"]]
    if not planner_r:
        continue
    planner_output = planner_r[0]["generated"]
    planner_steps = parse_plan(planner_output)

    # Step 2: For each step, ask the specialist to fill in args
    full_plan_lines = []
    for step in planner_steps:
        agent = step["agent"]
        tool = step["tool"]

        if agent in specialist_models:
            spec = specialist_models[agent]
            spec_prompt = f"Generate the arguments for this tool call:\\nTask: {step['task']}\\nAgent: {agent}\\nTool: {tool}\\nDependency: None"
            chat = [{"role": "user", "content": spec_prompt}]
            tokenized = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True, return_dict=True)
            input_ids = tokenized["input_ids"].to(spec.device)
            attention_mask = tokenized["attention_mask"].to(spec.device)
            with torch.no_grad():
                out = spec.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_new_tokens=256, temperature=TEMPERATURE, top_p=TOP_P,
                                     do_sample=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            spec_output = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            args_m = re.search(r'#Args:\\s*(.*)', spec_output)
            exp_m = re.search(r'#ExpectedOutput:\\s*(.*)', spec_output)
            args = args_m.group(1).strip() if args_m else "{}"
            expected = exp_m.group(1).strip() if exp_m else ""
        else:
            args, expected = "{}", ""

        full_plan_lines.append(f"#Task{step['step']}: {step['task']}")
        full_plan_lines.append(f"#Agent{step['step']}: {agent}")
        full_plan_lines.append(f"#Tool{step['step']}: {tool}")
        full_plan_lines.append(f"#Args{step['step']}: {args}")
        full_plan_lines.append(f"#Dependency{step['step']}: None")
        if expected:
            full_plan_lines.append(f"#ExpectedOutput{step['step']}: {expected}")
        full_plan_lines.append("")

    combined_plan = "\\n".join(full_plan_lines)
    metrics = evaluate_plan(combined_plan, sc["gold_plan"])
    metrics.update({"id": sc["id"], "question": sc["question"], "type": sc.get("type", ""),
                    "generated": combined_plan[:2000], "input_tokens": 0, "output_tokens": 0,
                    "mode": "Pipeline: Planner+Specialist"})
    pipeline_results.append(metrics)

pipeline_summary = summarize_results(pipeline_results, "Pipeline: Planner+Specialist")
print_summary(pipeline_summary)"""))

    # ══════════════════════════════════════════════════════════════════
    # 15. FULL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 15. Full Comparison — All Approaches"))

    cells.append(code("""all_summaries = [
    baseline_informed_summary,
    baseline_blind_summary,
    gen_informed_summary,
    gen_blind_summary,
    planner_blind_summary,
    pipeline_summary,
]

comp = pd.DataFrame(all_summaries)
display_cols = {
    "mode": "Mode", "format_valid_pct": "Format%", "avg_agent_tool_f1": "AT-F1",
    "avg_arg_key_f1": "ArgKey", "avg_arg_value_match": "ArgVal",
    "step_exact_match": "StepEM", "avg_step_ratio": "StepR",
    "avg_rouge_l": "ROUGE", "agent_correctness": "Agent%",
    "tool_correctness": "Tool%", "avg_steps": "Steps", "avg_input_tokens": "TokIn",
}
comp = comp[[c for c in display_cols if c in comp.columns]]
comp.columns = [display_cols[c] for c in comp.columns]

print("\\n" + "=" * 110)
print("  FULL COMPARISON: 6 Approaches")
print("=" * 110)
print(comp.to_string(index=False, float_format="%.3f"))

print(f"\\nKey results:")
print(f"  Baseline blind AT-F1:         {baseline_blind_summary['avg_agent_tool_f1']:.3f}")
print(f"  Generalist blind AT-F1:       {gen_blind_summary['avg_agent_tool_f1']:.3f}")
print(f"  Planner-only blind AT-F1:     {planner_blind_summary['avg_agent_tool_f1']:.3f}")
print(f"  Planner+Specialist AT-F1:     {pipeline_summary['avg_agent_tool_f1']:.3f}")"""))

    cells.append(code("""# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
modes = ["Base\\nInformed", "Base\\nBlind", "Gen\\nInformed", "Gen\\nBlind", "Planner\\nBlind", "Pipeline\\nP+S"]
colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899"]

for ax, metric, title in [
    (axes[0], "format_valid_pct", "Format Valid (%)"),
    (axes[1], "avg_agent_tool_f1", "Agent-Tool F1"),
    (axes[2], "agent_correctness", "Agent Correctness"),
]:
    vals = [s.get(metric, 0) for s in all_summaries]
    ax.bar(modes, vals, color=colors)
    ax.set_title(title, fontsize=11)
    for i, v in enumerate(vals):
        fmt = f"{v:.0f}%" if "pct" in metric else f"{v:.3f}"
        ax.text(i, v + max(vals)*0.02, fmt, ha="center", fontweight="bold", fontsize=7)

plt.suptitle(f"AssetOpsBench: {MODEL_ID} — All Approaches Compared", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/full_comparison.png", dpi=150, bbox_inches="tight")
plt.show()"""))

    # ══════════════════════════════════════════════════════════════════
    # 16. PER-DOMAIN ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 16. Per-Domain Analysis"))

    cells.append(code("""gen_by_id = {r["id"]: r for r in gen_blind_results}
plan_by_id = {r["id"]: r for r in planner_blind_results}
pipe_by_id = {r["id"]: r for r in pipeline_results}

rows = []
for sc in eval_scenarios:
    sid = sc["id"]
    rows.append({
        "id": sid, "type": sc["type"],
        "gen_atf1": gen_by_id.get(sid, {}).get("agent_tool_f1", 0),
        "planner_atf1": plan_by_id.get(sid, {}).get("agent_tool_f1", 0),
        "pipeline_atf1": pipe_by_id.get(sid, {}).get("agent_tool_f1", 0),
        "gen_rouge": gen_by_id.get(sid, {}).get("rouge_l", 0),
        "pipeline_rouge": pipe_by_id.get(sid, {}).get("rouge_l", 0),
    })

df = pd.DataFrame(rows)
print("Per-type comparison:")
type_comp = df.groupby("type").agg(
    gen_atf1=("gen_atf1", "mean"), planner_atf1=("planner_atf1", "mean"),
    pipeline_atf1=("pipeline_atf1", "mean"), count=("id", "count"),
).round(3)
print(type_comp.to_string())

# Which approach wins per type?
for t in type_comp.index:
    best_col = type_comp.loc[t, ["gen_atf1", "planner_atf1", "pipeline_atf1"]].idxmax()
    best_val = type_comp.loc[t, best_col]
    print(f"  {t}: best={best_col} ({best_val:.3f})")"""))

    # ══════════════════════════════════════════════════════════════════
    # 17. TOKEN SAVINGS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 17. Token Savings"))

    cells.append(code("""informed_tok = baseline_informed_summary["avg_input_tokens"]
blind_tok = gen_blind_summary["avg_input_tokens"]
savings = informed_tok - blind_tok

print("=" * 60)
print("  TOKEN SAVINGS ANALYSIS")
print("=" * 60)
print(f"  Informed prompt:     {informed_tok:.0f} tokens (with tool descriptions)")
print(f"  Blind prompt:        {blind_tok:.0f} tokens (no tool descriptions)")
print(f"  Savings per query:   {savings:.0f} tokens ({100*savings/informed_tok:.0f}%)")
print(f"  Per 152 scenarios:   {savings*152:,.0f} tokens saved")
print(f"  Per 1000 queries/day: {savings*1000:,.0f} tokens/day")

# The argument for specialist models
print(f"\\n  Production architecture argument:")
print(f"  - Planner model handles routing (small, fast)")
print(f"  - Specialist models handle args (domain-specific, can run in parallel)")
print(f"  - Combined latency can be LOWER than one generalist pass")
print(f"  - Each specialist stays small and focused")"""))

    # ══════════════════════════════════════════════════════════════════
    # 18. SAVE
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 18. Save Results"))

    cells.append(code("""results = {
    "config": {"model": MODEL_ID, "lora_r": LORA_R, "epochs": NUM_EPOCHS,
               "lr": LEARNING_RATE, "held_out": NUM_HELD_OUT, "timestamp": datetime.now().isoformat()},
    "summaries": {s["mode"]: s for s in all_summaries},
    "token_savings": {"informed": informed_tok, "blind": blind_tok, "savings": savings},
    "per_scenario": rows,
}
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
df.to_csv(f"{OUTPUT_DIR}/per_scenario.csv", index=False)
print(f"Saved to {OUTPUT_DIR}/")

print(f"\\n{'='*60}")
print(f"  EXPERIMENT COMPLETE")
print(f"{'='*60}")
for s in all_summaries:
    print(f"  {s['mode']:40s} AT-F1={s.get('avg_agent_tool_f1',0):.3f} Agent={s.get('agent_correctness',0):.1%}")
print(f"\\n  Token savings: {savings:.0f}/query ({100*savings/informed_tok:.0f}%)")"""))

    cells.append(md("## 19. (Optional) Download"))

    cells.append(code("""os.system("cd /content && zip -r /content/assetops_results.zip output/")
try:
    from google.colab import files
    files.download("/content/assetops_results.zip")
except ImportError:
    print("Results at /content/output/")"""))

    # ══════════════════════════════════════════════════════════════════
    # ASSEMBLE NOTEBOOK
    # ══════════════════════════════════════════════════════════════════
    for cell in cells:
        if isinstance(cell["source"], list):
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in cell["source"]]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    out = Path(__file__).parent / "AssetOpsBench_Gemma_FineTuning.ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook: {out} ({len(nb['cells'])} cells)")
