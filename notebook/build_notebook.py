#!/usr/bin/env python3
"""Generate the end-to-end Colab notebook as .ipynb.

Usage:
    python notebook/build_notebook.py
    # Produces: notebook/AssetOpsBench_Gemma_FineTuning.ipynb
"""

import json
from pathlib import Path


def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n") if "\n" in source else [source],
    }


def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n") if "\n" in source else [source],
        "outputs": [],
        "execution_count": None,
    }


def build_notebook():
    cells = []

    # ══════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""# Internalizing MCP Tool Knowledge in Small LLMs
## End-to-End: Baseline → QLoRA Fine-Tuning → Evaluation

**HPML Group 20 — Columbia University, Spring 2026**

This notebook runs the full experiment pipeline on **1x H100 (80 GB)**:

1. **Baseline evaluation** — test a base Gemma model on AssetOpsBench planning tasks in *informed* mode (with tool descriptions) and *blind* mode (no descriptions)
2. **QLoRA fine-tuning** — train the model on our 3 datasets (tool knowledge, planning, execution) to internalize tool descriptions
3. **Post-training evaluation** — re-test in both modes to measure improvement
4. **Analysis** — compare metrics, compute token savings, visualize results

**Research question:** *Can a small LLM internalize tool descriptions well enough to plan without them in the prompt, and how much prompt overhead does this save?*

---
**Runtime:** ~2-3 hours total on H100 (baseline ~20 min, training ~60-90 min, post-eval ~20 min)"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 0: SETUP
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 0. Setup & Installation"""))

    cells.append(code("""%%capture
!pip install -q torch torchvision torchaudio
!pip install -q transformers>=4.46.0 peft>=0.13.0 trl>=0.12.0
!pip install -q bitsandbytes>=0.44.0 accelerate>=1.0.0
!pip install -q datasets evaluate rouge-score
!pip install -q pandas matplotlib seaborn tqdm
!pip install -q sentencepiece protobuf
print("All packages installed.")"""))

    cells.append(code("""import os
import json
import re
import time
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Seed everything
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
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 1. Configuration

All hyperparameters in one place. Change the model ID here to experiment with different models."""))

    cells.append(code("""# ── Model ─────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3-4b-it"  # 4B params, fits easily on H100
# Alternatives:
# MODEL_ID = "google/gemma-3-12b-it"  # 12B dense, still fits on H100 with QLoRA
# MODEL_ID = "google/gemma-3-1b-it"   # 1B, fastest for debugging

# HuggingFace token (required for Gemma gated models)
# Get yours at https://huggingface.co/settings/tokens
HF_TOKEN = ""  # <-- PASTE YOUR TOKEN HERE
if not HF_TOKEN:
    from google.colab import userdata
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
        print(f"Loaded HF_TOKEN from Colab secrets")
    except Exception:
        print("WARNING: No HF_TOKEN found. Set it above or in Colab secrets.")
        print("You need to accept the Gemma license at https://huggingface.co/google/gemma-3-4b-it")

# ── GitHub Repo ───────────────────────────────────────────────────────
REPO_URL = "https://github.com/YuvalShemla/hpml-2026-project.git"
REPO_DIR = "/content/hpml-2026-project"

# ── QLoRA ─────────────────────────────────────────────────────────────
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch size = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
BF16 = True  # H100 supports bf16 natively

# ── Evaluation ────────────────────────────────────────────────────────
EVAL_SCENARIOS = 50        # number of scenarios to evaluate (max 152)
MAX_NEW_TOKENS = 1024      # max tokens for plan generation
TEMPERATURE = 0.1          # low temperature for deterministic plans
TOP_P = 0.9

# ── Output ────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/output"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"QLoRA: r={LORA_R}, alpha={LORA_ALPHA}, target={len(LORA_TARGET_MODULES)} modules")
print(f"Training: lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, effective_batch={PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Eval: {EVAL_SCENARIOS} scenarios, max_tokens={MAX_NEW_TOKENS}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: CLONE REPO & LOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 2. Clone Repository & Load Datasets"""))

    cells.append(code("""# Clone the repo
if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"Cloned to {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

# Verify datasets exist
datasets_dir = Path(REPO_DIR) / "benchmark" / "generate_data" / "datasets"
for fname in ["tool_knowledge.jsonl", "planning.jsonl", "execution.jsonl"]:
    path = datasets_dir / fname
    if path.exists():
        with open(path) as f:
            count = sum(1 for _ in f)
        print(f"  {fname}: {count} examples")
    else:
        print(f"  {fname}: NOT FOUND")"""))

    cells.append(code("""def load_jsonl(path):
    \"\"\"Load a JSONL file into a list of dicts.\"\"\"
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load all datasets
datasets_dir = Path(REPO_DIR) / "benchmark" / "generate_data" / "datasets"

ds_tool = load_jsonl(datasets_dir / "tool_knowledge.jsonl")
ds_plan = load_jsonl(datasets_dir / "planning.jsonl")
ds_exec = load_jsonl(datasets_dir / "execution.jsonl")

# Load gold plans for evaluation reference
gold_path = Path(REPO_DIR) / "benchmark" / "baseline_tests" / "gemini_flash_informed_results.json"
with open(gold_path) as f:
    gold_plans_raw = json.load(f)
gold_plans = {g["id"]: g for g in gold_plans_raw if g.get("plan_steps", 0) > 0}

print(f"\\nLoaded datasets:")
print(f"  Tool knowledge: {len(ds_tool)} examples")
print(f"  Planning:       {len(ds_plan)} examples")
print(f"  Execution:      {len(ds_exec)} examples")
print(f"  Gold plans:     {len(gold_plans)} (Gemini 2.5 Flash reference)")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: DATA EXPLORATION
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 3. Data Exploration"""))

    cells.append(code("""# Category breakdown
all_data = ds_tool + ds_plan + ds_exec
cats = Counter(d.get("metadata", {}).get("category", "?") for d in all_data)
sources = Counter(d.get("metadata", {}).get("source", "deterministic") for d in all_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Category distribution
cat_df = pd.DataFrame(cats.most_common(), columns=["Category", "Count"])
axes[0].barh(cat_df["Category"], cat_df["Count"], color="steelblue")
axes[0].set_title("Training Data by Category")
axes[0].set_xlabel("Count")

# Source distribution
src_df = pd.DataFrame(sources.most_common(), columns=["Source", "Count"])
axes[1].barh(src_df["Source"], src_df["Count"], color="coral")
axes[1].set_title("Training Data by Source")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.show()

# Show samples from each dataset
for name, ds in [("Tool Knowledge", ds_tool), ("Planning", ds_plan), ("Execution", ds_exec)]:
    sample = random.choice(ds)
    print(f"\\n{'='*60}")
    print(f"  Sample from {name}")
    print(f"{'='*60}")
    print(f"  User: {sample['messages'][0]['content'][:200]}")
    print(f"  Asst: {sample['messages'][1]['content'][:300]}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: LOAD SCENARIOS FOR EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 4. Load Evaluation Scenarios

We load the AssetOpsBench scenarios from HuggingFace and select a subset for evaluation. Each scenario has a question and a gold plan from Gemini 2.5 Flash."""))

    cells.append(code("""from datasets import load_dataset

hf_ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
hf_scenarios = [dict(row) for row in hf_ds["train"]]

# Build evaluation set: scenarios that have gold plans
eval_scenarios = []
for sc in hf_scenarios:
    if sc["id"] in gold_plans:
        eval_scenarios.append({
            "id": sc["id"],
            "question": sc["text"],
            "type": sc["type"],
            "category": sc["category"],
            "gold_plan": gold_plans[sc["id"]]["response"],
            "gold_steps": gold_plans[sc["id"]]["plan_steps"],
            "gold_agents": gold_plans[sc["id"]].get("agents_used", []),
            "gold_tools": gold_plans[sc["id"]].get("tools_used", []),
        })

# Limit to EVAL_SCENARIOS, ensuring diversity across types
random.shuffle(eval_scenarios)
eval_scenarios = eval_scenarios[:EVAL_SCENARIOS]

type_dist = Counter(s["type"] for s in eval_scenarios)
print(f"Evaluation set: {len(eval_scenarios)} scenarios")
print(f"Type distribution: {dict(type_dist)}")
print(f"\\nSample scenario:")
print(f"  Q: {eval_scenarios[0]['question']}")
print(f"  Gold steps: {eval_scenarios[0]['gold_steps']}")
print(f"  Gold agents: {eval_scenarios[0]['gold_agents']}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: TOOL DESCRIPTION PROMPT
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 5. Prompt Templates

The **informed prompt** includes full tool descriptions (~2,268 tokens of overhead).
The **blind prompt** includes only the question (~20-50 tokens).

After fine-tuning, the model should produce correct plans even with the blind prompt."""))

    cells.append(code("""# Tool descriptions (from AssetOpsBench MCP servers)
TOOL_DESCRIPTIONS = \"\"\"Available Agents and Tools:

IoTAgent: Handles IoT telemetry data access.
  - sites(): List all available IoT sites. Returns list of site names.
  - assets(site_name): List all asset IDs for a given site.
  - sensors(site_name, asset_id): List sensor names for a specific asset at a site.
  - history(site_name, asset_id, start, final?): Fetch historical sensor readings for a time range.

FMSRAgent: Provides failure mode analysis and sensor relevance reasoning.
  - get_failure_modes(asset_name): Return known failure modes for an asset type.
  - get_failure_mode_sensor_mapping(asset_name, failure_modes, sensors): Determine sensor relevancy for failure modes.

TSFMAgent: Time-series forecasting and anomaly detection.
  - get_ai_tasks(): List supported AI task types for time-series analysis.
  - get_tsfm_models(): List available pre-trained TinyTimeMixer model checkpoints.
  - run_tsfm_forecasting(dataset_path, timestamp_column, target_columns, ...): Zero-shot forecasting.
  - run_tsfm_finetuning(dataset_path, timestamp_column, target_columns, ...): Few-shot fine-tuning.
  - run_tsad(dataset_path, tsfm_output_json, timestamp_column, target_columns, ...): Anomaly detection.
  - run_integrated_tsad(dataset_path, timestamp_column, target_columns, ...): End-to-end forecasting + anomaly detection.

Utilities: General utility functions.
  - json_reader(file_name): Read and parse a JSON file.
  - current_date_time(): Return current UTC date and time.
  - current_time_english(): Return current UTC time as human-readable string.

WorkOrderAgent: Work order and maintenance event reasoning.
  - get_work_orders(asset_name, start_date?, end_date?): Retrieve work orders for an asset.
  - get_preventive_work_orders(asset_name): Retrieve preventive maintenance work orders.
  - get_corrective_work_orders(asset_name): Retrieve corrective work orders.
  - get_events(asset_name): Retrieve maintenance events and alerts.
  - get_failure_codes(asset_name): Retrieve failure codes from work orders.
  - get_work_order_distribution(asset_name, group_by?): Get work order distribution statistics.
  - predict_next_work_order(asset_name): Predict when next work order is needed.
  - analyze_alert_to_failure(asset_name): Analyze alert-to-failure correlations.
\"\"\"

INFORMED_PROMPT_TEMPLATE = \"\"\"You are an expert planner for industrial asset operations. Given a question and available tools, produce a structured plan.

{tool_descriptions}

OUTPUT FORMAT (use exactly this structure):
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Use ONLY agents and tools from the list above
- Keep plans concise (minimum steps needed)
- Use {{step_N}} placeholders when a later step needs results from an earlier step
- If a question references workorders/maintenance history, use WorkOrderAgent

QUESTION: {question}
\"\"\"

BLIND_PROMPT_TEMPLATE = \"\"\"You are an expert planner for industrial asset operations. Given a question, produce a structured plan using the available MCP tool agents.

OUTPUT FORMAT (use exactly this structure):
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Keep plans concise (minimum steps needed)
- Use {{step_N}} placeholders when a later step needs results from an earlier step

QUESTION: {question}
\"\"\"

print(f"Informed prompt overhead: ~{len(TOOL_DESCRIPTIONS.split())} words (~{len(TOOL_DESCRIPTIONS) // 4} tokens est.)")
print(f"Blind prompt: no tool descriptions")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: EVALUATION FRAMEWORK
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 6. Evaluation Framework

Metrics:
- **Plan format validity** — does the output contain `#Task/#Agent/#Tool` tags?
- **Agent correctness** — are all agents valid?
- **Tool correctness** — are all tools valid?
- **Agent-Tool pair F1** — Jaccard overlap with gold plan's (agent, tool) pairs
- **Step count comparison** — over/under decomposition vs gold
- **ROUGE-L** — textual similarity of plan to gold"""))

    cells.append(code("""VALID_AGENTS = {
    "IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities",
    "WorkOrderAgent", "VibrationAgent", "none",
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
    "none",
}


def parse_plan(text):
    \"\"\"Parse a plan string into structured steps.\"\"\"
    steps = []
    if not text:
        return steps
    blocks = re.split(r'(?=#Task\\d+:)', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        task_m = re.search(r'#Task(\\d+):\\s*(.*)', block)
        agent_m = re.search(r'#Agent\\d+:\\s*(\\S+)', block)
        tool_m = re.search(r'#Tool\\d+:\\s*(\\S+)', block)
        if task_m:
            steps.append({
                "step": int(task_m.group(1)),
                "task": task_m.group(2).strip(),
                "agent": agent_m.group(1).strip() if agent_m else "",
                "tool": (tool_m.group(1).strip().rstrip("()") if tool_m else ""),
            })
    return steps


def evaluate_plan(generated_text, gold_plan_text):
    \"\"\"Evaluate a generated plan against gold reference.\"\"\"
    result = {
        "has_plan_format": False,
        "num_steps": 0,
        "valid_agents": 0,
        "total_agents": 0,
        "valid_tools": 0,
        "total_tools": 0,
        "agent_tool_f1": 0.0,
        "gold_steps": 0,
        "step_ratio": 0.0,
    }

    # Check format
    if "#Task" in generated_text and "#Agent" in generated_text:
        result["has_plan_format"] = True

    # Parse
    gen_steps = parse_plan(generated_text)
    gold_steps = parse_plan(gold_plan_text)

    result["num_steps"] = len(gen_steps)
    result["gold_steps"] = len(gold_steps)

    if gold_steps:
        result["step_ratio"] = len(gen_steps) / len(gold_steps) if gold_steps else 0

    # Agent & tool validity
    for s in gen_steps:
        result["total_agents"] += 1
        if s["agent"] in VALID_AGENTS:
            result["valid_agents"] += 1
        result["total_tools"] += 1
        if s["tool"] in VALID_TOOLS:
            result["valid_tools"] += 1

    # Agent-Tool pair F1 (Jaccard)
    gen_pairs = {(s["agent"], s["tool"]) for s in gen_steps if s["agent"] and s["tool"]}
    gold_pairs = {(s["agent"], s["tool"]) for s in gold_steps if s["agent"] and s["tool"]}
    if gen_pairs or gold_pairs:
        intersection = gen_pairs & gold_pairs
        union = gen_pairs | gold_pairs
        result["agent_tool_f1"] = len(intersection) / len(union) if union else 0.0

    return result


def run_evaluation(model, tokenizer, scenarios, prompt_template, mode_name, tool_descriptions=""):
    \"\"\"Run evaluation on a set of scenarios and return results.\"\"\"
    results = []
    format_ok = 0
    agent_tool_f1s = []

    for i, sc in enumerate(tqdm(scenarios, desc=f"Eval ({mode_name})")):
        # Build prompt
        if "{tool_descriptions}" in prompt_template:
            prompt = prompt_template.format(
                tool_descriptions=tool_descriptions,
                question=sc["question"],
            )
        else:
            prompt = prompt_template.format(question=sc["question"])

        # Tokenize
        chat = [{"role": "user", "content": prompt}]
        tokenized = tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        input_len = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        # Evaluate
        metrics = evaluate_plan(generated, sc["gold_plan"])
        metrics["id"] = sc["id"]
        metrics["question"] = sc["question"]
        metrics["generated"] = generated[:2000]
        metrics["input_tokens"] = input_len
        metrics["output_tokens"] = output_ids.shape[1] - input_len
        metrics["mode"] = mode_name
        results.append(metrics)

        if metrics["has_plan_format"]:
            format_ok += 1
        agent_tool_f1s.append(metrics["agent_tool_f1"])

    # Summary
    n = len(results)
    summary = {
        "mode": mode_name,
        "total": n,
        "format_valid": format_ok,
        "format_valid_pct": 100 * format_ok / n if n else 0,
        "avg_agent_tool_f1": np.mean(agent_tool_f1s) if agent_tool_f1s else 0,
        "avg_steps": np.mean([r["num_steps"] for r in results]),
        "avg_input_tokens": np.mean([r["input_tokens"] for r in results]),
        "avg_output_tokens": np.mean([r["output_tokens"] for r in results]),
        "agent_correctness": (
            sum(r["valid_agents"] for r in results) /
            max(sum(r["total_agents"] for r in results), 1)
        ),
        "tool_correctness": (
            sum(r["valid_tools"] for r in results) /
            max(sum(r["total_tools"] for r in results), 1)
        ),
    }

    return results, summary


def print_summary(summary):
    \"\"\"Pretty-print evaluation summary.\"\"\"
    print(f"\\n{'='*55}")
    print(f"  {summary['mode']} — {summary['total']} scenarios")
    print(f"{'='*55}")
    print(f"  Plan format valid:   {summary['format_valid']}/{summary['total']} ({summary['format_valid_pct']:.1f}%)")
    print(f"  Agent-Tool pair F1:  {summary['avg_agent_tool_f1']:.3f}")
    print(f"  Agent correctness:   {summary['agent_correctness']:.1%}")
    print(f"  Tool correctness:    {summary['tool_correctness']:.1%}")
    print(f"  Avg steps generated: {summary['avg_steps']:.1f}")
    print(f"  Avg input tokens:    {summary['avg_input_tokens']:.0f}")
    print(f"  Avg output tokens:   {summary['avg_output_tokens']:.0f}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: LOAD BASE MODEL
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 7. Load Base Model (4-bit Quantized)"""))

    cells.append(code("""from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID} in 4-bit...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    attn_implementation="eager",  # sdpa can cause issues with some models during generation
)

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

elapsed = time.time() - t0
mem_used = torch.cuda.max_memory_allocated() / 1e9
print(f"Model loaded in {elapsed:.1f}s")
print(f"GPU memory used: {mem_used:.1f} GB")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: BASELINE — INFORMED MODE
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 8. Baseline Evaluation — Informed Mode (with tool descriptions)

This is the "easy" mode — the model gets full tool descriptions in the prompt. Even without fine-tuning, a good instruction-tuned model should produce some valid plans here."""))

    cells.append(code("""baseline_informed_results, baseline_informed_summary = run_evaluation(
    model, tokenizer, eval_scenarios,
    INFORMED_PROMPT_TEMPLATE, "Baseline: Informed",
    tool_descriptions=TOOL_DESCRIPTIONS,
)
print_summary(baseline_informed_summary)

# Show a few examples
for r in baseline_informed_results[:3]:
    print(f"\\n--- ID {r['id']}: {r['question'][:60]}... ---")
    print(f"  Format OK: {r['has_plan_format']}, Steps: {r['num_steps']} (gold: {r['gold_steps']}), AT-F1: {r['agent_tool_f1']:.2f}")
    print(f"  Generated: {r['generated'][:200]}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9: BASELINE — BLIND MODE
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 9. Baseline Evaluation — Blind Mode (no tool descriptions)

This is the "hard" mode — no tool descriptions in the prompt. The model must rely on its pre-trained knowledge alone. We expect ~0% valid plans here, which is the gap fine-tuning must close."""))

    cells.append(code("""baseline_blind_results, baseline_blind_summary = run_evaluation(
    model, tokenizer, eval_scenarios,
    BLIND_PROMPT_TEMPLATE, "Baseline: Blind",
)
print_summary(baseline_blind_summary)

# Show a few examples
for r in baseline_blind_results[:3]:
    print(f"\\n--- ID {r['id']}: {r['question'][:60]}... ---")
    print(f"  Format OK: {r['has_plan_format']}, Steps: {r['num_steps']}")
    print(f"  Generated: {r['generated'][:200]}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 10: BASELINE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 10. Baseline Results Summary"""))

    cells.append(code("""# Side-by-side comparison
baseline_df = pd.DataFrame([baseline_informed_summary, baseline_blind_summary])
baseline_df = baseline_df[["mode", "format_valid_pct", "avg_agent_tool_f1", "agent_correctness",
                           "tool_correctness", "avg_steps", "avg_input_tokens"]]
baseline_df.columns = ["Mode", "Format Valid %", "AT-F1", "Agent Correct", "Tool Correct", "Avg Steps", "Avg Input Tokens"]
print("\\nBaseline Results:")
print(baseline_df.to_string(index=False))

# Token overhead
informed_tokens = baseline_informed_summary["avg_input_tokens"]
blind_tokens = baseline_blind_summary["avg_input_tokens"]
overhead = informed_tokens - blind_tokens
print(f"\\nToken overhead from tool descriptions: {overhead:.0f} tokens/query")
print(f"That is {100 * overhead / informed_tokens:.0f}% of the informed prompt")

# Save baseline
with open(f"{RESULTS_DIR}/baseline_results.json", "w") as f:
    json.dump({
        "informed": {"summary": baseline_informed_summary, "results": baseline_informed_results},
        "blind": {"summary": baseline_blind_summary, "results": baseline_blind_results},
    }, f, indent=2, default=str)
print(f"\\nSaved to {RESULTS_DIR}/baseline_results.json")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 11: QLORA SETUP
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 11. QLoRA Fine-Tuning Setup

We apply LoRA adapters to all linear layers in the model, keeping the base model frozen in 4-bit. Only the adapters (~2% of parameters) are trained."""))

    cells.append(code("""from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
print(f"Trainable size: {trainable * 2 / 1e9:.2f} GB (fp16)")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 12: PREPARE TRAINING DATA
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 12. Prepare Training Data

We combine all three datasets and format them for the SFTTrainer. The chat template handles proper tokenization with the model's special tokens."""))

    cells.append(code("""from datasets import Dataset

def format_for_sft(examples):
    \"\"\"Convert our JSONL format to the chat format SFTTrainer expects.\"\"\"
    formatted = []
    for ex in examples:
        messages = ex["messages"]
        formatted.append({
            "messages": messages,
        })
    return formatted

# Combine datasets with curriculum ordering:
# 1. Tool knowledge (teaches tool taxonomy)
# 2. Planning (teaches plan generation)
# 3. Execution (teaches execution structure)
train_data = format_for_sft(ds_tool) + format_for_sft(ds_plan) + format_for_sft(ds_exec)

# Shuffle within each curriculum stage, but keep stages ordered
tool_end = len(ds_tool)
plan_end = tool_end + len(ds_plan)

# Actually, for SFTTrainer we should shuffle everything — curriculum ordering
# is handled by the natural gradient flow over epochs
random.shuffle(train_data)

# Split off a small eval set (5%)
split_idx = int(len(train_data) * 0.95)
train_split = train_data[:split_idx]
eval_split = train_data[split_idx:]

# Convert to HF Dataset
train_dataset = Dataset.from_list(train_split)
eval_dataset = Dataset.from_list(eval_split)

print(f"Training examples: {len(train_dataset)}")
print(f"Eval examples:     {len(eval_dataset)}")
print(f"\\nSample training example:")
sample = train_dataset[0]
print(f"  User: {sample['messages'][0]['content'][:150]}...")
print(f"  Asst: {sample['messages'][1]['content'][:150]}...")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 13: TRAINING
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 13. Fine-Tuning with SFTTrainer

Training with TRL's SFTTrainer using our QLoRA config. The model learns to:
1. Recognize tool names and their ownership
2. Route questions to the correct agent
3. Generate structured plans with proper format
4. Handle clarification and abstention"""))

    cells.append(code("""from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir=ADAPTER_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    bf16=BF16,
    max_seq_length=MAX_SEQ_LENGTH,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    seed=SEED,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print(f"Training config:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {PER_DEVICE_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Total steps: ~{len(train_dataset) * NUM_EPOCHS // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")"""))

    cells.append(code("""# Train!
print("Starting training...")
t0 = time.time()

train_result = trainer.train()

elapsed = time.time() - t0
print(f"\\nTraining complete in {elapsed / 60:.1f} minutes")
print(f"Final train loss: {train_result.training_loss:.4f}")

# Save the adapter
trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}")

# Plot training loss
log_history = trainer.state.log_history
train_losses = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
eval_losses = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(10, 4))
if train_losses:
    steps, losses = zip(*train_losses)
    ax.plot(steps, losses, label="Train Loss", alpha=0.7)
if eval_losses:
    steps, losses = zip(*eval_losses)
    ax.plot(steps, losses, label="Eval Loss", linewidth=2)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 14: POST-TRAINING EVAL — BLIND
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 14. Post-Training Evaluation — Blind Mode

**This is the key test.** After fine-tuning, can the model produce valid plans *without* tool descriptions in the prompt? If yes, we've successfully internalized the tool knowledge."""))

    cells.append(code("""# The model already has LoRA adapters merged from training
# Just run evaluation
finetuned_blind_results, finetuned_blind_summary = run_evaluation(
    model, tokenizer, eval_scenarios,
    BLIND_PROMPT_TEMPLATE, "Fine-tuned: Blind",
)
print_summary(finetuned_blind_summary)

# Show examples
for r in finetuned_blind_results[:5]:
    gold_steps = [s for s in eval_scenarios if s["id"] == r["id"]]
    gold_n = gold_steps[0]["gold_steps"] if gold_steps else "?"
    print(f"\\n--- ID {r['id']}: {r['question'][:60]}... ---")
    print(f"  Format: {'VALID' if r['has_plan_format'] else 'INVALID'}, Steps: {r['num_steps']} (gold: {gold_n}), AT-F1: {r['agent_tool_f1']:.2f}")
    print(f"  Output: {r['generated'][:250]}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 15: POST-TRAINING EVAL — INFORMED
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 15. Post-Training Evaluation — Informed Mode

Does fine-tuning maintain (or improve) performance when tool descriptions ARE provided?"""))

    cells.append(code("""finetuned_informed_results, finetuned_informed_summary = run_evaluation(
    model, tokenizer, eval_scenarios,
    INFORMED_PROMPT_TEMPLATE, "Fine-tuned: Informed",
    tool_descriptions=TOOL_DESCRIPTIONS,
)
print_summary(finetuned_informed_summary)"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 16: COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 16. Full Comparison — Before vs After Fine-Tuning"""))

    cells.append(code("""# Build comparison table
all_summaries = [
    baseline_informed_summary,
    baseline_blind_summary,
    finetuned_informed_summary,
    finetuned_blind_summary,
]

comparison = pd.DataFrame(all_summaries)
comparison = comparison[["mode", "format_valid_pct", "avg_agent_tool_f1",
                         "agent_correctness", "tool_correctness", "avg_steps", "avg_input_tokens"]]
comparison.columns = ["Mode", "Format Valid %", "AT-F1", "Agent Acc", "Tool Acc", "Avg Steps", "Avg Tokens In"]

# Format nicely
comparison["Format Valid %"] = comparison["Format Valid %"].map("{:.1f}%".format)
comparison["AT-F1"] = comparison["AT-F1"].map("{:.3f}".format)
comparison["Agent Acc"] = comparison["Agent Acc"].map("{:.1%}".format)
comparison["Tool Acc"] = comparison["Tool Acc"].map("{:.1%}".format)
comparison["Avg Steps"] = comparison["Avg Steps"].map("{:.1f}".format)
comparison["Avg Tokens In"] = comparison["Avg Tokens In"].map("{:.0f}".format)

print("\\n" + "=" * 90)
print("  FULL COMPARISON")
print("=" * 90)
print(comparison.to_string(index=False))
print()

# Key metric: blind mode improvement
blind_before = baseline_blind_summary["format_valid_pct"]
blind_after = finetuned_blind_summary["format_valid_pct"]
print(f"BLIND MODE IMPROVEMENT: {blind_before:.1f}% -> {blind_after:.1f}% (+{blind_after - blind_before:.1f}pp)")

atf1_before = baseline_blind_summary["avg_agent_tool_f1"]
atf1_after = finetuned_blind_summary["avg_agent_tool_f1"]
print(f"BLIND AT-F1 IMPROVEMENT: {atf1_before:.3f} -> {atf1_after:.3f} (+{atf1_after - atf1_before:.3f})")"""))

    cells.append(code("""# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

modes = ["Baseline\\nInformed", "Baseline\\nBlind", "Fine-tuned\\nInformed", "Fine-tuned\\nBlind"]
colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"]

# Format valid %
vals = [s["format_valid_pct"] for s in all_summaries]
axes[0].bar(modes, vals, color=colors)
axes[0].set_title("Plan Format Validity (%)")
axes[0].set_ylim(0, 105)
for i, v in enumerate(vals):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

# Agent-Tool F1
vals = [s["avg_agent_tool_f1"] for s in all_summaries]
axes[1].bar(modes, vals, color=colors)
axes[1].set_title("Agent-Tool Pair F1")
axes[1].set_ylim(0, 1.05)
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

# Input tokens (shows the savings)
vals = [s["avg_input_tokens"] for s in all_summaries]
axes[2].bar(modes, vals, color=colors)
axes[2].set_title("Avg Input Tokens per Query")
for i, v in enumerate(vals):
    axes[2].text(i, v + 10, f"{v:.0f}", ha="center", fontweight="bold")

plt.suptitle(f"AssetOpsBench: {MODEL_ID} — Before vs After Fine-Tuning", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Chart saved to {RESULTS_DIR}/comparison_chart.png")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 17: TOKEN SAVINGS
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 17. Token Savings Analysis

The whole point: if the fine-tuned model can plan in blind mode, we save ~2,268 tokens of tool descriptions per query."""))

    cells.append(code("""informed_tokens = baseline_informed_summary["avg_input_tokens"]
blind_tokens = baseline_blind_summary["avg_input_tokens"]
token_savings = informed_tokens - blind_tokens

ft_blind_quality = finetuned_blind_summary["format_valid_pct"]
ft_blind_atf1 = finetuned_blind_summary["avg_agent_tool_f1"]

print("=" * 60)
print("  TOKEN SAVINGS ANALYSIS")
print("=" * 60)
print(f"  Informed prompt tokens: {informed_tokens:.0f}")
print(f"  Blind prompt tokens:    {blind_tokens:.0f}")
print(f"  Savings per query:      {token_savings:.0f} tokens ({100 * token_savings / informed_tokens:.0f}%)")
print()
print(f"  For 152 scenarios:      {token_savings * 152:,.0f} tokens saved")
print(f"  For 1000 queries/day:   {token_savings * 1000:,.0f} tokens/day saved")
print()
print(f"  Fine-tuned blind quality:")
print(f"    Format valid: {ft_blind_quality:.1f}%")
print(f"    AT-F1:        {ft_blind_atf1:.3f}")
print()

# Quality-adjusted savings
if ft_blind_quality > 50:
    print(f"  The fine-tuned model achieves {ft_blind_quality:.0f}% valid plans WITHOUT")
    print(f"  tool descriptions, saving {token_savings:.0f} tokens per query.")
    print(f"  This validates the core hypothesis.")
else:
    remaining_gap = 100 - ft_blind_quality
    print(f"  The fine-tuned model achieves {ft_blind_quality:.0f}% in blind mode.")
    print(f"  Remaining gap: {remaining_gap:.0f}pp. Consider:")
    print(f"    - More training data (current: {len(train_data)} examples)")
    print(f"    - Larger model (current: {MODEL_ID})")
    print(f"    - More epochs (current: {NUM_EPOCHS})")
    print(f"    - Curriculum training (staged datasets)")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 18: PER-SCENARIO ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 18. Per-Scenario Analysis"""))

    cells.append(code("""# Build per-scenario comparison
per_scenario = []
for i, sc in enumerate(eval_scenarios):
    row = {
        "id": sc["id"],
        "type": sc["type"],
        "question": sc["question"][:50] + "...",
        "gold_steps": sc["gold_steps"],
        "base_informed_ok": baseline_informed_results[i]["has_plan_format"],
        "base_blind_ok": baseline_blind_results[i]["has_plan_format"],
        "ft_informed_ok": finetuned_informed_results[i]["has_plan_format"],
        "ft_blind_ok": finetuned_blind_results[i]["has_plan_format"],
        "ft_blind_atf1": finetuned_blind_results[i]["agent_tool_f1"],
    }
    per_scenario.append(row)

per_df = pd.DataFrame(per_scenario)

# Show where blind mode improved
improved = per_df[per_df["ft_blind_ok"] & ~per_df["base_blind_ok"]]
print(f"Scenarios where blind mode went from FAIL to PASS after fine-tuning: {len(improved)}")
if len(improved) > 0:
    print(improved[["id", "type", "question", "ft_blind_atf1"]].head(10).to_string(index=False))

# Show where blind mode still fails
still_fails = per_df[~per_df["ft_blind_ok"]]
print(f"\\nScenarios still failing in blind mode: {len(still_fails)}")
if len(still_fails) > 0:
    type_dist = still_fails["type"].value_counts()
    print(f"  By type: {dict(type_dist)}")

# Success rate by scenario type
type_results = per_df.groupby("type").agg(
    ft_blind_pass=("ft_blind_ok", "mean"),
    ft_blind_atf1=("ft_blind_atf1", "mean"),
    count=("id", "count"),
).round(3)
print(f"\\nFine-tuned blind mode results by scenario type:")
print(type_results.to_string())"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 19: SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 19. Save All Results"""))

    cells.append(code("""# Save comprehensive results
final_results = {
    "config": {
        "model": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "eval_scenarios": EVAL_SCENARIOS,
        "train_examples": len(train_data),
        "timestamp": datetime.now().isoformat(),
    },
    "summaries": {
        "baseline_informed": baseline_informed_summary,
        "baseline_blind": baseline_blind_summary,
        "finetuned_informed": finetuned_informed_summary,
        "finetuned_blind": finetuned_blind_summary,
    },
    "token_savings": {
        "informed_avg_tokens": informed_tokens,
        "blind_avg_tokens": blind_tokens,
        "savings_per_query": token_savings,
        "savings_pct": 100 * token_savings / informed_tokens if informed_tokens else 0,
    },
    "per_scenario": per_scenario,
}

results_path = f"{RESULTS_DIR}/full_results.json"
with open(results_path, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"Full results saved to {results_path}")

# Also save as CSV for easy analysis
per_df.to_csv(f"{RESULTS_DIR}/per_scenario.csv", index=False)
print(f"Per-scenario CSV saved to {RESULTS_DIR}/per_scenario.csv")

# Print final summary
print("\\n" + "=" * 60)
print("  EXPERIMENT COMPLETE")
print("=" * 60)
print(f"  Model:     {MODEL_ID}")
print(f"  Training:  {len(train_data)} examples, {NUM_EPOCHS} epochs")
print(f"  Evaluated: {EVAL_SCENARIOS} scenarios")
print()
print(f"  BASELINE (blind):     {baseline_blind_summary['format_valid_pct']:.1f}% valid, AT-F1={baseline_blind_summary['avg_agent_tool_f1']:.3f}")
print(f"  FINE-TUNED (blind):   {finetuned_blind_summary['format_valid_pct']:.1f}% valid, AT-F1={finetuned_blind_summary['avg_agent_tool_f1']:.3f}")
print(f"  Token savings:        {token_savings:.0f} tokens/query ({100 * token_savings / informed_tokens:.0f}%)")
print()
print(f"  Adapter: {ADAPTER_DIR}")
print(f"  Results: {RESULTS_DIR}")"""))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 20: OPTIONAL — DOWNLOAD ADAPTER
    # ══════════════════════════════════════════════════════════════════════
    cells.append(md("""## 20. (Optional) Download Adapter & Results

Run this cell to download the trained LoRA adapter and results to your local machine."""))

    cells.append(code("""# Zip adapter + results for download
os.system("cd /content && zip -r /content/assetops_gemma_results.zip output/")

# In Colab, this triggers a download
try:
    from google.colab import files
    files.download("/content/assetops_gemma_results.zip")
except ImportError:
    print("Not in Colab. Find results at /content/output/")"""))

    # ══════════════════════════════════════════════════════════════════════
    # BUILD NOTEBOOK
    # ══════════════════════════════════════════════════════════════════════
    # Fix source formatting — ensure each line ends with \n
    for cell in cells:
        if isinstance(cell["source"], list):
            cell["source"] = [
                line if line.endswith("\n") else line + "\n"
                for line in cell["source"]
            ]
            # Remove trailing \n from last line
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
            "accelerator": "GPU",
            "gpuClass": "standard",
            "colab": {
                "provenance": [],
                "gpuType": "H100",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


if __name__ == "__main__":
    nb = build_notebook()
    output_path = Path(__file__).parent / "AssetOpsBench_Gemma_FineTuning.ipynb"
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook written to {output_path}")
    print(f"Cells: {len(nb['cells'])}")
