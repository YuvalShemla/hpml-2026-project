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
2. **Baseline results** — hardcoded from prior Gemma 3 run (informed vs blind mode)
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
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")"""))

    # ══════════════════════════════════════════════════════════════════
    # 1. CONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 1. Configuration"))

    cells.append(code("""# ── Model ─────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-E4B-it"  # 4.5B params (2.3B effective)
# MODEL_ID = "google/gemma-3-4b-it"  # fallback if Gemma 4 has issues
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
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# ── Training ──────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
BF16 = True

# ── Evaluation ────────────────────────────────────────────────
NUM_HELD_OUT = 50  # scenarios held out for eval (stratified by type)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.9

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"QLoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Training: lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, eff_batch={PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Held-out eval scenarios: {NUM_HELD_OUT}")"""))

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
    steps = []
    if not text: return steps
    for block in re.split(r'(?=#Task\\d+:)', text):
        block = block.strip()
        task_m = re.search(r'#Task(\\d+):\\s*(.*)', block)
        agent_m = re.search(r'#Agent\\d+:\\s*(\\S+)', block)
        tool_m = re.search(r'#Tool\\d+:\\s*(\\S+)', block)
        if task_m:
            steps.append({
                "step": int(task_m.group(1)),
                "task": task_m.group(2).strip(),
                "agent": agent_m.group(1).strip() if agent_m else "",
                "tool": tool_m.group(1).strip().rstrip("()") if tool_m else "",
            })
    return steps


def evaluate_plan(gen_text, gold_text):
    result = {"has_plan_format": False, "num_steps": 0, "valid_agents": 0,
              "total_agents": 0, "valid_tools": 0, "total_tools": 0,
              "agent_tool_f1": 0.0, "gold_steps": 0, "rouge_l": 0.0}
    if "#Task" in gen_text and "#Agent" in gen_text:
        result["has_plan_format"] = True
    gen_steps = parse_plan(gen_text)
    gold_steps = parse_plan(gold_text)
    result["num_steps"] = len(gen_steps)
    result["gold_steps"] = len(gold_steps)
    for s in gen_steps:
        result["total_agents"] += 1
        if s["agent"] in VALID_AGENTS: result["valid_agents"] += 1
        result["total_tools"] += 1
        if s["tool"] in VALID_TOOLS: result["valid_tools"] += 1
    gen_pairs = {(s["agent"], s["tool"]) for s in gen_steps if s["agent"] and s["tool"]}
    gold_pairs = {(s["agent"], s["tool"]) for s in gold_steps if s["agent"] and s["tool"]}
    if gen_pairs or gold_pairs:
        result["agent_tool_f1"] = len(gen_pairs & gold_pairs) / len(gen_pairs | gold_pairs)
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
        "avg_rouge_l": np.mean([r["rouge_l"] for r in results]),
        "avg_steps": np.mean([r["num_steps"] for r in results]),
        "avg_input_tokens": np.mean([r["input_tokens"] for r in results]),
        "avg_output_tokens": np.mean([r["output_tokens"] for r in results]),
        "agent_correctness": sum(r["valid_agents"] for r in results) / max(sum(r["total_agents"] for r in results), 1),
        "tool_correctness": sum(r["valid_tools"] for r in results) / max(sum(r["total_tools"] for r in results), 1),
    }


def print_summary(s):
    print(f"\\n{'='*55}")
    print(f"  {s['mode']} — {s['total']} scenarios")
    print(f"{'='*55}")
    print(f"  Format valid:   {s.get('format_valid',0)}/{s['total']} ({s.get('format_valid_pct',0):.1f}%)")
    print(f"  Agent-Tool F1:  {s.get('avg_agent_tool_f1',0):.3f}")
    print(f"  ROUGE-L:        {s.get('avg_rouge_l',0):.3f}")
    print(f"  Agent correct:  {s.get('agent_correctness',0):.1%}")
    print(f"  Tool correct:   {s.get('tool_correctness',0):.1%}")
    print(f"  Avg steps:      {s.get('avg_steps',0):.1f}")
    print(f"  Avg tokens in:  {s.get('avg_input_tokens',0):.0f}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 5. BASELINE (HARDCODED)
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 5. Baseline Results (Pre-computed, Gemma 3 4B)

Baseline from prior run on `google/gemma-3-4b-it` (4-bit, 50 scenarios).

**Key finding:** Blind mode produces valid *format* (100%) but completely hallucinated agents ("Data Retrieval Agent", "MCP Query Tool") — **0% correct agents/tools**. The format is learned from pre-training; the actual tool catalog is not."""))

    cells.append(code("""# Hardcoded from prior Gemma 3 run (avoids re-running ~40 min)
baseline_informed_summary = {
    "mode": "Baseline: Informed (Gemma 3)",
    "total": 50, "format_valid": 43, "format_valid_pct": 86.0,
    "avg_agent_tool_f1": 0.237, "avg_rouge_l": 0.0, "avg_steps": 3.8,
    "avg_input_tokens": 796.4, "avg_output_tokens": 320.0,
    "agent_correctness": 0.742, "tool_correctness": 0.953,
}
baseline_blind_summary = {
    "mode": "Baseline: Blind (Gemma 3)",
    "total": 50, "format_valid": 50, "format_valid_pct": 100.0,
    "avg_agent_tool_f1": 0.0, "avg_rouge_l": 0.0, "avg_steps": 3.22,
    "avg_input_tokens": 162.4, "avg_output_tokens": 303.0,
    "agent_correctness": 0.0, "tool_correctness": 0.0,
}

print_summary(baseline_informed_summary)
print_summary(baseline_blind_summary)

token_overhead = baseline_informed_summary["avg_input_tokens"] - baseline_blind_summary["avg_input_tokens"]
print(f"\\nToken overhead from tool descriptions: {token_overhead:.0f} tokens ({100*token_overhead/baseline_informed_summary['avg_input_tokens']:.0f}% of informed prompt)")"""))

    # ══════════════════════════════════════════════════════════════════
    # 6. LOAD MODEL
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 6. Load Base Model"))

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
    # Gemma 3 token_type_ids fix
    if "gemma-3" in MODEL_ID:
        _orig = m.forward
        def _patched(*args, **kwargs):
            if "input_ids" in kwargs and "token_type_ids" not in kwargs:
                kwargs["token_type_ids"] = torch.zeros_like(kwargs["input_ids"])
            return _orig(*args, **kwargs)
        m.forward = _patched
        print("Applied Gemma 3 token_type_ids patch")
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
    # 8. PREPARE TRAINING DATA
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 8. Prepare Training Data"))

    cells.append(code("""# Combine clean data for generalist model
all_train = [{"messages": d["messages"]} for d in clean_tool + clean_plan + clean_exec]
random.shuffle(all_train)
split_idx = int(len(all_train) * 0.95)
generalist_train = all_train[:split_idx]
generalist_eval = all_train[split_idx:]

print(f"Generalist training: {len(generalist_train)}, eval: {len(generalist_eval)}")

# Split planning data by primary agent for specialist models
def get_primary_agent(plan_text):
    agents = [a for a in re.findall(r'#Agent\\d+:\\s*(\\S+)', plan_text) if a not in ('none', '')]
    return agents[0] if agents else None

specialist_data = {"IoTAgent": [], "FMSRAgent": [], "TSFMAgent": [], "WorkOrderAgent": []}
for d in clean_plan:
    if d.get("metadata", {}).get("category") != "planning":
        continue
    agent = get_primary_agent(d["messages"][1]["content"])
    if agent in specialist_data:
        specialist_data[agent].append({"messages": d["messages"]})

# Add tool knowledge to each specialist (filtered to their tools)
agent_tool_keywords = {
    "IoTAgent": ["sites", "assets", "sensors", "history", "IoTAgent", "IoT", "telemetry"],
    "FMSRAgent": ["failure_mode", "sensor_mapping", "FMSRAgent", "FMSR", "failure"],
    "TSFMAgent": ["tsfm", "forecast", "anomaly", "TSFMAgent", "time-series", "TTM"],
    "WorkOrderAgent": ["work_order", "maintenance", "WorkOrderAgent", "corrective", "preventive"],
}
for agent, keywords in agent_tool_keywords.items():
    for d in clean_tool:
        text = d["messages"][0]["content"] + " " + d["messages"][1]["content"]
        if any(kw.lower() in text.lower() for kw in keywords):
            specialist_data[agent].append({"messages": d["messages"]})

print(f"\\nSpecialist training data:")
for agent, data in specialist_data.items():
    print(f"  {agent}: {len(data)} examples")"""))

    # ══════════════════════════════════════════════════════════════════
    # 9. TRAIN GENERALIST
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 9. Train Generalist Model

One model trained on ALL tool knowledge + planning + execution data."""))

    cells.append(code("""generalist_dir = f"{OUTPUT_DIR}/generalist"
peft_model = setup_lora(model)
generalist_trainer = train_model(peft_model, generalist_train, generalist_eval, generalist_dir)"""))

    cells.append(code("""# Plot generalist training loss
log_h = generalist_trainer.state.log_history
train_losses = [(h["step"], h["loss"]) for h in log_h if "loss" in h]
eval_losses = [(h["step"], h["eval_loss"]) for h in log_h if "eval_loss" in h]
fig, ax = plt.subplots(figsize=(10, 4))
if train_losses:
    ax.plot(*zip(*train_losses), label="Train", alpha=0.7)
if eval_losses:
    ax.plot(*zip(*eval_losses), label="Eval", lw=2)
ax.set(xlabel="Step", ylabel="Loss", title="Generalist Training Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()"""))

    # ══════════════════════════════════════════════════════════════════
    # 10. EVALUATE GENERALIST
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 10. Evaluate Generalist (Blind Mode — No Tool Descriptions)"))

    cells.append(code("""gen_blind_results, gen_blind_summary = run_evaluation(
    peft_model, tokenizer, eval_scenarios, BLIND_PROMPT, "Generalist: Blind")
print_summary(gen_blind_summary)

# Show examples
for r in gen_blind_results[:3]:
    print(f"\\n--- ID {r['id']} ({r['type']}): {r['question'][:55]}... ---")
    print(f"  AT-F1: {r['agent_tool_f1']:.2f}, ROUGE-L: {r['rouge_l']:.2f}, Steps: {r['num_steps']} (gold: {r['gold_steps']})")
    print(f"  Output: {r['generated'][:200]}")"""))

    cells.append(code("""# Also test generalist in informed mode (should be at least as good)
gen_informed_results, gen_informed_summary = run_evaluation(
    peft_model, tokenizer, eval_scenarios, INFORMED_PROMPT, "Generalist: Informed",
    tool_descriptions=TOOL_DESCRIPTIONS)
print_summary(gen_informed_summary)"""))

    # ══════════════════════════════════════════════════════════════════
    # 11. TRAIN SPECIALISTS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 11. Train Specialist Models

Four domain-specific models, each trained only on its agent's data.
The hypothesis: specialists outperform the generalist on their domain because they don't need to handle the full tool catalog.

In production, a lightweight router (or the generalist) picks the agent, then the specialist generates the detailed plan."""))

    cells.append(code("""# We need to reload base model for each specialist (LoRA adapters are different)
# First, unload the generalist adapter
del peft_model
torch.cuda.empty_cache()

specialist_trainers = {}
specialist_models = {}

for agent_name, data in specialist_data.items():
    if len(data) < 20:
        print(f"\\nSkipping {agent_name}: only {len(data)} examples (need >=20)")
        continue
    print(f"\\n{'='*50}")
    print(f"  Training specialist: {agent_name} ({len(data)} examples)")
    print(f"{'='*50}")

    # Fresh LoRA on base model
    spec_model = setup_lora(model)
    random.shuffle(data)
    sp = int(len(data) * 0.9)
    spec_dir = f"{OUTPUT_DIR}/specialist_{agent_name}"

    spec_trainer = train_model(spec_model, data[:sp], data[sp:] if sp < len(data) else None, spec_dir, epochs=NUM_EPOCHS)
    specialist_trainers[agent_name] = spec_trainer
    specialist_models[agent_name] = spec_model

print(f"\\nTrained {len(specialist_models)} specialist models: {list(specialist_models.keys())}")"""))

    # ══════════════════════════════════════════════════════════════════
    # 12. EVALUATE SPECIALISTS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("""## 12. Evaluate Specialist Models

For each scenario, we route it to the specialist that matches its primary agent.
Scenarios requiring multiple agents use the specialist for the first agent in the gold plan."""))

    cells.append(code("""# Route each eval scenario to the right specialist
specialist_results = []
specialist_hit = 0
specialist_miss = 0

for sc in tqdm(eval_scenarios, desc="Eval (Specialists)"):
    # Determine which specialist should handle this
    gold_agents = [a for a in sc.get("gold_agents", []) if a not in ("none", "")]
    primary_agent = gold_agents[0] if gold_agents else None

    if primary_agent and primary_agent in specialist_models:
        spec_model = specialist_models[primary_agent]
        specialist_hit += 1
    else:
        # Fall back to first available specialist or skip
        spec_model = list(specialist_models.values())[0] if specialist_models else None
        specialist_miss += 1

    if spec_model is None:
        continue

    # Run inference
    prompt = BLIND_PROMPT.format(question=sc["question"])
    chat = [{"role": "user", "content": prompt}]
    tokenized = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True, return_dict=True)
    input_ids = tokenized["input_ids"].to(spec_model.device)
    attention_mask = tokenized["attention_mask"].to(spec_model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = spec_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                          top_p=TOP_P, do_sample=True,
                                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    metrics = evaluate_plan(generated, sc["gold_plan"])
    metrics.update({"id": sc["id"], "question": sc["question"], "type": sc.get("type", ""),
                    "generated": generated[:2000], "input_tokens": input_len,
                    "output_tokens": output_ids.shape[1] - input_len, "mode": "Specialist: Blind",
                    "routed_to": primary_agent})
    specialist_results.append(metrics)

spec_blind_summary = summarize_results(specialist_results, "Specialist: Blind")
print(f"\\nRouting: {specialist_hit} matched specialist, {specialist_miss} fell back")
print_summary(spec_blind_summary)"""))

    # ══════════════════════════════════════════════════════════════════
    # 13. FULL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 13. Full Comparison — Baseline vs Generalist vs Specialists"))

    cells.append(code("""all_summaries = [
    baseline_informed_summary,
    baseline_blind_summary,
    gen_informed_summary,
    gen_blind_summary,
    spec_blind_summary,
]

comp = pd.DataFrame(all_summaries)
cols = ["mode", "format_valid_pct", "avg_agent_tool_f1", "avg_rouge_l",
        "agent_correctness", "tool_correctness", "avg_steps", "avg_input_tokens"]
comp = comp[[c for c in cols if c in comp.columns]]
comp.columns = ["Mode", "Format%", "AT-F1", "ROUGE-L", "Agent%", "Tool%", "Steps", "Tokens In"]

print("\\n" + "=" * 100)
print("  FULL COMPARISON")
print("=" * 100)
print(comp.to_string(index=False, float_format="%.3f"))

# Key improvements
blind_before = baseline_blind_summary["avg_agent_tool_f1"]
gen_after = gen_blind_summary["avg_agent_tool_f1"]
spec_after = spec_blind_summary["avg_agent_tool_f1"]
print(f"\\nBLIND MODE AT-F1: Baseline {blind_before:.3f} → Generalist {gen_after:.3f} → Specialist {spec_after:.3f}")
print(f"Token savings: {token_overhead:.0f} tokens/query ({100*token_overhead/baseline_informed_summary['avg_input_tokens']:.0f}% of informed prompt)")"""))

    cells.append(code("""# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
modes = ["Base\\nInformed", "Base\\nBlind", "Gen\\nInformed", "Gen\\nBlind", "Spec\\nBlind"]
colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]

for ax, metric, title in [
    (axes[0], "format_valid_pct", "Format Valid (%)"),
    (axes[1], "avg_agent_tool_f1", "Agent-Tool F1"),
    (axes[2], "agent_correctness", "Agent Correctness"),
    (axes[3], "avg_input_tokens", "Avg Input Tokens"),
]:
    vals = [s.get(metric, 0) for s in all_summaries]
    bars = ax.bar(modes, vals, color=colors)
    ax.set_title(title)
    for i, v in enumerate(vals):
        fmt = f"{v:.0f}" if metric in ("format_valid_pct", "avg_input_tokens") else f"{v:.3f}"
        ax.text(i, v + max(vals)*0.02, fmt, ha="center", fontweight="bold", fontsize=8)

plt.suptitle(f"AssetOpsBench: {MODEL_ID}", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()"""))

    # ══════════════════════════════════════════════════════════════════
    # 14. PER-DOMAIN SPECIALIST VS GENERALIST
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 14. Per-Domain: Specialist vs Generalist"))

    cells.append(code("""# Compare specialist vs generalist per scenario type
gen_by_id = {r["id"]: r for r in gen_blind_results}
spec_by_id = {r["id"]: r for r in specialist_results}

rows = []
for sc in eval_scenarios:
    sid = sc["id"]
    g = gen_by_id.get(sid, {})
    s = spec_by_id.get(sid, {})
    rows.append({
        "id": sid, "type": sc["type"],
        "gen_atf1": g.get("agent_tool_f1", 0), "gen_rouge": g.get("rouge_l", 0),
        "spec_atf1": s.get("agent_tool_f1", 0), "spec_rouge": s.get("rouge_l", 0),
        "spec_better": s.get("agent_tool_f1", 0) > g.get("agent_tool_f1", 0),
    })

df = pd.DataFrame(rows)
print("Per-type comparison (AT-F1):")
type_comp = df.groupby("type").agg(
    gen_atf1=("gen_atf1", "mean"), spec_atf1=("spec_atf1", "mean"),
    gen_rouge=("gen_rouge", "mean"), spec_rouge=("spec_rouge", "mean"),
    spec_wins=("spec_better", "sum"), count=("id", "count"),
).round(3)
print(type_comp.to_string())

spec_wins = df["spec_better"].sum()
print(f"\\nSpecialist wins: {spec_wins}/{len(df)} scenarios ({100*spec_wins/len(df):.0f}%)")"""))

    # ══════════════════════════════════════════════════════════════════
    # 15. TOKEN SAVINGS ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 15. Token Savings Analysis"))

    cells.append(code("""print("=" * 60)
print("  TOKEN SAVINGS ANALYSIS")
print("=" * 60)
informed_tok = baseline_informed_summary["avg_input_tokens"]
blind_tok = gen_blind_summary["avg_input_tokens"]
savings = informed_tok - blind_tok

print(f"  Informed prompt:  {informed_tok:.0f} tokens")
print(f"  Blind prompt:     {blind_tok:.0f} tokens")
print(f"  Savings/query:    {savings:.0f} tokens ({100*savings/informed_tok:.0f}%)")
print(f"  For 152 scenarios: {savings*152:,.0f} tokens saved")
print(f"  For 1000 queries/day: {savings*1000:,.0f} tokens/day")
print()
gen_q = gen_blind_summary.get("avg_agent_tool_f1", 0)
spec_q = spec_blind_summary.get("avg_agent_tool_f1", 0)
best_q = max(gen_q, spec_q)
best_name = "Generalist" if gen_q >= spec_q else "Specialist"
print(f"  Best blind-mode model: {best_name} (AT-F1={best_q:.3f})")
print(f"  This model achieves {best_q:.1%} agent-tool accuracy WITHOUT tool descriptions.")"""))

    # ══════════════════════════════════════════════════════════════════
    # 16. SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 16. Save Results"))

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

print(f"Results saved to {OUTPUT_DIR}/")
print(f"\\n{'='*60}")
print(f"  EXPERIMENT COMPLETE")
print(f"{'='*60}")
for s in all_summaries:
    print(f"  {s['mode']:35s}  AT-F1={s.get('avg_agent_tool_f1',0):.3f}  Agent={s.get('agent_correctness',0):.1%}")
print(f"  Token savings: {savings:.0f}/query")"""))

    # ══════════════════════════════════════════════════════════════════
    # 17. DOWNLOAD
    # ══════════════════════════════════════════════════════════════════
    cells.append(md("## 17. (Optional) Download"))

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
