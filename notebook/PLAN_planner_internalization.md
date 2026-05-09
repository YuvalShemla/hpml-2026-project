# Notebook Plan: Tool-Description Internalization for MCP Planning

> **Goal**: Fine-tune Gemma 4 E4B to internalize AssetOpsBench tool descriptions so it can produce correct MCP tool-use plans **without tool descriptions in the prompt** (blind mode). Systematically ablate training data, LoRA configuration, and measure catastrophic forgetting.

---

## Critical Finding: Train/Test Contamination

**The current datasets have 100% overlap between training and evaluation.**

- The 152 HF `AssetOpsBench` scenarios are the eval gold standard
- ALL 150 unique gold-plan questions appear verbatim in `planning.jsonl` (1,589 examples)
- The remaining ~1,439 training examples are LLM-generated paraphrases of those same 150 questions
- Many paraphrases have >0.7 similarity to gold questions (13/20 sampled)
- `generate_sft_dataset.py` uses `random.choice(scenarios)` over the entire HF pool — no held-out split

**This means all prior "blind mode" results after fine-tuning are unreliable** — the model may be memorizing question-plan pairs, not internalizing tool knowledge.

### Fix: Pattern-Aware Scenario-Level 80/20 Split

We split the 152 HF scenarios 80/20 with a strategy that ensures **test is IID with train** (same tool-use patterns) but genuinely unseen:

**Step 1: Cluster scenarios by tool-use pattern**

Each scenario maps to a pattern = `(type, sorted_tools, step_count)`. Analysis shows:
- 105 scenarios fall in "splittable" patterns (2+ scenarios share the same tool combo)
- 47 scenarios are "singletons" (unique tool combo appears only once)

**Step 2: Stratify by (type, complexity)**

| Type | Simple (<=2 steps) | Complex (>2 steps) | Test (simple) | Test (complex) |
|------|-------------------|--------------------|--------------:|---------------:|
| IoT | 10 | 10 | 2 | 2 |
| FMSA | 9 | 11 | 2 | 2 |
| TSFM | 22 | 1 | 4 | 1 |
| Workorder | 41 | 6 | 8 | 1 |
| multiagent | 11 | 31 | 2 | 6 |
| **Total** | **93** | **59** | **18** | **12** |

**Step 3: Within each stratum, split pattern clusters**

- For splittable patterns (>=2 members): put at least one in train, one in test. This guarantees test uses **the same tool combinations** the model saw during training, just with different questions.
- For singletons: keep most in train. Put a few in test to separately measure **novel-pattern generalization**.

**Step 4: Augment train only**

- The ~1,439 paraphrases in `planning.jsonl` are assigned to train scenarios only (matched by source question similarity)
- `tool_knowledge.jsonl` (307 examples) goes entirely to training (no scenario overlap, pure knowledge)
- `execution.jsonl` examples from test scenarios are excluded

**Step 5: Validate**

- Zero exact or fuzzy (>0.8) overlap between any training question and any test question
- Test = original HF questions only, evaluated against their Gemini gold plans

**Result**: ~122 train scenarios (~1,200 augmented examples) + ~30 test scenarios (clean, IID, unseen)

**Bonus**: Report "seen-pattern accuracy" vs "novel-pattern accuracy" separately (splittable vs singleton test scenarios) to measure generalization.

---

## Notebook Structure

### Section 0: Setup & Data Audit

**Cells:**
- Install dependencies (transformers, peft, trl, bitsandbytes, datasets, evaluate, rouge-score, wandb, litellm)
- Set constants (MODEL_ID, API keys, SEED, output paths)
- Clone repo if on Colab
- Load all three datasets: `tool_knowledge.jsonl` (307), `planning.jsonl` (1,589), `execution.jsonl` (120)
- Load HF scenarios and gold plans
- **Run contamination audit**: show exact overlap count, similarity histogram, print examples of leakage
- **Cluster scenarios by tool-use pattern**: show pattern distribution per type, identify splittable vs singleton
- **Build clean split**: pattern-aware stratified 80/20 split, assign paraphrases to train only, validate zero leakage
- Print dataset composition table (train vs test, per type, per complexity)

**Output**: Clean `train_scenarios`, `test_scenarios`, `train_data` (with paraphrases), guaranteed zero contamination. Separate lists of "seen-pattern" and "novel-pattern" test scenarios.

---

### Section 1: Understand the Data

**Cells:**
- Show example from each dataset category with formatting
- Dataset composition breakdown:

| Dataset | Examples | Categories | Purpose |
|---------|----------|------------|---------|
| `tool_knowledge.jsonl` | 307 | agent_listing (1), tool_listing (6), tool_ownership (31), tool_arguments (31), tool_routing (23+55), hard_negative (52), clarification (45), varied (63) | Teach what tools exist and who owns them |
| `planning.jsonl` | 1,589 | planning (1,520), planning_clarification (69) | Teach question→plan mapping |
| `execution.jsonl` | 120 | execution (120) | Teach plan+execution trace |

- Explain the plan output format (#Task/#Agent/#Tool/#Args/#Dependency/#ExpectedOutput)
- Show the 6 agents and their tool inventories
- Explain blind vs informed mode and the token savings motivation (~2,400 tokens per query)

---

### Section 2: Baseline Evaluation (Informed Mode — With Tool Descriptions)

**Purpose**: Establish the upper bound. How well does the base model plan when it has all tool descriptions?

**Cells:**
- Load Gemma 4 E4B with 8-bit quantization
- Define `INFORMED_PROMPT` (includes full tool descriptions — the ~2,400 token prompt)
- Define `BLIND_PROMPT` (no tool descriptions)
- Run inference on all eval scenarios in **informed mode**
- Run inference on all eval scenarios in **blind mode** (pre-training baseline — should be ~0%)
- Compute structural metrics (agent-tool pair F1, tool set F1, sequence match, arg key overlap, composite score)
- **NEW: Run Gemini LLM-as-judge** on both sets (correctness, agent_routing, tool_selection, argument_quality, efficiency, dependency_correctness — 1-5 scale each)
- Display comparison table: informed vs blind, structural vs judge metrics
- Count prompt tokens for both modes to show the savings potential

**Key outputs:**
- `baseline_informed_results`: upper bound (structural score ~0.5-0.8 for E4B)
- `baseline_blind_results`: lower bound (should be ~0%)
- Token counts: informed ~2,400+ tokens, blind ~200 tokens

---

### Section 3: Define Training Data Configurations

**Purpose**: Set up the dataset ablation study. We test which combination of training data best teaches tool internalization.

**Five configurations:**

| Config | Training Data | Hypothesis |
|--------|--------------|------------|
| **A: Planning-only** | `planning.jsonl` (clean) | Direct question→plan examples. Most straightforward but may memorize patterns without understanding tools. |
| **B: Tool-knowledge-only** | `tool_knowledge.jsonl` | Pure tool awareness. Model learns what tools exist and their arguments. May not learn to compose plans. |
| **C: Tool-knowledge + Planning** | Both datasets | Two-phase: first bake tool knowledge, then teach planning. Should give best of both worlds. |
| **D: Curriculum (staged)** | Tool-knowledge (stage 1) → Planning (stage 2) | Same data as C but trained in stages with separate LoRA passes. Tests whether curriculum order matters. |
| **E: All data** | tool_knowledge + planning + execution | Everything available. Tests whether execution traces help or hurt planning. |

**Implementation:**
- For each config, build the `train_data` list with proper chat template formatting
- Apply the clean split filter to remove all held-out scenario contamination
- For config D (curriculum), save a checkpoint after stage 1 and continue from it
- Log dataset sizes and category distributions for each config

---

### Section 4: Training Loop

**Cells:**
- Define shared training infrastructure:
  - `setup_lora(model, r, alpha)` — prepare model with LoRA
  - `train_model(model, data, config_name)` — SFT training with W&B logging
  - `run_evaluation(model, scenarios, prompt_template, mode_name)` — generate plans and score them
  - `run_gemini_judge(results, gold_plans)` — call Gemini API for LLM-as-judge scoring
- Default hyperparameters: `r=16, alpha=32, lr=2e-4, epochs=2, cosine scheduler, 8-bit quant`
- For each of the 5 configs (A through E):
  1. Setup LoRA on fresh base model
  2. Train
  3. Evaluate in **blind mode** (no tool descriptions)
  4. Evaluate in **informed mode** (with tool descriptions, as sanity check)
  5. Run Gemini judge on blind-mode outputs
  6. Log to W&B
  7. Save adapter checkpoint
- Display comparison table across all 5 configs

**Key metrics per config:**
- Structural: agent-tool pair F1, tool set F1, sequence match, arg key overlap, composite
- Judge: correctness, agent_routing, tool_selection, argument_quality (1-5 each)
- Token savings: blind prompt tokens vs informed prompt tokens

---

### Section 5: LoRA Configuration Ablation

**Purpose**: For the best-performing data config from Section 4, test LoRA variants.

**Sub-experiments:**

#### 5a: QLoRA (4-bit) vs LoRA (8-bit) vs LoRA (16-bit)

| Config | Quantization | Memory | Training Speed |
|--------|-------------|--------|----------------|
| QLoRA 4-bit NF4 | 4-bit | ~5 GB | Fastest |
| QLoRA 8-bit | 8-bit | ~9 GB | Medium |
| LoRA 16-bit (bf16) | None | ~18 GB | Slowest, highest quality |

For each: train with same data config, evaluate blind mode, compare quality vs efficiency.

#### 5b: LoRA Rank Scaling

Using the best quantization from 5a, sweep over ranks:

| Rank (r) | Alpha | Trainable Params (approx) |
|----------|-------|--------------------------|
| 4 | 8 | ~2M |
| 8 | 16 | ~4M |
| 16 | 32 | ~8M |
| 32 | 64 | ~16M |
| 64 | 128 | ~32M |

For each rank:
- Train, evaluate blind mode
- Plot: rank vs blind-mode F1 (does more capacity help?)
- Plot: rank vs training loss convergence speed
- Plot: rank vs forgetting score (Section 6)

**Output**: Pareto frontier of quality vs trainable parameters.

---

### Section 6: Catastrophic Forgetting Analysis

**Purpose**: Verify that fine-tuning for tool planning doesn't destroy the model's general capabilities.

#### Benchmark Selection

We use a lightweight forgetting test suite (~15-20 minutes per model on H100). These are the standard benchmarks used by the community (Open LLM Leaderboard, lm-evaluation-harness) for detecting capability degradation:

| Benchmark | Examples | What It Tests | Source | Metric |
|-----------|----------|---------------|--------|--------|
| **HellaSwag** | 200 (subset) | Commonsense reasoning / sentence completion | `Rowan/hellaswag` | `acc_norm` (log-likelihood) |
| **ARC-Challenge** | Full (1,172) | Grade-school science reasoning (MCQ) | `allenai/ai2_arc` ARC-Challenge | `acc_norm` |
| **TruthfulQA (mc2)** | Full (817) | Factual accuracy, resistance to misconceptions | `truthfulqa/truthful_qa` | `acc` |
| **GSM8K** | 100 (subset) | Multi-step math reasoning | `openai/gsm8k` | `exact_match` |

**Why these four:**
- **HellaSwag** is the most sensitive to general degradation — models that forget show clear drops here first
- **ARC-Challenge** (not ARC-Easy) requires genuine reasoning, not pattern matching; small enough to run in full
- **TruthfulQA** tests a different axis (factual grounding); only 817 examples so runs fast
- **GSM8K** tests multi-step reasoning which is fragile after fine-tuning; Gemma 4 E4B baseline is likely 30-50%, so even small drops are detectable

**Implementation:**
- Load subsets with fixed seed (deterministic sampling for apples-to-apples comparison)
- For HellaSwag: compute log-likelihood of each completion, select highest (`acc_norm`)
- For ARC-Challenge: multiple choice, check if model selects correct letter
- For TruthfulQA: mc2 format, multiple choice scoring
- For GSM8K: generate answer, extract final number, check exact match
- Run on: (1) base model before any fine-tuning, (2) each fine-tuned variant from Sections 4 and 5
- Compute **retention rate** = fine-tuned accuracy / base accuracy for each benchmark
- Flag any config where retention drops below 90% on any benchmark; >5% drop is a concern, >10% is meaningful forgetting

**Key output:**
- Table: model variant x benchmark accuracy (with base model row as reference)
- Retention heatmap (% of base performance retained)
- Identify which training configs cause the most forgetting
- Correlation: does more training data / higher rank / more epochs = more forgetting?

---

### Section 7: Evaluation Deep-Dive

**Purpose**: Aggregate all results and produce paper-ready analysis.

**Cells:**

#### 7a: Main Results Table

| Model | Data Config | LoRA Config | Blind AT-F1 | Blind Judge Score | Informed AT-F1 | Token Savings | Forgetting (avg retention) |
|-------|-------------|-------------|-------------|-------------------|-----------------|---------------|---------------------------|
| Base (no FT) | - | - | 0.00 | 1.0 | X.XX | 0% | 100% |
| Config A | planning | r=16/8bit | X.XX | X.X | X.XX | ~92% | XX% |
| ... | ... | ... | ... | ... | ... | ... | ... |

#### 7b: Per-Type Analysis

Break down blind-mode performance by scenario type (IoT, FMSA, TSFM, Workorder, multiagent).
- Which types benefit most from internalization?
- Do multi-agent scenarios suffer more?

#### 7c: Error Analysis

- Show 5 best blind-mode predictions vs gold (model learned well)
- Show 5 worst blind-mode predictions vs gold (model still fails)
- Agent confusion matrix: predicted agent vs gold agent
- Common failure patterns: wrong agent, wrong tool, wrong args, missing steps

#### 7d: Gemini Judge vs Structural Metrics Correlation

- Scatter plot: structural composite score vs judge overall score
- Do they agree? Where do they diverge?
- This validates whether our heuristic metrics are trustworthy

#### 7e: Token Savings Analysis

- Compute per-scenario: informed prompt tokens, blind prompt tokens, savings
- Average savings across all scenarios
- Plot: quality (blind F1) vs token savings — is the tradeoff worth it?

---

### Section 8: Conclusions & Next Steps

**Cells:**
- Summary of which data config works best
- Summary of which LoRA config is optimal
- Whether catastrophic forgetting is a concern
- Recommended configuration for the planner model
- What to do next: train the executor model (Step 2 of the two-model split)

---

## Gemini LLM-as-Judge Integration

Use the existing `litellm` infrastructure in the repo. The judge call:

```python
import litellm

JUDGE_PROMPT = """You are evaluating an AI-generated MCP tool-use plan.

## Question
{question}

## Gold Reference Plan
{gold_plan}

## Candidate Plan (from fine-tuned model, generated WITHOUT tool descriptions)
{candidate_plan}

## Available Agents & Tools
- IoTAgent: sites(), assets(site_name), sensors(site_name, asset_id), history(...)
- FMSRAgent: get_failure_modes(asset_name), get_failure_mode_sensor_mapping(...)
- TSFMAgent: get_ai_tasks(), get_tsfm_models(), run_tsfm_forecasting(...), ...
- Utilities: json_reader(...), current_date_time(), current_time_english()
- WorkOrderAgent: get_work_orders(...), get_preventive_work_orders(...), ...

Rate 1-5 on each dimension:
1. correctness: Would this plan answer the question?
2. agent_routing: Are the correct agents assigned?
3. tool_selection: Are the correct tools selected?
4. argument_quality: Are tool arguments correct?
5. efficiency: Is the plan appropriately sized?
6. dependency_correctness: Are step dependencies correct?

Respond with ONLY JSON:
{{"correctness": N, "agent_routing": N, "tool_selection": N, "argument_quality": N, "efficiency": N, "dependency_correctness": N, "explanation": "..."}}
"""

def judge_plan(question, gold_plan, candidate_plan):
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}],
        temperature=0,
        max_tokens=1024,
        api_key=os.environ["GEMINI_API_KEY"],
    )
    return json.loads(response.choices[0].message.content)
```

Cost: ~$0.001 per scenario, ~$0.03 for 30 eval scenarios. Negligible.

---

## Clean Data Split Implementation

```python
from datasets import load_dataset
from difflib import SequenceMatcher
from collections import defaultdict
import random

SEED = 42
random.seed(SEED)

# ── Step 1: Load HF scenarios + gold plans ──
hf_ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
scenarios = [dict(row) for row in hf_ds["train"]]
gold_by_id = {g["id"]: g for g in gold_plans_raw}

# ── Step 2: Assign each scenario a tool-use pattern ──
for sc in scenarios:
    g = gold_by_id.get(sc["id"], {})
    sc["pattern"] = (sc["type"], tuple(sorted(g.get("tools_used", []))), g.get("plan_steps", 0))
    sc["complexity"] = "simple" if g.get("plan_steps", 0) <= 2 else "complex"

# ── Step 3: Group by (type, complexity), then by pattern ──
by_stratum = defaultdict(lambda: defaultdict(list))  # (type, complexity) -> pattern -> [scenarios]
for sc in scenarios:
    stratum = (sc["type"], sc["complexity"])
    by_stratum[stratum][sc["pattern"]].append(sc)

# ── Step 4: Split each stratum 80/20, preferring to split pattern clusters ──
test_scenarios = []
train_scenarios = []

for stratum, pattern_groups in by_stratum.items():
    stype, complexity = stratum
    all_in_stratum = [sc for group in pattern_groups.values() for sc in group]
    n_test = max(1, round(len(all_in_stratum) * 0.20))

    # First: pick one from each splittable pattern (>=2 members) for test
    test_pool = []
    remaining = []
    for pattern, members in pattern_groups.items():
        random.shuffle(members)
        if len(members) >= 2:
            test_pool.append(members[0])       # one to test
            remaining.extend(members[1:])       # rest to train
        else:
            remaining.extend(members)           # singletons default to train

    # Fill test quota from test_pool (splittable patterns first)
    random.shuffle(test_pool)
    actual_test = test_pool[:n_test]
    remaining.extend(test_pool[n_test:])        # overflow back to train

    # If we still need more test scenarios, take from singletons (novel-pattern test)
    if len(actual_test) < n_test:
        shortfall = n_test - len(actual_test)
        random.shuffle(remaining)
        actual_test.extend(remaining[:shortfall])
        remaining = remaining[shortfall:]

    for sc in actual_test:
        sc["test_kind"] = "seen_pattern" if any(
            sc["pattern"] == t["pattern"] for t in remaining
        ) else "novel_pattern"

    test_scenarios.extend(actual_test)
    train_scenarios.extend(remaining)

test_ids = {sc["id"] for sc in test_scenarios}
train_ids = {sc["id"] for sc in train_scenarios}
test_questions = {sc["text"].strip().lower() for sc in test_scenarios}

# ── Step 5: Filter training data — remove anything derived from test scenarios ──
def is_contaminated(question, test_questions, threshold=0.8):
    q = question.strip().lower()
    if q in test_questions:
        return True
    for tq in test_questions:
        if SequenceMatcher(None, q, tq).ratio() > threshold:
            return True
    return False

clean_plan = [ex for ex in ds_plan if not is_contaminated(ex["messages"][0]["content"], test_questions)]
clean_exec = [ex for ex in ds_exec if not is_contaminated(ex["messages"][0]["content"], test_questions)]
clean_tool = ds_tool  # tool_knowledge has no scenario-specific questions — safe to use all

# ── Step 6: Validate zero leakage ──
train_qs = {ex["messages"][0]["content"].strip().lower() for ex in clean_plan + clean_exec}
leaked = train_qs & test_questions
assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} questions appear in both train and test!"
print(f"Train scenarios: {len(train_ids)} | Test scenarios: {len(test_ids)}")
print(f"Clean train examples: {len(clean_tool)} tool + {len(clean_plan)} plan + {len(clean_exec)} exec")
print(f"Test: {sum(1 for s in test_scenarios if s['test_kind']=='seen_pattern')} seen-pattern, "
      f"{sum(1 for s in test_scenarios if s['test_kind']=='novel_pattern')} novel-pattern")
```

---

## Estimated Runtime (Colab A100/H100)

| Section | Estimated Time | Notes |
|---------|---------------|-------|
| 0-1: Setup & data | 5 min | Data loading, audit |
| 2: Baseline eval | 15-20 min | 30 scenarios x 2 modes + Gemini judge |
| 3: Data configs | 2 min | Just building datasets |
| 4: Training (5 configs) | 2-3 hours | ~25-35 min each, 8-bit, r=16 |
| 5a: Quantization ablation | 1-1.5 hours | 3 variants |
| 5b: Rank scaling | 2-3 hours | 5 rank values |
| 6: Forgetting tests | 1-1.5 hours | 400 examples x ~8 models |
| 7: Analysis | 10 min | Aggregation and plots |

**Total: ~7-10 hours on a single H100.** Can be reduced with LIGHT_MODE (fewer configs, smaller rank sweep, fewer forgetting examples).

---

## W&B Logging Plan

- Project: `hpml-group20-assetops`
- Groups: `internalization-data-ablation`, `internalization-lora-ablation`, `internalization-forgetting`
- Per run: training loss curve, eval metrics, per-scenario W&B Table, forgetting scores
- Tags: `data-config-{A..E}`, `lora-r{N}`, `quant-{4bit,8bit,16bit}`, `light-mode` / `full-run`
- Comparison dashboard: parallel coordinates plot of all configs

---

## File Dependencies

| File | Usage |
|------|-------|
| `benchmark/generate_data/datasets/tool_knowledge.jsonl` | Training data (307 examples) |
| `benchmark/generate_data/datasets/planning.jsonl` | Training data (1,589 examples) |
| `benchmark/generate_data/datasets/execution.jsonl` | Training data (120 examples) |
| `benchmark/baseline_tests/gemini_flash_informed_results.json` | Gold plans (148 with steps) |
| `benchmark/baseline_tests/evaluate_plan_quality.py` | Reference for structural eval + judge prompt |
| `src/llm/litellm.py` | Reference for Gemini API calls |
| `src/workflow/planner.py` | Plan prompt template + parser |

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 30 eval scenarios is small | Report confidence intervals; run 3 seeds |
| Gemini judge may be noisy | Correlate with structural metrics; use temperature=0 |
| 4-bit quantization may break on Gemma 4 | Fallback to 8-bit; flag in notebook if NaN losses |
| Forgetting benchmarks may be slow | Use 100-example subsets; cache base model scores |
| Paraphrase contamination (fuzzy match) | Use 0.8 threshold; manually inspect borderline cases |
| Different configs need fresh base model loads | Use `gc.collect()` + `torch.cuda.empty_cache()` between configs |
