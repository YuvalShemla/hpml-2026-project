# HPML Group 20: Internalizing MCP Tool Knowledge in Small LLMs

**Course:** High Performance Machine Learning (Columbia, Spring 2026)
**Team:** Group 20
**Repo:** Fork of [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench)

---

## Research Question

Can Gemma internalize AssetOpsBench tool descriptions well enough to reduce prompt cost without losing planning and execution quality, and which adaptation strategy handles newly added tools best: plain SFT, curriculum SFT, or modular continual LoRA?

## Project Overview

We train **Gemma 4 E4B** via curriculum QLoRA to produce correct MCP tool-use plans **without tool descriptions in the prompt**. Currently, the plan-execute framework pays a ~2,400 token tax per query to include tool descriptions (~95% of plan call input). Fine-tuning internalizes the tool catalog, eliminating this overhead.

**Benchmark:** [AssetOpsBench](https://arxiv.org/pdf/2506.03828) -- 141 core scenarios (99 single-agent + 42 multi-agent), 6 domain agents, 30+ tools.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.public .env
# Edit .env: add GEMINI_API_KEY (required), OPENROUTER_API_KEY (optional)

# 3. Start CouchDB (required for IoTAgent tools)
docker compose -f src/couchdb/docker-compose.yaml up -d

# 4. Run a single query
uv run plan-execute --model-id "gemini/gemma-4-26b-a4b-it" --show-plan "What IoT sites are available?"
```

---

## Three Studies

### Study 1: Tool-Description Internalization
Evaluate on 141 core scenarios in three inference settings:
1. **Full tool manual** in prompt (current baseline)
2. **Compressed tool cards** in prompt
3. **No tool descriptions** in prompt (target)

Planning-only first, then full MCP execution.

### Study 2: Curriculum vs Non-Curriculum Tuning

| Regime | Target |
|--------|--------|
| Plan-SFT baseline | scenario + tool inventory -> planning_steps |
| Plan+execution SFT | -> planning_steps + execution_steps + execution_links |
| Curriculum Stage A | Tool taxonomy and ownership |
| Curriculum Stage B | Tool selection and argument prediction |
| Curriculum Stage C | Full plan generation |
| Curriculum Stage D | Plan + execution traces |
| Curriculum Stage E | Clarification / abstention / insufficient-data |

### Study 3: Continual Learning
- **Hold-one-tool-family-out:** Train without Vibration server, then adapt. Measure retention + acquisition.
- **Core-to-expanded:** Train on 141 core scenarios, adapt to new asset classes (compressor, hydraulic, PHM).

---

## Running Evaluations

### Track 1: Plan Quality (141+ scenarios)
```bash
# Gemini 2.5 Flash (generates gold reference plans)
uv run python benchmark/baseline_tests/run_gemini_track1.py

# Small models via Gemini API / OpenRouter
uv run python benchmark/baseline_tests/run_both_models_informed.py --model gemma_3_4b
```

### Plan Quality Evaluation
```bash
# Structural only (fast, free)
uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
    --candidate benchmark/baseline_tests/gemma_4_26b_a4b_informed_results.json \
    --mode structural

# Structural + LLM judge (~$0.10)
uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
    --candidate benchmark/baseline_tests/gemma_4_26b_a4b_informed_results.json \
    --mode all
```

### E2E Execution with Token Tracking
```bash
uv run python benchmark/baseline_tests/run_gemma4_e2e.py
```

### Dashboard
```bash
uv run python benchmark/baseline_tests/build_dashboard.py
open benchmark/baseline_tests/gemma4_e2e_dashboard.html
```

Shows per-scenario pipeline visualization, token breakdowns, gold vs model plan comparison. See [DASHBOARD_GUIDE.md](benchmark/baseline_tests/DASHBOARD_GUIDE.md).

---

## Full Tool Inventory

| Agent | Tools | In Our Repo |
|-------|-------|-------------|
| **IoTAgent** | sites, assets, sensors, history | Yes |
| **FMSRAgent** | get_failure_modes, get_failure_mode_sensor_mapping | Yes |
| **TSFMAgent** | get_ai_tasks, get_tsfm_models, run_tsfm_forecasting, run_tsfm_finetuning, run_tsad, run_integrated_tsad | Yes |
| **Utilities** | json_reader, current_date_time, current_time_english | Yes |
| **WorkOrder** | get_work_orders, get_preventive/corrective_work_orders, get_events, get_failure_codes, get_work_order_distribution, predict_next_work_order, analyze_alert_to_failure | Upstream only |
| **Vibration** | get_vibration_data, list_vibration_sensors, compute_fft_spectrum, compute_envelope_spectrum, assess_vibration_severity, calculate_bearing_frequencies, list_known_bearings, diagnose_vibration | Upstream only (ideal for continual-learning) |

---

## Metrics

### Paper's Core 3 (rubric-based LLM judge)
Task Completion, Data Retrieval Accuracy, Result Verification

### Plan Alignment
ROUGE-1/2/L/Lsum, BERTScore + ROUGE-L combined

### Our Structural Metrics (automated, free)
Agent-Tool pair F1 (30%), Tool set F1 (20%), Agent set F1 (15%), Tool sequence match (15%), Arg key overlap (10%), Efficiency penalty (10%)

### Token Efficiency
Tool description tokens per plan call, token reduction vs full-manual baseline, total tokens per scenario

---

## Baseline Results (Informed Mode)

| Model | Active Params | Structural | Judge | Valid Plans |
|-------|--------------|-----------|-------|-------------|
| Gemini 2.5 Flash (gold) | ~100B+ | 1.000 | 5.00/5.0 | 148/152 |
| Gemma 4 26B-A4B (MoE) | 3.8B | 0.811 | 4.54/5.0 | 55/55 |
| Llama 3.1 8B | 8B | 0.696 | 3.82/5.0 | 12/12 |
| Gemma 3n E4B | 4.5B | 0.502 | 3.38/5.0 | 152/152 |

**All models score 0% in blind mode.** Paper's single-agent baseline: 26.95% task completion.

Detailed reports: [docs/reports/](docs/reports/)

---

## Training Data (3 datasets)

1. **Tool knowledge:** tool existence, ownership, arguments, routing, hard negatives
2. **Planning:** scenario -> concise planning_steps
3. **Execution-structured:** scenario -> planning_steps + execution_steps + execution_links

Plus clarification / abstention / insufficient-data examples. Never leak eval scenarios into training.

```bash
# Existing SFT generation script
uv run python benchmark/generate_data/generate_sft_dataset.py
```

Gold plans: `benchmark/baseline_tests/gemini_flash_informed_results.json` (148 valid)

---

## Model & Training

- **Primary:** Gemma 4 E4B (instruction-tuned) + QLoRA
- **Stack:** HuggingFace Transformers + PEFT + TRL + bitsandbytes
- **Hardware:** 1x H100
- **Secondary:** Gemma 4 26B A4B (MoE) if Study 1 succeeds

---

## Architecture

```
Question -> DISCOVER (MCP) -> PLAN (LLM) -> EXECUTE (MCP tools) -> SUMMARIZE (LLM) -> Answer
                                 ^
                                 |
                    Tool descriptions in prompt (~2,268 tokens)
                    THIS IS WHAT FINE-TUNING ELIMINATES
```

---

## Repository Structure

```
hpml_group_20/
├── src/
│   ├── llm/                    # LLM backend (litellm routing)
│   ├── workflow/                # Plan-execute runner (planner, executor, runner, CLI)
│   ├── servers/                 # 4 MCP servers (iot, fmsr, tsfm, utilities)
│   └── couchdb/                # Docker compose for CouchDB
├── benchmark/
│   ├── baseline_tests/         # Eval scripts, results JSON, dashboard builder
│   ├── generate_data/          # SFT training data generation
│   └── cods_track1/            # Planning benchmark (Docker)
├── docs/reports/               # Detailed baseline reports
├── INSTRUCTIONS.md             # Full setup guide
├── CLAUDE.md                   # Project instructions for Claude Code
└── .env.public                 # Environment template
```

---

## Links

- [AssetOpsBench Paper](https://arxiv.org/pdf/2506.03828)
- [HuggingFace Dataset](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
- [IBM AssetOpsBench Repo](https://github.com/IBM/AssetOpsBench)
- [Gemma Tuning Guide](https://ai.google.dev/gemma/docs/lora_tuning)
