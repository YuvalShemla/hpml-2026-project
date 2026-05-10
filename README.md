# Internalizing MCP Tool Knowledge in Small LLMs via QLoRA Fine-Tuning

**Course:** HPML — High Performance Machine Learning, Columbia University, Spring 2026
**Team:** Group 20 — Tanmay Agarwal, Yuval Shemla, Ayal Yakobe
**GitHub:** [github.com/YuvalS0workin/hpml_group_20](https://github.com/YuvalS0workin/hpml_group_20)

---

## Problem Statement

MCP-based LLM agents include full tool schemas in every prompt, creating ~2,400 tokens of overhead per query (82.6% of input tokens). Small models (~4B parameters) cannot plan without these descriptions, forcing reliance on expensive frontier models. We fine-tune small models to internalize tool knowledge, eliminating this overhead while improving planning quality.

## Key Result

Fine-tuned ~4B models operating **without any tool descriptions** outperform the unfine-tuned baseline that receives full tool schemas:

![Main Results](notebook/figures/main_results_bar.png)

| Configuration | AT-F1 | Judge Score (1–5) | Input Tokens |
|---|---|---|---|
| Description-free baseline (no FT) | 0.000 | 1.88 | ~128 |
| Informed baseline (no FT, full schemas) | 0.470 | 2.88 | ~2,400 |
| **Gemma 4 E4B fine-tuned (no schemas)** | **0.635** | **3.60** | **~128** |
| **Qwen3-4B fine-tuned (no schemas)** | **0.605** | **3.78** | **~128** |

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.public .env
# Edit .env: add GEMINI_API_KEY (required)

# 3. Start CouchDB (required for IoTAgent tools)
docker compose -f src/couchdb/docker-compose.yaml up -d

# 4. Run a single query (informed mode)
uv run plan-execute --model-id "gemini/gemma-4-26b-a4b-it" --show-plan "What IoT sites are available?"

# 5. Evaluate plan quality (structural, free)
uv run python benchmark/baseline_tests/evaluate_plan_quality.py \
    --candidate benchmark/baseline_tests/gemini_flash_informed_results.json \
    --mode structural
```

### Reproducing Fine-Tuning Results

The fine-tuning experiments run on a single A100 80GB GPU. See the notebooks in `notebook/` — the primary experiment notebook is `Planner_Internalization_Experiment.ipynb` with sequential runs documented in `*_run_2` through `*_run_6`.

---

## Experiment Tracking

**Weights & Biases Dashboard:** [wandb.ai/group20/hpml-asset-ops-group20](https://wandb.ai/group20/hpml-asset-ops-group20)

All training runs, evaluation metrics, LoRA rank sweeps, and forgetting benchmarks are logged to W&B.

---

## Methodology

### Benchmark

[AssetOpsBench](https://openreview.net/forum?id=ld6JUQbhes) (IBM Research) — 152 natural-language scenarios for industrial asset operations, requiring multi-step tool-use planning across 5 agent families and 23 tools.

### Models

| Model | Total Params | Active Params | Architecture |
|---|---|---|---|
| Gemma 4 E4B-it | ~8B | 4.5B | Hybrid attention (36 sliding + 6 global) |
| Qwen3-4B | 4.0B | 3.6B | GQA (32Q / 8KV) |

### Training

- **Method:** QLoRA (8-bit quantization, LoRA rank 32, α=64)
- **Data:** ~1,741 examples (tool knowledge + planning scenarios)
- **Hardware:** Single NVIDIA A100 80GB
- **Cost:** ~$62 GPU compute (15.8 A100-hours)
- **Training time:** 56 min (Gemma), 40 min (Qwen3)

### Evaluation

- **Structural metrics:** AT-F1 (agent-tool pair F1), ArgKey-F1
- **LLM-as-judge:** Gemini 2.5 Flash rates plans on 6 dimensions (1–5 scale)
- **Forgetting:** 100 MCQ questions from MMLU, ARC-Challenge, HellaSwag

---

## Results Summary

### Internalization
Fine-tuned models produce correct plans without tool descriptions, achieving 95–98% agent/tool accuracy and surpassing the informed baseline on all metrics.

### Profiling (A100 80GB)

| Metric | Gemma 4 E4B | Qwen3-4B |
|---|---|---|
| Base memory (8-bit) | 11.5 GB | 4.42 GB |
| Peak training memory | 24.1 GB | 16.06 GB |
| Inference speed | 1.4 tok/s | 3.49 tok/s |
| Dominant CUDA op | MatMul8bitLt (56.3%) | — |

### Forgetting
- Gemma retains 79.8–82.1% of base MCQ accuracy
- Qwen3 retains 61.3% (higher adaptation intensity: 1.62% vs 0.63% trainable params)

---

## Repository Structure

```
hpml_group_20/
├── src/
│   ├── llm/                    # LLM backend (litellm routing)
│   ├── workflow/               # Plan-execute pipeline (planner, executor, runner, CLI)
│   ├── servers/                # 4 MCP servers (iot, fmsr, tsfm, utilities)
│   └── couchdb/                # Docker compose for CouchDB
├── benchmark/
│   ├── baseline_tests/         # Eval scripts, gold plans, result JSONs
│   ├── generate_data/          # SFT training data generation
│   │   └── datasets/           # Generated JSONL datasets
│   └── cods_track1/            # Planning benchmark (Docker)
├── notebook/
│   ├── Planner_Internalization_Experiment*.ipynb  # Fine-tuning experiments
│   ├── figures/                # Paper figures (PNG)
│   └── final_report.tex        # LaTeX source
├── deliverables/               # Final report PDF & presentation
├── docs/                       # Experiment results documentation
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Project config (uv)
├── LICENSE                     # Apache-2.0
└── .env.public                 # Environment template
```

---

## Team Contributions

| Member | Contributions |
|---|---|
| **Tanmay Agarwal** | Fine-tuning pipeline, QLoRA implementation, LoRA rank sweep, data ablation, quantization experiments |
| **Yuval Shemla** | Baseline evaluation framework, SFT data generation, profiling, forgetting analysis, Qwen3 comparison |
| **Ayal Yakobe** | MCP server setup, E2E execution pipeline, benchmark infrastructure, experiment tracking |

---

## AI Tool Use

Our team used AI tools, including ChatGPT, Gemini, and Claude, to help write and run specific tests and to polish the writing in our paper. All research decisions, ideas, analysis, and conclusions are our own.

---

## Citation
If you build on this work, please cite:
```bibtex
@misc{group20_2026hpml,
title = {[Internalizing MCP Tool Knowledge in Small LLMs via QLoRA Fine-Tuning
]},
author = {Agarwal, Tanmay and Shemla, Yuval and Yakobe, Ayal},
year = {2026},
note = {HPML Spring 2026 Final Project, Columbia University},
url = {https://github.com/YuvalShemla/hpml-2026-project.git}
}
```
## Contact
Open a GitHub Issue or email: 
- Tanmay Agarwal: ta2830@columbia.edu
- Yuval Shemla: ys3571@columbia.edu
- Ayal Yakobe: amy2127@columbia.edu

---

## Links

- [AssetOpsBench Paper](https://openreview.net/forum?id=ld6JUQbhes)
- [HuggingFace Dataset](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
- [IBM AssetOpsBench Repo](https://github.com/IBM/AssetOpsBench)
- [W&B Dashboard](https://wandb.ai/group20/hpml-asset-ops-group20)
- [Gemma 4 Docs](https://ai.google.dev/gemma/docs/core)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
