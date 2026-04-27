# HPML Group 20 — TODO

## Task 1: Add Weights & Biases Logging
- [ ] Integrate W&B into the notebook (`wandb.init()`, `report_to="wandb"` in SFTConfig)
- [ ] Log training loss, eval loss, learning rate per step
- [ ] Log evaluation metrics after each model (baseline, generalist, planner, specialists, pipeline)
- [ ] Log token savings, per-scenario results, comparison tables as W&B Tables
- [ ] Log training config (model, LoRA rank, lr, epochs, batch size, dataset sizes) as run config
- [ ] Create a W&B project: `hpml-group20-assetops`
- [ ] Ensure every run is tagged (e.g., `light-mode`, `full-run`, `hp-sweep`)

## Task 2: Full Training Run with Best Parameters
- [ ] Set `LIGHT_MODE = False`
- [ ] Use all ~1,400 clean training examples (no cap)
- [ ] 3 epochs, LoRA r=32, alpha=64, lr=2e-4, cosine schedule
- [ ] Train all 3 model types: generalist, planner, 4 specialists
- [ ] Evaluate on 50 held-out scenarios (stratified by type)
- [ ] Log everything to W&B
- [ ] Save all adapters to Google Drive
- [ ] Fix `torch_dtype` → `dtype` deprecation warning
- [ ] Fix WorkOrder gold plan mismatch (gold uses IoTAgent, training uses WorkOrderAgent)

## Task 3: Hyperparameter Tuning
- [ ] Sweep LoRA rank: r=8, 16, 32, 64
- [ ] Sweep learning rate: 1e-4, 2e-4, 5e-4
- [ ] Sweep epochs: 1, 2, 3, 5
- [ ] Test seq_length: 512, 1024, 2048
- [ ] Compare 8-bit vs 4-bit quantization (if Gemma 4 4-bit bug is fixed)
- [ ] Use W&B sweeps or manual grid search
- [ ] Find best config for generalist blind-mode AT-F1
- [ ] Find best config for planner routing accuracy
- [ ] Document best hyperparameters in README

## Task 4: Specialist vs Generalist Analysis
- [ ] Fix pipeline evaluation so specialist args actually flow into metrics (ArgKey, ArgVal)
- [ ] Run proper comparison: generalist vs planner+specialists on all 50 held-out scenarios
- [ ] Break down results per scenario type (IoT, FMSR, TSFM, WorkOrder, multiagent)
- [ ] Measure: which approach wins on single-agent tasks? On multi-agent tasks?
- [ ] Measure specialist inference latency vs generalist (can specialists run in parallel?)
- [ ] Analyze failure cases: where does the generalist beat specialists and vice versa?
- [ ] Test: what happens when planner routes to the wrong specialist?
- [ ] Create clear comparison table and visualizations for the paper

## Task 5: Improve Evaluation
- [ ] Fix pipeline specialist arg parsing (args not flowing into AT-F1 properly)
- [ ] Add per-tool accuracy breakdown (which tools does the model get right/wrong most?)
- [ ] Add confusion matrix: predicted agent vs gold agent
- [ ] Add dependency correctness metric (does the model get step ordering right?)
- [ ] Consider adding LLM-as-judge evaluation (Task Completion, Retrieval Accuracy from paper)
- [ ] Test on expanded HF configs (compressor, hydraulic_pump, PHM) as out-of-distribution eval
- [ ] Compare our metrics against the paper's reported baselines

## Task 6: Write Paper and Final Report
- [ ] Introduction: the token overhead problem in MCP tool planning
- [ ] Related work: AssetOpsBench paper, tool-use LLMs, QLoRA/PEFT, specialist models
- [ ] Method: 3-tier architecture (generalist, planner, specialists), data generation pipeline
- [ ] Experiments: baseline → fine-tuned comparison, specialist vs generalist, HP sweep results
- [ ] Results: tables, charts from W&B, token savings analysis
- [ ] Discussion: what worked, what didn't, WorkOrder gap, multi-agent challenges
- [ ] Include the production architecture argument (planner + parallel specialists)
- [ ] Prepare final presentation slides
