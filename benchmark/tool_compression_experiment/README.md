# Tool Description Compression Experiment

## Goal
Compare original MCP tool descriptions against Gemini-compressed descriptions using Qwen, measuring task accuracy and token usage.

## Current status: Preliminary baseline
- Pilot model: Qwen3-14B via OpenRouter.
- Original-description baseline: 10 scenarios attempted; 9 completed, 1 failed during summarization because the tool response exceeded the model's context window.
- A separate limited-output runner was created to prevent oversized summarization prompts.
- Scenario 9 completed with the limited runner: 4 execution steps, 11,317 tokens.
- Scenario 9 did not retrieve the requested sensor data because the history tool received an invalid sensor identifier.
- Other scenarios also returned tool-level errors despite being marked successful by the runner.

## Files
- `run_qwen_e2e.py`: Original-description baseline runner.
- `run_qwen_e2e_limited.py`: Experimental runner with per-step summarization-output limits.
- `qwen14b_original_baseline_results_initial.json`: Preserved initial baseline results.
- `qwen14b_original_baseline_results.json`: Original-description baseline results.
- `qwen14b_limited_test_results.json`: Limited-runner Scenario 9 result.
- `qwen14b_pilot_results.json`: Initial pilot result.

## Limitations and next steps
- Execution completion is not equivalent to task correctness.
- Tool-response errors and actual task outcomes require separate evaluation.
- Reference plans may differ from the current tool interfaces.
- Qwen3-14B is a pilot model; the final model selection is pending confirmation.
- Gemini compression and a matched original-versus-compressed comparison have not yet been completed.
