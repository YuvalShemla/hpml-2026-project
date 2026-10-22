# AssetOpsBench: Gemini Tool Compression Results

## Experiment
- Planner: Qwen3-14B via OpenRouter
- Compressor: Gemini 2.5 Flash
- MCP tools compressed: 62
- Scenarios: 10 per condition
- Summarization tool-output limit: 12,000 characters per step

## Results
| Metric | Original | Compressed |
|---|---:|---:|
| Total tokens | 117,167 | 87,848 |
| Input tokens | 101,094 | 66,318 |
| Pipeline completion | 10/10 | 7/10 |

Total token reduction: **25.0%** (29,319 tokens).
Estimated description-token reduction: **35.9%**.

## Error analysis
Original: 5 scenarios flagged for embedded tool errors.
Compressed: 8 scenarios flagged; scenarios 8, 10, and 113 also had invalid agent steps.

## Limitations
Pipeline completion is not equivalent to task accuracy. Some tools returned errors despite successful execution status. Missing dataset records also affected results. These are preliminary findings from a small benchmark, not statistically significant evidence that compression caused the failures.

The compressed condition was run twice; the saved JSON contains the latest run.

## Reproduction
Run from the repository root:

- Original: uv run python benchmark/tool_compression_experiment/run_qwen_comparison.py --mode original
- Compressed: uv run python benchmark/tool_compression_experiment/run_qwen_comparison.py --mode compressed

Re-running a condition overwrites its previous results file.
