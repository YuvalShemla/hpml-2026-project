# Gemma 4 E2E Dashboard Guide

## How to Read the Dashboard

### Summary Cards (Top)

| Card | What It Means |
|------|---------------|
| **Scenarios Ran** | How many questions were successfully processed end-to-end |
| **Full E2E Success** | Scenarios where every tool step executed without errors |
| **Tool Steps OK** | Total individual MCP tool calls that succeeded / total attempted |
| **Avg LLM Calls/Scenario** | Average number of LLM invocations per question (plan + arg resolve + summarize) |
| **Avg Tool Calls/Scenario** | Average number of MCP tool invocations per question |
| **Avg Plan Match (vs Gold)** | Jaccard similarity of (agent, tool) pairs between Gemma 4's plan and the Gemini 2.5 Flash gold plan (see "Plan Match Scoring" below) |
| **Avg Tokens/Scenario** | Mean total tokens (input + output) across all LLM calls per scenario |
| **Tool Desc % of Input** | What fraction of ALL input tokens (across all LLM calls) are tool descriptions. Only the plan call includes tool descriptions; arg resolve and summarize calls don't. So this % is diluted by non-plan calls. For plan calls alone, tool descriptions are ~95% of input. |
| **Active Params (MoE 26B)** | Gemma 4 is a Mixture-of-Experts model: 26B total parameters but only 3.8B active per inference |

### Token Stats Bar

Shows aggregate token counts across all scenarios:
- **Total tokens**: input + output across all LLM calls
- **Input / Output**: breakdown
- **Tool descriptions (est)**: estimated tokens consumed by tool/agent descriptions in the plan prompt. This is the overhead that fine-tuning eliminates (the model learns the tools, so they don't need to be in the prompt)

### Per-Scenario View

Click any scenario header to expand it. Each expanded scenario shows:

#### Pipeline Visualization

Each block in the pipeline is color-coded:
- **Purple blocks** = LLM calls (Gemma 4 inference)
- **Blue blocks** = MCP tool calls (actual tool execution via MCP servers)
- Green left border = tool call succeeded
- Red left border = tool call failed

The pipeline reads top-to-bottom:

```
LLM CALL #1: PLAN GENERATION
  ↓
[MCP TOOL: agent.tool_name]       ← Step 1 (no arg resolution needed)
  ↓
LLM CALL #2: ARG RESOLVE          ← Only if step has {step_N} placeholders
  ↓
[MCP TOOL: agent.tool_name]       ← Step 2 (with resolved args)
  ↓
...more steps...
  ↓
LLM CALL #N: SUMMARIZE            ← Always the last LLM call
```

Each LLM call shows a **token badge**: `X in / Y out = Z tok`
- "in" = input tokens (prompt size sent to Gemma 4)
- "out" = output tokens (Gemma 4's response)

Each MCP tool call shows:
- Agent name (colored badge: blue=IoT, purple=FMSR, amber=TSFM, gray=Utilities)
- Tool name (monospace)
- Arguments sent to the tool (JSON)
- Response or error from the tool

#### Final Answer

The natural language answer Gemma 4 produced after synthesizing all tool results.

#### Evaluation Panel (bottom)

Split into two columns:

**Left: Gold Plan** — The reference plan from Gemini 2.5 Flash (our gold standard). Shows the "ideal" decomposition of the question into agent-tool steps.

**Right: Plan Quality Assessment**
- **Agent-Tool Match**: Jaccard similarity (see below)
- **Step Count**: Gemma 4 steps vs gold steps
- **Execution**: Whether all tool calls succeeded
- **LLM Calls**: Total LLM invocations for this scenario
- **Total Tokens**: Combined input + output tokens
- **Tool Desc Overhead**: Estimated tokens for tool descriptions in the plan prompt
- **Time**: Wall-clock time including API rate limit delays
- **Notes**: "Over-decomposed" (more steps than gold), "Under-decomposed" (fewer), or "Step count matches"

### Plan Match Scoring

The match score uses **Jaccard similarity on unique (agent, tool) pairs**:

```
score = |gold_pairs ∩ model_pairs| / |gold_pairs ∪ model_pairs|
```

Example (ID 8 — Chiller 6 Tonnage, 67% match):
- Gold pairs: {(IoTAgent, assets), (IoTAgent, history)} — 2 unique pairs
- Gemma 4 pairs: {(IoTAgent, assets), (IoTAgent, sensors), (IoTAgent, history)} — 3 unique pairs
- Intersection: 2, Union: 3
- Score: 2/3 = 67%

The `sensors` call is the "extra" — Gemma 4 added an intermediate step to verify the Tonnage sensor exists. This is defensively reasonable but unnecessary per the gold plan, so the Jaccard metric penalizes it.

### Keyboard Navigation

- **n** or **Down Arrow**: Open next scenario
- **p** or **Up Arrow**: Open previous scenario

---

## Where Is the Gemma 4 Model Used?

### Model Identity

- **Model**: Gemma 4 26B-A4B-IT (instruction-tuned)
- **Architecture**: Mixture of Experts — 26B total params, 3.8B active per forward pass
- **Access**: Via Google's Gemini API (not self-hosted)
- **Model ID string**: `gemini/gemma-4-26b-a4b-it`
- **API Key**: `GEMINI_API_KEY` environment variable (set in `.env`)

### Where It's Called

Gemma 4 is called through `src/llm/litellm.py:LiteLLMBackend.generate()`. The `gemini/` prefix routes the request to the Gemini API via the litellm library.

Every LLM call in the pipeline goes through this single backend:

| Call | Who Triggers It | File | Purpose |
|------|----------------|------|---------|
| **PLAN** | `Planner.generate_plan()` | `src/workflow/planner.py` | Decompose question into tool steps |
| **ARG RESOLVE** | `Executor.execute_step()` | `src/workflow/executor.py` | Replace `{step_N}` placeholders with actual values |
| **SUMMARIZE** | `PlanExecuteRunner.run()` | `src/workflow/runner.py` | Synthesize tool results into final answer |

### Rate Limits

The Gemini API limits Gemma 4 to **16,000 input tokens per minute**. The E2E runner (`run_gemma4_e2e.py`) adds 45-second delays between scenarios and retries with exponential backoff on 429 errors.

---

## How AssetOpsBench Works

### Overview

AssetOpsBench is IBM's benchmark for evaluating AI agents on industrial asset operations. It tests whether an LLM can:
1. Understand a natural language question about industrial equipment
2. Decompose it into a plan of tool calls
3. Execute those tools via MCP (Model Context Protocol) servers
4. Synthesize results into a human-readable answer

### Domain Agents & Tools (15 tools across 4 agents)

| Agent | MCP Server | Tools | Data Source |
|-------|-----------|-------|-------------|
| **IoTAgent** | `src/servers/iot_mcp.py` | `sites`, `assets`, `sensors`, `history` | CouchDB (Docker) |
| **FMSRAgent** | `src/servers/fmsr_mcp.py` | `get_failure_modes`, `get_failure_mode_sensor_mapping` | YAML files + WatsonX LLM fallback |
| **TSFMAgent** | `src/servers/tsfm_mcp.py` | `get_ai_tasks`, `get_tsfm_models`, `run_tsfm_forecasting`, `run_tsfm_finetuning`, `run_tsad`, `run_integrated_tsad` | IBM Granite TTM models |
| **Utilities** | `src/servers/util_mcp.py` | `json_reader`, `current_date_time`, `current_time_english` | Local |

### The Plan-Execute Pipeline (4 phases)

```
QUESTION
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ 1. DISCOVER                                              │
│    executor.get_agent_descriptions()                     │
│    → Connect to each MCP server via stdio                │
│    → Call session.list_tools() on each                   │
│    → Collect tool names, signatures, descriptions        │
│    → Output: dict of agent → tool descriptions text      │
│    (No LLM call — just MCP protocol)                     │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ 2. PLAN                                     [LLM CALL]   │
│    planner.generate_plan(question, agent_descriptions)   │
│    → Format prompt: question + all tool descriptions     │
│    → LLM generates structured plan:                      │
│        #Task1: description                               │
│        #Agent1: IoTAgent                                 │
│        #Tool1: assets                                    │
│        #Args1: {"site_name": "MAIN"}                     │
│        #Dependency1: None                                │
│    → parse_plan() extracts via regex into Plan object    │
│    → {step_N} placeholders left unresolved               │
│                                                          │
│    This is the most expensive call: ~2,375 input tokens  │
│    because ALL tool descriptions are included (~2,268    │
│    tokens = 95% of plan input).                          │
│    THIS IS WHAT FINE-TUNING ELIMINATES.                  │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ 3. EXECUTE (per step, in dependency order)               │
│    executor.execute_plan(plan, question)                 │
│                                                          │
│    For each step (topologically sorted):                 │
│                                                          │
│    a) If args contain {step_N} placeholders:             │
│       → _resolve_args_with_llm()         [LLM CALL]     │
│         Prompt: "Given these prior results, what are     │
│         the actual values for these args?"               │
│         ~170-200 input tokens (lightweight)              │
│                                                          │
│    b) Call MCP tool:                                     │
│       → _call_tool(server_path, tool, resolved_args)     │
│         Spawns MCP server via stdio, calls tool,         │
│         returns JSON response                            │
│                                                          │
│    c) Store result in context dict for later steps       │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ 4. SUMMARIZE                                [LLM CALL]   │
│    runner.run() → llm.generate(summarize_prompt)         │
│    → Input: question + all step results concatenated     │
│    → Output: natural language answer                     │
│    → Token cost varies wildly: 109 tokens (simple) to   │
│      13,934 tokens (when FMSR returns huge mappings)     │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
                   FINAL ANSWER
```

### The {step_N} Placeholder Mechanism

When the planner generates a multi-step plan, later steps often need results from earlier ones. For example:

```
Step 1: Get assets at MAIN         → Args: {"site_name": "MAIN"}
Step 2: Get sensors for Chiller 6  → Args: {"site_name": "MAIN", "asset_id": "{step_1}"}
```

The `{step_1}` is a placeholder meaning "use the result from step 1." During execution:

1. **Detect**: `executor._has_placeholders()` uses regex `r"\{step_(\d+)\}"` to find placeholders
2. **Collect context**: Gather results from referenced prior steps
3. **LLM resolve**: Ask the LLM "Given step 1 returned `['Chiller 6']`, what should `asset_id` be?" The LLM responds with `{"asset_id": "Chiller 6"}`
4. **Merge**: Combine resolved values with non-placeholder args
5. **Call tool**: Execute with fully concrete arguments

This is why multi-step plans require additional LLM calls beyond just plan + summarize.

---

## File Locations & How to Run

### Repository Structure

```
AssetOpsBenchGroup20/
├── src/
│   ├── llm/
│   │   ├── base.py              # LLMBackend abstract base class
│   │   └── litellm.py           # LiteLLMBackend (routes to Gemini/WatsonX/proxy)
│   ├── workflow/
│   │   ├── cli.py               # CLI entry point (`plan-execute` command)
│   │   ├── runner.py            # PlanExecuteRunner orchestrator
│   │   ├── planner.py           # Plan generation + parsing
│   │   └── executor.py          # Step execution + arg resolution + MCP calls
│   ├── servers/
│   │   ├── iot_mcp.py           # IoTAgent MCP server (CouchDB)
│   │   ├── fmsr_mcp.py          # FMSRAgent MCP server (failure modes)
│   │   ├── tsfm_mcp.py          # TSFMAgent MCP server (time series)
│   │   └── util_mcp.py          # Utilities MCP server
│   └── couchdb/
│       └── docker-compose.yaml  # CouchDB for IoTAgent data
├── benchmark/
│   ├── baseline_tests/
│   │   ├── run_gemma4_e2e.py    # Batch E2E runner with token tracking
│   │   ├── build_dashboard.py   # HTML dashboard generator
│   │   ├── gemma4_e2e_results.json    # Raw results (5 scenarios)
│   │   ├── gemma4_e2e_dashboard.html  # The interactive dashboard
│   │   └── DASHBOARD_GUIDE.md   # This file
│   ├── cods_track1/             # Planning-only benchmark (no execution)
│   └── cods_track2/             # Legacy AgentHive execution (NOT used)
└── .env                         # API keys (GEMINI_API_KEY, etc.)
```

### Prerequisites

```bash
# 1. Install dependencies
cd AssetOpsBenchGroup20
uv sync

# 2. Start CouchDB (required for IoTAgent tools)
docker compose -f src/couchdb/docker-compose.yaml up -d

# 3. Set API key in .env
echo 'GEMINI_API_KEY=your_key_here' >> .env
```

### Running a Single Query

```bash
uv run plan-execute \
  --model-id "gemini/gemma-4-26b-a4b-it" \
  --show-plan \
  --show-history \
  "What IoT sites are available?"
```

Flags:
- `--model-id`: LiteLLM model string (prefix determines API: `gemini/`, `watsonx/`, or bare for proxy)
- `--show-plan`: Print the generated plan before execution
- `--show-history`: Print each step's execution result
- `--json`: Output as JSON instead of text

### Running the Batch E2E Evaluation

```bash
# Run 5 scenarios with token tracking (45s delays for rate limits)
uv run python benchmark/baseline_tests/run_gemma4_e2e.py
```

This reads scenarios from `/tmp/e2e_test_scenarios_small.json`, runs each through the full pipeline, captures token usage per LLM call, and saves results to `gemma4_e2e_results.json`.

### Building / Rebuilding the Dashboard

```bash
# Generate HTML from results JSON
uv run python benchmark/baseline_tests/build_dashboard.py

# Open in browser
open benchmark/baseline_tests/gemma4_e2e_dashboard.html
```

### What Happens at Each Step (Concrete Example)

**Question**: "Download sensor data for Chiller 6's Tonnage from the last week of 2020 at the MAIN site"

| Phase | What Happens | Tokens | Time |
|-------|-------------|--------|------|
| **DISCOVER** | Connect to 4 MCP servers, list all 15 tools | 0 (no LLM) | ~2s |
| **PLAN** (LLM #1) | Gemma 4 receives question + 15 tool descriptions (~2,268 tokens). Produces 3-step plan: assets → sensors → history | 2,395 in / 4,676 out | ~5s |
| **EXECUTE Step 1** | MCP call: IoTAgent.assets({"site_name": "MAIN"}) → returns ["Chiller 6"] | 0 (tool call) | ~3s |
| **ARG RESOLVE** (LLM #2) | Resolve {"asset_id": "{step_1}"} → {"asset_id": "Chiller 6"} | 177 in / 269 out | ~3s |
| **EXECUTE Step 2** | MCP call: IoTAgent.sensors({"site_name": "MAIN", "asset_id": "Chiller 6"}) → 11 sensors | 0 (tool call) | ~3s |
| **ARG RESOLVE** (LLM #3) | Resolve {"asset_id": "{step_1}"} for history call | 204 in / 289 out | ~3s |
| **EXECUTE Step 3** | MCP call: IoTAgent.history({...}) → 0 observations (no data in that range) | 0 (tool call) | ~3s |
| **SUMMARIZE** (LLM #4) | Synthesize: "No sensor data was found for Chiller 6's Tonnage..." | 560 in / 316 out | ~3s |
| **TOTAL** | 4 LLM calls + 3 MCP tool calls | 8,886 tokens | 121s (includes rate limit delays) |

---

## Why Fine-Tuning Matters

The plan call includes ~2,268 tokens of tool descriptions every single time. This is:
- **95% of plan call input tokens**
- **~40% of all input tokens** across all calls (diluted by arg resolve + summarize)
- **Constant regardless of question complexity**

After fine-tuning, the model has internalized the tool catalog. It can produce correct plans without the tool descriptions in the prompt, saving ~2,268 tokens per question. For a 152-scenario benchmark, that's ~345K tokens saved.
