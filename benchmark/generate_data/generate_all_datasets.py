#!/usr/bin/env python3
"""Generate all three training datasets for Gemma fine-tuning on AssetOpsBench.

Datasets:
  1. Tool Knowledge   - tool taxonomy, ownership, arguments, routing, hard negatives
  2. Planning          - scenario -> concise planning_steps (gold + synthetic)
  3. Execution         - scenario -> planning_steps + execution_steps + execution_links

Uses Gemini 2.5 Flash as the teacher model for synthetic generation.

Usage:
    cd /Users/yuvalshemla/Desktop/hpml_group_20
    uv run python benchmark/generate_data/generate_all_datasets.py
    uv run python benchmark/generate_data/generate_all_datasets.py --skip-synthetic  # gold-only, no API calls
"""

import argparse
import json
import os
import random
import re
import sys
import time
import hashlib
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import litellm
litellm.suppress_debug_info = True

# ─── Constants ────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TEACHER_MODEL = "gemini/gemini-2.5-flash"
OUTPUT_DIR = Path(__file__).parent / "datasets"

SEED = 42
random.seed(SEED)

# ─── Tool Catalog (ground truth) ─────────────────────────────────────────────

TOOL_CATALOG = {
    "IoTAgent": {
        "description": "Handles IoT telemetry data access — sites, assets, sensors, and historical readings from industrial equipment.",
        "tools": {
            "sites": {
                "description": "List all available IoT sites",
                "args": {},
                "returns": "List of site names",
            },
            "assets": {
                "description": "List all asset IDs for a given site",
                "args": {"site_name": "str - Name of the site"},
                "returns": "List of asset names/IDs at the site",
            },
            "sensors": {
                "description": "List sensor names for a specific asset at a site",
                "args": {"site_name": "str - Name of the site", "asset_id": "str - Asset identifier"},
                "returns": "List of sensor names",
            },
            "history": {
                "description": "Fetch historical sensor readings for a time range",
                "args": {
                    "site_name": "str - Name of the site",
                    "asset_id": "str - Asset identifier",
                    "start": "str - ISO 8601 start timestamp",
                    "final": "str (optional) - ISO 8601 end timestamp",
                },
                "returns": "Time-series sensor data as JSON",
            },
        },
    },
    "FMSRAgent": {
        "description": "Provides failure mode analysis and sensor relevance reasoning for industrial assets.",
        "tools": {
            "get_failure_modes": {
                "description": "Return known failure modes for an asset type",
                "args": {"asset_name": "str - Type of asset (e.g. 'chiller', 'ahu')"},
                "returns": "List of failure modes with descriptions",
            },
            "get_failure_mode_sensor_mapping": {
                "description": "For each failure mode and sensor pair, determine relevancy. Returns bidirectional mappings.",
                "args": {
                    "asset_name": "str - Type of asset",
                    "failure_modes": "list[str] - Failure modes to check",
                    "sensors": "list[str] - Sensors to check against",
                },
                "returns": "Bidirectional fm->sensors and sensor->fms maps with relevancy details",
            },
        },
    },
    "TSFMAgent": {
        "description": "Time-series forecasting and anomaly detection using IBM Granite TinyTimeMixer models.",
        "tools": {
            "get_ai_tasks": {
                "description": "List supported AI task types for time-series analysis",
                "args": {},
                "returns": "List of task types (forecasting, anomaly detection, etc.)",
            },
            "get_tsfm_models": {
                "description": "List available pre-trained TinyTimeMixer model checkpoints",
                "args": {},
                "returns": "List of model checkpoint names",
            },
            "run_tsfm_forecasting": {
                "description": "Run zero-shot time-series forecasting with a TTM model",
                "args": {
                    "dataset_path": "str - Path to the dataset CSV",
                    "timestamp_column": "str - Name of the timestamp column",
                    "target_columns": "list[str] - Columns to forecast",
                    "model_checkpoint": "str (optional) - Model to use",
                    "forecast_horizon": "int (optional) - Number of steps to forecast",
                },
                "returns": "Path to JSON predictions file",
            },
            "run_tsfm_finetuning": {
                "description": "Few-shot fine-tune a TTM model on domain data",
                "args": {
                    "dataset_path": "str - Path to the dataset CSV",
                    "timestamp_column": "str - Name of the timestamp column",
                    "target_columns": "list[str] - Columns to fine-tune on",
                    "model_checkpoint": "str (optional) - Base model to fine-tune",
                    "save_model_dir": "str (optional) - Where to save the fine-tuned model",
                },
                "returns": "Saved checkpoint path and metrics",
            },
            "run_tsad": {
                "description": "Conformal anomaly detection on top of forecasting output",
                "args": {
                    "dataset_path": "str - Path to the dataset CSV",
                    "tsfm_output_json": "str - Path to forecasting output JSON",
                    "timestamp_column": "str - Name of the timestamp column",
                    "target_columns": "list[str] - Columns to check for anomalies",
                },
                "returns": "CSV with anomaly labels",
            },
            "run_integrated_tsad": {
                "description": "End-to-end forecasting + anomaly detection in one call",
                "args": {
                    "dataset_path": "str - Path to the dataset CSV",
                    "timestamp_column": "str - Name of the timestamp column",
                    "target_columns": "list[str] - Columns to analyze",
                    "model_checkpoint": "str (optional) - Model to use",
                    "false_alarm": "float (optional) - False alarm rate threshold",
                },
                "returns": "Combined CSV with forecasts and anomaly labels",
            },
        },
    },
    "Utilities": {
        "description": "General utility functions — file reading, date/time queries.",
        "tools": {
            "json_reader": {
                "description": "Read and parse a JSON file from disk",
                "args": {"file_name": "str - Path to JSON file"},
                "returns": "Parsed JSON content",
            },
            "current_date_time": {
                "description": "Return the current UTC date and time as JSON",
                "args": {},
                "returns": "JSON with currentDateTime and description",
            },
            "current_time_english": {
                "description": "Return the current UTC time as a human-readable string",
                "args": {},
                "returns": "Human-readable date/time string",
            },
        },
    },
    "WorkOrderAgent": {
        "description": "Work order and maintenance event reasoning — queries historical maintenance records, failure codes, and predicts future work orders.",
        "tools": {
            "get_work_orders": {
                "description": "Retrieve work orders for an asset, optionally filtered by date range",
                "args": {"asset_name": "str - Asset identifier", "start_date": "str (optional)", "end_date": "str (optional)"},
                "returns": "List of work order records",
            },
            "get_preventive_work_orders": {
                "description": "Retrieve preventive maintenance work orders for an asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "List of preventive work orders",
            },
            "get_corrective_work_orders": {
                "description": "Retrieve corrective/reactive work orders for an asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "List of corrective work orders",
            },
            "get_events": {
                "description": "Retrieve maintenance events and alerts for an asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "List of events/alerts",
            },
            "get_failure_codes": {
                "description": "Retrieve failure codes associated with an asset's work orders",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "List of failure codes with descriptions",
            },
            "get_work_order_distribution": {
                "description": "Get distribution/statistics of work orders by type, priority, or time period",
                "args": {"asset_name": "str - Asset identifier", "group_by": "str (optional) - Grouping dimension"},
                "returns": "Work order distribution statistics",
            },
            "predict_next_work_order": {
                "description": "Predict when the next work order will be needed for an asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "Predicted date and type of next work order",
            },
            "analyze_alert_to_failure": {
                "description": "Analyze the relationship between alerts/events and subsequent failures",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "Alert-to-failure correlation analysis",
            },
        },
    },
    "VibrationAgent": {
        "description": "Vibration analysis for rotating equipment — FFT spectra, envelope analysis, bearing diagnostics, severity assessment.",
        "tools": {
            "get_vibration_data": {
                "description": "Retrieve raw vibration measurement data for a sensor",
                "args": {"sensor_id": "str - Vibration sensor identifier", "start": "str (optional)", "end": "str (optional)"},
                "returns": "Time-series vibration data",
            },
            "list_vibration_sensors": {
                "description": "List all vibration sensors for a given asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "List of vibration sensor IDs and locations",
            },
            "compute_fft_spectrum": {
                "description": "Compute FFT frequency spectrum from vibration data",
                "args": {"sensor_id": "str - Sensor identifier", "window_size": "int (optional)"},
                "returns": "Frequency spectrum data",
            },
            "compute_envelope_spectrum": {
                "description": "Compute envelope spectrum for bearing fault detection",
                "args": {"sensor_id": "str - Sensor identifier"},
                "returns": "Envelope spectrum data",
            },
            "assess_vibration_severity": {
                "description": "Assess vibration severity against ISO standards",
                "args": {"sensor_id": "str - Sensor identifier"},
                "returns": "Severity classification and RMS values",
            },
            "calculate_bearing_frequencies": {
                "description": "Calculate characteristic bearing fault frequencies",
                "args": {"bearing_model": "str - Bearing model/type", "shaft_rpm": "float - Shaft rotation speed"},
                "returns": "BPFO, BPFI, BSF, FTF frequencies",
            },
            "list_known_bearings": {
                "description": "List known bearing models in the database",
                "args": {},
                "returns": "List of bearing model numbers and specifications",
            },
            "diagnose_vibration": {
                "description": "Run comprehensive vibration diagnosis for an asset",
                "args": {"asset_name": "str - Asset identifier"},
                "returns": "Diagnosis report with fault indicators and recommendations",
            },
        },
    },
}

ALL_AGENTS = list(TOOL_CATALOG.keys())
ALL_TOOLS = {agent: list(info["tools"].keys()) for agent, info in TOOL_CATALOG.items()}
FLAT_TOOLS = [(agent, tool) for agent, tools in ALL_TOOLS.items() for tool in tools]


# ─── Gemini API Helper ───────────────────────────────────────────────────────

def call_gemini(prompt: str, temperature: float = 0.3, max_tokens: int = 4096, retries: int = 3) -> str | None:
    """Call Gemini with retries and rate limit handling."""
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=GEMINI_API_KEY,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"  API error: {err[:100]}")
                if attempt < retries - 1:
                    time.sleep(3)
                else:
                    return None
    return None


# ─── Dataset 1: Tool Knowledge ───────────────────────────────────────────────

def generate_tool_knowledge_dataset() -> list[dict]:
    """Generate tool knowledge QA pairs — taxonomy, ownership, args, routing, hard negatives."""
    examples = []

    # --- 1a. Agent listing ---
    examples.append({
        "instruction": "What are all the available sub-agents in the AssetOpsBench system?",
        "response": f"The available sub-agents are: {', '.join(ALL_AGENTS)}.",
        "category": "agent_listing",
    })

    # --- 1b. Per-agent tool listing ---
    for agent, info in TOOL_CATALOG.items():
        tools = list(info["tools"].keys())
        examples.append({
            "instruction": f"What tools does {agent} provide?",
            "response": f"{agent} provides these tools: {', '.join(tools)}. {info['description']}",
            "category": "tool_listing",
        })

    # --- 1c. Tool ownership (which agent owns this tool?) ---
    for agent, tools in ALL_TOOLS.items():
        for tool in tools:
            tool_info = TOOL_CATALOG[agent]["tools"][tool]
            examples.append({
                "instruction": f"Which agent owns the '{tool}' tool?",
                "response": f"The '{tool}' tool belongs to {agent}. It {tool_info['description'].lower()}.",
                "category": "tool_ownership",
            })

    # --- 1d. Tool arguments ---
    for agent, tools in ALL_TOOLS.items():
        for tool in tools:
            tool_info = TOOL_CATALOG[agent]["tools"][tool]
            args = tool_info["args"]
            if args:
                args_desc = "; ".join(f"{k}: {v}" for k, v in args.items())
                examples.append({
                    "instruction": f"What arguments does the {agent}.{tool} tool accept?",
                    "response": f"The {tool} tool ({agent}) accepts: {args_desc}. It returns: {tool_info['returns']}.",
                    "category": "tool_arguments",
                })
            else:
                examples.append({
                    "instruction": f"What arguments does the {agent}.{tool} tool accept?",
                    "response": f"The {tool} tool ({agent}) takes no arguments. It returns: {tool_info['returns']}.",
                    "category": "tool_arguments",
                })

    # --- 1e. Tool routing — which tool for this task? ---
    routing_pairs = [
        ("I need to find out what sites have IoT sensors deployed.", "IoTAgent", "sites"),
        ("List the equipment at the MAIN facility.", "IoTAgent", "assets"),
        ("What sensors are installed on Chiller 6?", "IoTAgent", "sensors"),
        ("Get temperature readings for Chiller 6 from January 2020.", "IoTAgent", "history"),
        ("What can go wrong with a chiller?", "FMSRAgent", "get_failure_modes"),
        ("Which sensors can detect compressor overheating?", "FMSRAgent", "get_failure_mode_sensor_mapping"),
        ("What time-series AI tasks are available?", "TSFMAgent", "get_ai_tasks"),
        ("Which forecasting models can I use?", "TSFMAgent", "get_tsfm_models"),
        ("Run a forecast on the chiller sensor data.", "TSFMAgent", "run_tsfm_forecasting"),
        ("Fine-tune the time-series model on my dataset.", "TSFMAgent", "run_tsfm_finetuning"),
        ("Detect anomalies in the sensor readings.", "TSFMAgent", "run_integrated_tsad"),
        ("What is the current date and time?", "Utilities", "current_date_time"),
        ("Read the configuration file.", "Utilities", "json_reader"),
        ("Show me work orders for Chiller 6.", "WorkOrderAgent", "get_work_orders"),
        ("What preventive maintenance is scheduled?", "WorkOrderAgent", "get_preventive_work_orders"),
        ("When is the next work order predicted?", "WorkOrderAgent", "predict_next_work_order"),
        ("What failure codes appear most often?", "WorkOrderAgent", "get_failure_codes"),
        ("Analyze the correlation between alerts and failures.", "WorkOrderAgent", "analyze_alert_to_failure"),
        ("Get vibration data for the compressor.", "VibrationAgent", "get_vibration_data"),
        ("Compute the FFT spectrum for bearing analysis.", "VibrationAgent", "compute_fft_spectrum"),
        ("Run a vibration diagnosis on the pump.", "VibrationAgent", "diagnose_vibration"),
        ("What bearing models are in the database?", "VibrationAgent", "list_known_bearings"),
        ("Assess vibration severity for sensor V-101.", "VibrationAgent", "assess_vibration_severity"),
    ]
    for query, agent, tool in routing_pairs:
        examples.append({
            "instruction": f"Which agent and tool should handle this request: \"{query}\"",
            "response": f"Agent: {agent}, Tool: {tool}. {TOOL_CATALOG[agent]['tools'][tool]['description']}.",
            "category": "tool_routing",
        })

    # --- 1f. Hard negatives — wrong agent/tool for the task ---
    hard_negatives = [
        ("I need sensor data for Chiller 6.", "FMSRAgent", "get_failure_modes",
         "IoTAgent", "sensors",
         "This is a sensor data query, not a failure mode query. Use IoTAgent.sensors to list sensors, or IoTAgent.history for readings."),
        ("Predict when the chiller will need maintenance.", "IoTAgent", "history",
         "WorkOrderAgent", "predict_next_work_order",
         "Maintenance prediction requires work order analysis, not raw sensor history. Use WorkOrderAgent.predict_next_work_order."),
        ("What failure modes affect the AHU?", "IoTAgent", "assets",
         "FMSRAgent", "get_failure_modes",
         "Failure mode information comes from FMSRAgent, not IoTAgent. IoTAgent handles telemetry data, not failure analysis."),
        ("Run anomaly detection on the pump.", "FMSRAgent", "get_failure_mode_sensor_mapping",
         "TSFMAgent", "run_integrated_tsad",
         "Anomaly detection is a time-series task handled by TSFMAgent, not FMSRAgent. FMSRAgent handles failure mode analysis."),
        ("What is the FFT spectrum for the motor bearing?", "TSFMAgent", "run_tsfm_forecasting",
         "VibrationAgent", "compute_fft_spectrum",
         "FFT spectrum computation is a vibration analysis task. Use VibrationAgent.compute_fft_spectrum, not TSFMAgent."),
        ("Get corrective work orders for Chiller 6.", "IoTAgent", "history",
         "WorkOrderAgent", "get_corrective_work_orders",
         "Work order retrieval requires WorkOrderAgent, not IoTAgent. IoTAgent handles sensor data, not maintenance records."),
    ]
    for query, wrong_agent, wrong_tool, correct_agent, correct_tool, explanation in hard_negatives:
        examples.append({
            "instruction": f"A system incorrectly routed the query \"{query}\" to {wrong_agent}.{wrong_tool}. What is the correct routing and why?",
            "response": f"Incorrect: {wrong_agent}.{wrong_tool}. Correct: {correct_agent}.{correct_tool}. {explanation}",
            "category": "hard_negative",
        })

    # --- 1g. Clarification / abstention ---
    clarification_examples = [
        ("Get sensor data.", "This query is incomplete. I need to know: which site (e.g., MAIN), which asset (e.g., Chiller 6), and optionally a time range. Please provide more details."),
        ("Run the model.", "This query is ambiguous. Do you want to: (a) run forecasting with TSFMAgent.run_tsfm_forecasting, (b) run anomaly detection with TSFMAgent.run_integrated_tsad, or (c) fine-tune a model with TSFMAgent.run_tsfm_finetuning? Please specify the task and dataset."),
        ("What happened to the equipment?", "This query is too vague. Please specify: (a) which equipment/asset, (b) what type of information you need (sensor data, work orders, failure modes, events), and (c) a time period if relevant."),
        ("Fix the chiller.", "I cannot execute physical maintenance actions. I can help you with: (a) diagnosing the issue using sensor data (IoTAgent), (b) identifying failure modes (FMSRAgent), (c) predicting when maintenance is needed (WorkOrderAgent), or (d) detecting anomalies (TSFMAgent). What analysis would be helpful?"),
        ("Is the pump going to fail?", "To assess failure risk, I need more context: (a) which specific pump/asset, (b) at which site, and (c) what type of analysis — anomaly detection on sensor data (TSFMAgent), failure mode analysis (FMSRAgent), or work order history (WorkOrderAgent)?"),
    ]
    for query, response in clarification_examples:
        examples.append({
            "instruction": query,
            "response": response,
            "category": "clarification",
        })

    return examples


# ─── Dataset 2: Planning ─────────────────────────────────────────────────────

def load_gold_plans() -> list[dict]:
    """Load and parse Gemini gold plans."""
    path = Path(__file__).parent.parent / "baseline_tests" / "gemini_flash_informed_results.json"
    with open(path) as f:
        data = json.load(f)

    plans = []
    for entry in data:
        if entry.get("plan_steps", 0) > 0 and entry.get("response"):
            plans.append({
                "id": entry["id"],
                "prompt": entry["prompt"],
                "plan": entry["response"],
                "num_steps": entry["plan_steps"],
                "agents": entry.get("agents_used", []),
                "tools": entry.get("tools_used", []),
            })
    return plans


def load_hf_scenarios() -> list[dict]:
    """Load scenarios from HuggingFace dataset."""
    from datasets import load_dataset
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    return [dict(row) for row in ds["train"]]


def build_planning_dataset_from_gold(gold_plans: list[dict]) -> list[dict]:
    """Convert gold plans into planning SFT examples."""
    examples = []
    for plan in gold_plans:
        # Clean up the plan: remove "none" agent steps (summary steps from Gemini)
        cleaned_plan = plan["plan"]

        examples.append({
            "instruction": plan["prompt"],
            "response": cleaned_plan,
            "category": "planning",
            "source": "gold",
            "num_steps": plan["num_steps"],
            "scenario_id": plan["id"],
        })
    return examples


def generate_plan_paraphrases(gold_plans: list[dict], num_per_scenario: int = 3) -> list[dict]:
    """Generate paraphrased questions with the same gold plan."""
    examples = []
    total = len(gold_plans)

    for i, plan in enumerate(gold_plans):
        if i % 10 == 0:
            print(f"  Paraphrasing {i}/{total}...")

        prompt = f"""Rewrite this industrial asset operations question in {num_per_scenario} different ways.
Keep the same meaning and intent. Vary the phrasing, formality, and specificity.
Return each paraphrase on its own line, numbered 1-{num_per_scenario}. No other text.

Question: {plan['prompt']}"""

        resp = call_gemini(prompt, temperature=0.7, max_tokens=512)
        if not resp:
            continue

        for line in resp.strip().split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(line) > 10 and line != plan["prompt"]:
                examples.append({
                    "instruction": line,
                    "response": plan["plan"],
                    "category": "planning",
                    "source": "paraphrase",
                    "num_steps": plan["num_steps"],
                    "scenario_id": plan["id"],
                })

        # Rate limit
        time.sleep(1)

    return examples


def generate_synthetic_plans_for_expanded(scenarios: list[dict], existing_ids: set) -> list[dict]:
    """Generate plans for expanded dataset scenarios (FMSR, PHM, rule_logic, etc.)."""
    examples = []

    # Build tool catalog string for the prompt
    tool_desc = []
    for agent, info in TOOL_CATALOG.items():
        tool_desc.append(f"\n{agent}: {info['description']}")
        for tool, tinfo in info["tools"].items():
            args_str = ", ".join(f"{k}" for k in tinfo["args"].keys()) if tinfo["args"] else "none"
            tool_desc.append(f"  - {tool}({args_str}): {tinfo['description']}")
    tool_catalog_str = "\n".join(tool_desc)

    total = len(scenarios)
    for i, sc in enumerate(scenarios):
        if sc.get("id") in existing_ids:
            continue
        if i % 10 == 0:
            print(f"  Generating plans for expanded scenarios {i}/{total}...")

        prompt = f"""You are an expert planner for industrial asset operations. Given the question below and the available tools, produce a structured plan.

AVAILABLE TOOLS:
{tool_catalog_str}

OUTPUT FORMAT (exactly this structure, no extra text):
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Use ONLY agents and tools from the catalog above
- Keep plans concise — use the minimum number of steps
- Use {{step_N}} placeholders when a later step needs results from an earlier step
- If the question is ambiguous or insufficient, respond ONLY with: CLARIFICATION: <what info is needed>

QUESTION: {sc['text']}"""

        resp = call_gemini(prompt, temperature=0.1, max_tokens=2048)
        if not resp:
            continue

        # Check if it's a clarification response
        if resp.strip().startswith("CLARIFICATION:"):
            examples.append({
                "instruction": sc["text"],
                "response": resp.strip(),
                "category": "planning_clarification",
                "source": "synthetic_expanded",
                "scenario_id": sc.get("id"),
            })
        elif "#Task1:" in resp and "#Agent1:" in resp:
            # Count steps
            num_steps = len(re.findall(r"#Task\d+:", resp))
            examples.append({
                "instruction": sc["text"],
                "response": resp.strip(),
                "category": "planning",
                "source": "synthetic_expanded",
                "num_steps": num_steps,
                "scenario_id": sc.get("id"),
            })

        time.sleep(1)

    return examples


# ─── Dataset 3: Execution-Structured ─────────────────────────────────────────

def generate_execution_dataset(gold_plans: list[dict]) -> list[dict]:
    """Generate execution-structured examples: plan + execution steps + execution links."""
    examples = []
    total = len(gold_plans)

    for i, plan in enumerate(gold_plans):
        if i % 10 == 0:
            print(f"  Generating execution traces {i}/{total}...")

        prompt = f"""Given this question and plan, generate the execution trace showing how each step would be executed.

QUESTION: {plan['prompt']}

PLAN:
{plan['plan']}

Generate output in this exact format:

PLANNING_STEPS:
<copy the plan steps exactly as given>

EXECUTION_STEPS:
Step 1: Call <Agent>.<tool>(<args>) -> <expected result description>
Step 2: ...
(for each step, show the actual tool call with resolved arguments where possible)

EXECUTION_LINKS:
Step 1 -> Step 2: <what data flows from step 1 to step 2>
(only include links where one step depends on another's output)

Be concise. Only include steps that actually call tools (skip summary/reasoning steps)."""

        resp = call_gemini(prompt, temperature=0.1, max_tokens=2048)
        if not resp:
            continue

        if "PLANNING_STEPS:" in resp and "EXECUTION_STEPS:" in resp:
            examples.append({
                "instruction": plan["prompt"],
                "response": resp.strip(),
                "category": "execution",
                "source": "synthetic_execution",
                "num_steps": plan["num_steps"],
                "scenario_id": plan["id"],
            })

        time.sleep(1)

    return examples


# ─── Deduplication & Validation ───────────────────────────────────────────────

def deduplicate(examples: list[dict]) -> list[dict]:
    """Deduplicate by instruction content hash."""
    seen = set()
    unique = []
    for ex in examples:
        h = hashlib.md5(ex["instruction"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def validate_plan(plan_text: str) -> dict:
    """Validate a plan string and return quality metrics."""
    issues = []
    if "#Task" not in plan_text and "CLARIFICATION:" not in plan_text:
        issues.append("missing_task_tags")

    # Check for valid agents
    agents_found = re.findall(r"#Agent\d+:\s*(\S+)", plan_text)
    valid_agents = set(ALL_AGENTS) | {"none", ""}
    for a in agents_found:
        if a not in valid_agents:
            issues.append(f"unknown_agent:{a}")

    # Check for valid tools
    tools_found = re.findall(r"#Tool\d+:\s*(\S+)", plan_text)
    valid_tools = {t for _, t in FLAT_TOOLS} | {"none", ""}
    for t in tools_found:
        t_clean = t.rstrip("()")  # handle sites() format
        if t_clean not in valid_tools:
            issues.append(f"unknown_tool:{t}")

    # Check args are valid JSON
    args_found = re.findall(r"#Args\d+:\s*(.+)", plan_text)
    for arg_str in args_found:
        arg_str = arg_str.strip()
        if arg_str and arg_str != "{}":
            try:
                json.loads(arg_str)
            except json.JSONDecodeError:
                issues.append("invalid_json_args")

    return {"valid": len(issues) == 0, "issues": issues, "num_steps": len(re.findall(r"#Task\d+:", plan_text))}


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def save_jsonl(examples: list[dict], path: Path, format_type: str = "chat"):
    """Save examples as JSONL in chat format for SFT training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            if format_type == "chat":
                record = {
                    "messages": [
                        {"role": "user", "content": ex["instruction"]},
                        {"role": "assistant", "content": ex["response"]},
                    ],
                    "metadata": {k: v for k, v in ex.items() if k not in ("instruction", "response")},
                }
            else:
                record = ex
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples to {path}")


def print_stats(name: str, examples: list[dict]):
    """Print dataset statistics."""
    cats = Counter(ex.get("category", "unknown") for ex in examples)
    sources = Counter(ex.get("source", "unknown") for ex in examples)
    print(f"\n{'='*60}")
    print(f"  {name}: {len(examples)} examples")
    print(f"{'='*60}")
    print(f"  Categories: {dict(cats.most_common())}")
    print(f"  Sources:    {dict(sources.most_common())}")

    # Validate plans
    plan_examples = [ex for ex in examples if "planning" in ex.get("category", "") and "clarification" not in ex.get("category", "")]
    if plan_examples:
        validations = [validate_plan(ex["response"]) for ex in plan_examples]
        valid = sum(1 for v in validations if v["valid"])
        print(f"  Plan validation: {valid}/{len(validations)} valid ({100*valid/len(validations):.1f}%)")
        all_issues = Counter()
        for v in validations:
            for issue in v["issues"]:
                all_issues[issue] += 1
        if all_issues:
            print(f"  Issues: {dict(all_issues.most_common(10))}")


def main():
    parser = argparse.ArgumentParser(description="Generate training datasets for Gemma fine-tuning")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic generation (gold-only, no API calls)")
    parser.add_argument("--paraphrases-per-scenario", type=int, default=3, help="Number of paraphrases per gold plan")
    parser.add_argument("--expanded-configs", nargs="*", default=["failure_mode_sensor_mapping", "compressor", "hydrolic_pump", "prognostics_and_health_management"],
                        help="HF dataset configs to include for expanded scenarios")
    args = parser.parse_args()

    print("="*60)
    print("  AssetOpsBench Training Data Generation")
    print("="*60)

    # ─── Load sources ─────────────────────────────────────────────────
    print("\n[1/6] Loading gold plans...")
    gold_plans = load_gold_plans()
    print(f"  Loaded {len(gold_plans)} gold plans from Gemini 2.5 Flash")

    print("\n[2/6] Loading HF scenarios...")
    hf_scenarios = load_hf_scenarios()
    print(f"  Loaded {len(hf_scenarios)} scenarios from HuggingFace")
    gold_ids = {p["id"] for p in gold_plans}

    # ─── Dataset 1: Tool Knowledge ────────────────────────────────────
    print("\n[3/6] Generating tool knowledge dataset...")
    tool_knowledge = generate_tool_knowledge_dataset()
    tool_knowledge = deduplicate(tool_knowledge)
    print_stats("Dataset 1: Tool Knowledge", tool_knowledge)
    save_jsonl(tool_knowledge, OUTPUT_DIR / "tool_knowledge.jsonl")

    # ─── Dataset 2: Planning ──────────────────────────────────────────
    print("\n[4/6] Generating planning dataset...")
    planning_data = build_planning_dataset_from_gold(gold_plans)
    print(f"  Gold plans: {len(planning_data)} examples")

    if not args.skip_synthetic:
        # Paraphrases of gold plans
        print(f"  Generating {args.paraphrases_per_scenario} paraphrases per scenario...")
        paraphrases = generate_plan_paraphrases(gold_plans, args.paraphrases_per_scenario)
        planning_data.extend(paraphrases)
        print(f"  Paraphrases: {len(paraphrases)} examples")

        # Plans for expanded scenarios
        if args.expanded_configs:
            print(f"  Loading expanded scenarios from HF configs: {args.expanded_configs}")
            expanded_scenarios = []
            for config in args.expanded_configs:
                try:
                    from datasets import load_dataset
                    ds = load_dataset("ibm-research/AssetOpsBench", config)
                    for row in ds["train"]:
                        expanded_scenarios.append(dict(row))
                    print(f"    {config}: {len(ds['train'])} scenarios")
                except Exception as e:
                    print(f"    {config}: failed to load ({e})")

            if expanded_scenarios:
                expanded_plans = generate_synthetic_plans_for_expanded(expanded_scenarios, gold_ids)
                planning_data.extend(expanded_plans)
                print(f"  Expanded plans: {len(expanded_plans)} examples")

    planning_data = deduplicate(planning_data)
    print_stats("Dataset 2: Planning", planning_data)
    save_jsonl(planning_data, OUTPUT_DIR / "planning.jsonl")

    # ─── Dataset 3: Execution ─────────────────────────────────────────
    print("\n[5/6] Generating execution dataset...")
    if args.skip_synthetic:
        execution_data = []
        print("  Skipped (--skip-synthetic)")
    else:
        execution_data = generate_execution_dataset(gold_plans)
        execution_data = deduplicate(execution_data)

    print_stats("Dataset 3: Execution", execution_data)
    if execution_data:
        save_jsonl(execution_data, OUTPUT_DIR / "execution.jsonl")

    # ─── Combined dataset ─────────────────────────────────────────────
    print("\n[6/6] Building combined dataset...")
    all_data = tool_knowledge + planning_data + execution_data
    all_data = deduplicate(all_data)
    save_jsonl(all_data, OUTPUT_DIR / "combined_all.jsonl")

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  GENERATION COMPLETE")
    print("="*60)
    print(f"  Tool Knowledge: {len(tool_knowledge):>6} examples  -> datasets/tool_knowledge.jsonl")
    print(f"  Planning:       {len(planning_data):>6} examples  -> datasets/planning.jsonl")
    print(f"  Execution:      {len(execution_data):>6} examples  -> datasets/execution.jsonl")
    print(f"  Combined:       {len(all_data):>6} examples  -> datasets/combined_all.jsonl")
    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
