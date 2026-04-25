#!/usr/bin/env python3
"""Scale up the dataset to ~2,500 examples for better training convergence.

Main strategy: high-volume paraphrasing (reuses validated plans) + more tool knowledge.

Usage:
    uv run python benchmark/generate_data/scale_up_dataset.py
"""

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini/gemini-2.5-flash"
SEED = 123
random.seed(SEED)
DATASETS_DIR = Path(__file__).parent / "datasets"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def call_gemini(prompt, temperature=0.7, max_tokens=2048, retries=3):
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
                api_key=GEMINI_API_KEY,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
            elif attempt < retries - 1:
                time.sleep(3)
            else:
                return None
    return None


def make_example(instruction, response, category, source):
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"category": category, "source": source},
    }


def deduplicate_against(new_examples, existing_hashes):
    """Deduplicate new examples against existing data."""
    unique = []
    for ex in new_examples:
        h = hashlib.md5(ex["messages"][0]["content"].strip().lower().encode()).hexdigest()
        if h not in existing_hashes:
            existing_hashes.add(h)
            unique.append(ex)
    return unique


# ─── Load existing data hashes ────────────────────────────────────────────────

def load_existing_hashes():
    hashes = set()
    for fname in ["tool_knowledge.jsonl", "planning.jsonl", "execution.jsonl"]:
        path = DATASETS_DIR / fname
        if path.exists():
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    h = hashlib.md5(d["messages"][0]["content"].strip().lower().encode()).hexdigest()
                    hashes.add(h)
    return hashes


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BATCH PARAPHRASES — 5 per plan, batched 5 plans at a time for efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def generate_batch_paraphrases(plans, num_per_plan=5, batch_size=5):
    """Generate paraphrases efficiently by batching multiple plans per API call."""
    examples = []
    total = len(plans)

    for batch_start in range(0, total, batch_size):
        batch = plans[batch_start:batch_start + batch_size]
        batch_idx = batch_start // batch_size

        if batch_idx % 5 == 0:
            print(f"    Batch {batch_idx}, plans {batch_start}/{total}, {len(examples)} examples so far...")

        # Build batch prompt
        questions_block = ""
        for j, plan in enumerate(batch):
            questions_block += f"\nQ{j+1}: {plan['prompt']}\n"

        prompt = f"""Generate {num_per_plan} paraphrases for EACH of the following industrial asset operations questions.

IMPORTANT CONTEXT:
- "sites" = physical industrial monitoring sites (e.g., "MAIN site"), NOT websites
- "assets" = industrial equipment (chillers, AHUs, pumps, compressors), NOT financial assets
- Preserve all specific entity names (site names, asset names, sensor names, dates)
- Vary phrasing, formality, and sentence structure
- Keep the same meaning and intent

{questions_block}

For each question, output exactly {num_per_plan} paraphrases:
Q1-P1: <paraphrase>
Q1-P2: <paraphrase>
...
Q{len(batch)}-P{num_per_plan}: <paraphrase>

Output ONLY the paraphrases in this format, nothing else."""

        resp = call_gemini(prompt, temperature=0.8, max_tokens=4096)
        if not resp:
            time.sleep(1)
            continue

        # Parse responses per question
        for j, plan in enumerate(batch):
            qnum = j + 1
            pattern = rf'Q{qnum}-P(\d+):\s*(.+)'
            matches = re.findall(pattern, resp)
            for _, para_text in matches:
                para_text = para_text.strip().strip('"').strip("'")
                if len(para_text) > 10 and para_text.lower() != plan["prompt"].lower():
                    examples.append(make_example(
                        para_text, plan["response"],
                        "planning", "paraphrase_v3",
                    ))

        time.sleep(0.5)

    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOOL KNOWLEDGE — scenario-based routing questions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scenario_routing():
    """Generate 'which tool for this scenario' questions across all agents."""
    examples = []

    scenarios_by_agent = {
        "IoTAgent": [
            "check sensor readings", "monitor temperature", "view asset list",
            "look at historical data", "find equipment at a site", "get telemetry",
            "review sensor names", "check what sites exist", "pull data from last week",
            "see what instruments are on the chiller",
        ],
        "FMSRAgent": [
            "identify failure modes", "check what can go wrong", "analyze sensor relevance",
            "map sensors to failures", "understand failure patterns",
            "determine which sensors detect overheating", "review FMEA data",
            "check failure-sensor correlation", "assess risk factors",
            "find which sensors are relevant for bearing damage",
        ],
        "TSFMAgent": [
            "forecast sensor values", "predict future readings", "detect anomalies",
            "run time-series analysis", "check for unusual patterns",
            "fine-tune a forecasting model", "list AI capabilities",
            "run end-to-end anomaly detection", "check available models",
            "predict next week's temperature trend",
        ],
        "WorkOrderAgent": [
            "review maintenance history", "check work orders", "see upcoming maintenance",
            "find corrective repairs", "analyze maintenance patterns",
            "predict next service date", "check failure codes",
            "review preventive maintenance schedule", "analyze alert patterns",
            "check equipment maintenance records for last year",
        ],
        "VibrationAgent": [
            "check vibration levels", "run FFT analysis", "diagnose bearing faults",
            "assess vibration severity", "compute frequency spectrum",
            "check envelope spectrum", "list vibration sensors",
            "calculate bearing frequencies", "run vibration diagnosis",
            "evaluate ISO vibration standards compliance",
        ],
        "Utilities": [
            "read a JSON file", "check current time", "get today's date",
            "parse configuration data", "load a data file",
        ],
    }

    agent_tools = {
        "IoTAgent": ["sites", "assets", "sensors", "history"],
        "FMSRAgent": ["get_failure_modes", "get_failure_mode_sensor_mapping"],
        "TSFMAgent": ["get_ai_tasks", "get_tsfm_models", "run_tsfm_forecasting", "run_tsfm_finetuning", "run_integrated_tsad"],
        "WorkOrderAgent": ["get_work_orders", "get_preventive_work_orders", "get_corrective_work_orders", "get_events", "get_failure_codes", "predict_next_work_order", "analyze_alert_to_failure"],
        "VibrationAgent": ["get_vibration_data", "list_vibration_sensors", "compute_fft_spectrum", "compute_envelope_spectrum", "assess_vibration_severity", "calculate_bearing_frequencies", "diagnose_vibration"],
        "Utilities": ["json_reader", "current_date_time", "current_time_english"],
    }

    assets = ["Chiller 6", "AHU-3", "Compressor 1", "Pump 2", "Cooling Tower 1", "Motor 1"]
    sites = ["MAIN", "SITE_A", "Building_7"]

    print("  Generating scenario-based routing questions...")
    for agent, scenarios in scenarios_by_agent.items():
        tools = agent_tools[agent]
        for scenario in scenarios:
            asset = random.choice(assets)
            site = random.choice(sites)
            tool = random.choice(tools)

            # Vary the question format
            templates = [
                f"I need to {scenario} for {asset} at {site}. Which agent and tool should I use?",
                f"Help me {scenario} on {asset}.",
                f"How do I {scenario} for equipment at the {site} facility?",
                f"What's the right tool to {scenario}?",
                f"I want to {scenario} for {asset}. What tool handles this?",
            ]
            question = random.choice(templates)
            answer = f"Use {agent}. The appropriate tool is {agent}.{tool} which handles {scenario}."

            examples.append(make_example(question, answer, "tool_routing_varied", "synthetic_routing"))

    print(f"  Generated {len(examples)} routing examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-AGENT SYNTHETIC SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_multi_agent_plans():
    """Generate plans that require multiple agents working together."""
    examples = []

    multi_agent_questions = [
        "Check the sensor readings for Chiller 6 at MAIN and identify its failure modes.",
        "List all assets at MAIN, then get the failure modes for each chiller found.",
        "Get the current date, then pull sensor history for Chiller 6 from the past week and run anomaly detection.",
        "Find all sensors on Chiller 6 at MAIN, get the failure modes, then map which sensors can detect each failure.",
        "Review the work orders for Chiller 6, check its failure modes, and determine which sensors are relevant for the most common failures.",
        "Run vibration diagnosis on Pump 2, and also check its recent work orders for any related maintenance.",
        "List all sites, then for each site list assets and their sensor counts.",
        "Get the failure modes for AHU-3, check which sensors can detect them, then run anomaly detection on those sensors.",
        "Check vibration severity on Motor 1, compute FFT spectrum, and also review its maintenance history.",
        "Pull temperature sensor history for Chiller 6, run forecasting on it, then detect anomalies in the forecast.",
        "Find preventive work orders for Compressor 1, check its failure modes, and assess vibration severity.",
        "Get current date/time, list assets at MAIN, check sensors on Chiller 6, and get its failure modes.",
        "Run integrated anomaly detection on Chiller 6 cooling water flow, and check work order history for any related corrective maintenance.",
        "List vibration sensors on Pump 2, get their data, and compute FFT spectra for each.",
        "Check what AI tasks are available, list forecasting models, then run a forecast on Chiller 6 condenser data.",
        "Analyze the alert-to-failure correlation for Chiller 6 and compare with its known failure modes.",
        "Get all sensors for AHU-3 at MAIN, identify failure modes for AHUs, then create a sensor-failure mapping.",
        "Check bearing frequencies for a 6205 bearing at 1800 RPM, then run envelope analysis on sensor V-101.",
        "Review corrective work orders for Compressor 1, list its failure modes, and predict when the next work order will be needed.",
        "Get historical data for all Chiller 6 sensors from June 2020, run forecasting, and detect anomalies.",
        "List known bearings, calculate fault frequencies for bearing 6308 at 3600 RPM, and run vibration diagnosis on Motor 2.",
        "Check failure codes for Pump 2, get its work order distribution, and analyze alert-to-failure patterns.",
        "Get today's date, find assets at MAIN, check sensors on each, and get failure modes for all chillers.",
        "Run vibration diagnosis on Compressor 1, check severity levels, and pull recent corrective work orders.",
        "Forecast Chiller 6 condenser water temperature, detect anomalies, and check relevant failure modes.",
    ]

    # Tool catalog for the prompt
    tool_catalog = """IoTAgent: IoT telemetry data
  - sites(): List monitoring sites
  - assets(site_name): List assets at a site
  - sensors(site_name, asset_id): List sensors for an asset
  - history(site_name, asset_id, start, final?): Get sensor readings

FMSRAgent: Failure mode analysis
  - get_failure_modes(asset_name): Get failure modes for an asset type
  - get_failure_mode_sensor_mapping(asset_name, failure_modes, sensors): Map sensor relevance to failures

TSFMAgent: Time-series forecasting & anomaly detection
  - get_ai_tasks(): List AI task types
  - get_tsfm_models(): List model checkpoints
  - run_tsfm_forecasting(dataset_path, timestamp_column, target_columns): Forecast
  - run_tsfm_finetuning(dataset_path, timestamp_column, target_columns): Fine-tune model
  - run_integrated_tsad(dataset_path, timestamp_column, target_columns): Forecast + anomaly detection

Utilities: General utilities
  - json_reader(file_name): Read JSON file
  - current_date_time(): Get current UTC date/time

WorkOrderAgent: Maintenance records
  - get_work_orders(asset_name, start_date?, end_date?): Get work orders
  - get_preventive_work_orders(asset_name): Get preventive WOs
  - get_corrective_work_orders(asset_name): Get corrective WOs
  - get_events(asset_name): Get maintenance events/alerts
  - get_failure_codes(asset_name): Get failure codes
  - get_work_order_distribution(asset_name): Get WO statistics
  - predict_next_work_order(asset_name): Predict next WO date
  - analyze_alert_to_failure(asset_name): Analyze alert-failure correlation

VibrationAgent: Vibration analysis
  - get_vibration_data(sensor_id): Get raw vibration data
  - list_vibration_sensors(asset_name): List vibration sensors
  - compute_fft_spectrum(sensor_id): Compute FFT spectrum
  - compute_envelope_spectrum(sensor_id): Envelope analysis
  - assess_vibration_severity(sensor_id): Assess ISO severity
  - calculate_bearing_frequencies(bearing_model, shaft_rpm): Calculate fault frequencies
  - diagnose_vibration(asset_name): Full vibration diagnosis"""

    print(f"  Generating {len(multi_agent_questions)} multi-agent plans...")
    for i, question in enumerate(multi_agent_questions):
        prompt = f"""Generate a structured plan for this industrial asset operations question.

AVAILABLE TOOLS:
{tool_catalog}

OUTPUT FORMAT:
#Task1: <description>
#Agent1: <agent_name>
#Tool1: <tool_name>
#Args1: {{"arg": "value"}}
#Dependency1: None
#ExpectedOutput1: <what to expect>

Rules:
- Use ONLY listed agents/tools
- Keep plans concise
- Use {{step_N}} for dependencies between steps

QUESTION: {question}"""

        resp = call_gemini(prompt, temperature=0.1, max_tokens=2048)
        if resp and "#Task1:" in resp and "#Agent1:" in resp:
            examples.append(make_example(question, resp.strip(), "planning", "synthetic_multiagent"))
        time.sleep(0.5)

    print(f"  Generated {len(examples)} multi-agent plans")
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MORE HARD NEGATIVES — systematic coverage
# ═══════════════════════════════════════════════════════════════════════════════

def generate_more_hard_negatives():
    """Generate hard negatives for commonly confused tool pairs."""
    examples = []

    confusions = [
        ("What sensors are on the chiller?", "FMSRAgent.get_failure_modes", "IoTAgent.sensors",
         "Listing physical sensors is an IoTAgent task. FMSRAgent handles failure mode analysis, not sensor enumeration."),
        ("When will the pump need maintenance next?", "IoTAgent.history", "WorkOrderAgent.predict_next_work_order",
         "Maintenance prediction uses WorkOrderAgent. IoTAgent.history returns raw sensor data, not maintenance schedules."),
        ("Is the motor vibrating too much?", "TSFMAgent.run_integrated_tsad", "VibrationAgent.assess_vibration_severity",
         "Vibration severity assessment uses VibrationAgent with ISO standards. TSFMAgent handles time-series anomaly detection on sensor data."),
        ("What's the frequency spectrum of the pump bearing?", "IoTAgent.history", "VibrationAgent.compute_fft_spectrum",
         "FFT spectrum computation is VibrationAgent's specialty. IoTAgent.history returns raw time-series data, not frequency analysis."),
        ("Show me the maintenance records for Chiller 6.", "IoTAgent.history", "WorkOrderAgent.get_work_orders",
         "Maintenance records are work orders managed by WorkOrderAgent. IoTAgent.history provides sensor telemetry, not maintenance logs."),
        ("Which failures can the temperature sensor detect?", "IoTAgent.sensors", "FMSRAgent.get_failure_mode_sensor_mapping",
         "Mapping sensors to detectable failures is FMSRAgent's role. IoTAgent.sensors just lists sensor names."),
        ("Are there any anomalies in the vibration data?", "VibrationAgent.diagnose_vibration", "TSFMAgent.run_integrated_tsad",
         "Statistical anomaly detection on time-series data uses TSFMAgent. VibrationAgent.diagnose_vibration does domain-specific vibration analysis (FFT, bearing faults)."),
        ("Read the sensor configuration file.", "IoTAgent.sensors", "Utilities.json_reader",
         "Reading files from disk is Utilities.json_reader. IoTAgent.sensors queries live sensor metadata from CouchDB."),
        ("What time is it?", "IoTAgent.sites", "Utilities.current_date_time",
         "Date/time queries use Utilities. IoTAgent handles sensor data, not system time."),
        ("Forecast the chiller temperature for next week.", "IoTAgent.history", "TSFMAgent.run_tsfm_forecasting",
         "Forecasting future values uses TSFMAgent models. IoTAgent.history only retrieves past recorded data."),
        ("List all corrective repairs for AHU-3.", "FMSRAgent.get_failure_modes", "WorkOrderAgent.get_corrective_work_orders",
         "Corrective repair records are work orders from WorkOrderAgent. FMSRAgent provides theoretical failure modes, not repair history."),
        ("Check if bearing 6308 has characteristic fault frequencies.", "TSFMAgent.get_ai_tasks", "VibrationAgent.calculate_bearing_frequencies",
         "Bearing fault frequency calculation is VibrationAgent's domain. TSFMAgent handles general AI task types."),
        ("What maintenance events happened last month?", "FMSRAgent.get_failure_modes", "WorkOrderAgent.get_events",
         "Maintenance events/alerts are tracked by WorkOrderAgent. FMSRAgent provides failure mode analysis, not event history."),
        ("How are work orders distributed across the year?", "IoTAgent.history", "WorkOrderAgent.get_work_order_distribution",
         "Work order statistics come from WorkOrderAgent. IoTAgent.history provides sensor time-series, not maintenance statistics."),
        ("Fine-tune the forecasting model on our chiller data.", "FMSRAgent.get_failure_mode_sensor_mapping", "TSFMAgent.run_tsfm_finetuning",
         "Model fine-tuning is a TSFMAgent capability. FMSRAgent handles failure-sensor mapping, not model training."),
        ("What envelope spectrum pattern does sensor V-101 show?", "TSFMAgent.run_tsad", "VibrationAgent.compute_envelope_spectrum",
         "Envelope spectrum is a vibration analysis technique from VibrationAgent. TSFMAgent.run_tsad does statistical anomaly detection."),
        ("Do alerts for Compressor 1 correlate with actual failures?", "FMSRAgent.get_failure_modes", "WorkOrderAgent.analyze_alert_to_failure",
         "Alert-to-failure correlation analysis is WorkOrderAgent territory. FMSRAgent lists failure modes theoretically."),
        ("What models are available for time-series prediction?", "Utilities.json_reader", "TSFMAgent.get_tsfm_models",
         "Available prediction models are listed by TSFMAgent. Utilities.json_reader reads arbitrary files, not model registries."),
    ]

    print(f"  Generating {len(confusions)} hard negatives...")
    for query, wrong, correct, explanation in confusions:
        instruction = f'The query "{query}" was routed to {wrong}. What is the correct routing?'
        response = f"Incorrect: {wrong}. Correct: {correct}. {explanation}"
        examples.append(make_example(instruction, response, "hard_negative", "synthetic_hn_v2"))

    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DATASET SCALE-UP")
    print("=" * 60)

    existing_hashes = load_existing_hashes()
    print(f"Existing unique examples: {len(existing_hashes)}")

    all_new = []

    # 1. Batch paraphrases for gold plans (biggest volume)
    print("\n[1/5] Batch paraphrases for gold plans...")
    gold_path = Path(__file__).parent.parent / "baseline_tests" / "gemini_flash_informed_results.json"
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold_plans = [g for g in gold_raw if g.get("plan_steps", 0) > 0]
    paras = generate_batch_paraphrases(gold_plans, num_per_plan=5, batch_size=5)
    paras = deduplicate_against(paras, existing_hashes)
    all_new.extend(paras)
    print(f"  New unique paraphrases: {len(paras)}")

    # 2. Batch paraphrases for WO plans
    print("\n[2/5] Paraphrases for WorkOrder plans...")
    with open(DATASETS_DIR / "planning.jsonl") as f:
        planning = [json.loads(line) for line in f]
    wo_plans = [p for p in planning if p.get("metadata", {}).get("source") == "synthetic_workorder"]
    # Convert to gold-like format
    wo_for_para = [{"prompt": p["messages"][0]["content"], "response": p["messages"][1]["content"]} for p in wo_plans]
    wo_paras = generate_batch_paraphrases(wo_for_para, num_per_plan=5, batch_size=5)
    wo_paras = deduplicate_against(wo_paras, existing_hashes)
    all_new.extend(wo_paras)
    print(f"  New unique WO paraphrases: {len(wo_paras)}")

    # 3. Scenario-based routing questions
    print("\n[3/5] Scenario-based routing questions...")
    routing = generate_scenario_routing()
    routing = deduplicate_against(routing, existing_hashes)
    all_new.extend(routing)
    print(f"  New unique routing: {len(routing)}")

    # 4. Multi-agent plans
    print("\n[4/5] Multi-agent synthetic plans...")
    multi = generate_multi_agent_plans()
    multi = deduplicate_against(multi, existing_hashes)
    all_new.extend(multi)
    print(f"  New unique multi-agent plans: {len(multi)}")

    # 5. More hard negatives
    print("\n[5/5] More hard negatives...")
    hn = generate_more_hard_negatives()
    hn = deduplicate_against(hn, existing_hashes)
    all_new.extend(hn)
    print(f"  New unique hard negatives: {len(hn)}")

    # ─── Merge into existing datasets ─────────────────────────────────
    print(f"\n[MERGE] Total new: {len(all_new)}")

    # Load existing
    with open(DATASETS_DIR / "tool_knowledge.jsonl") as f:
        tk = [json.loads(line) for line in f]
    with open(DATASETS_DIR / "planning.jsonl") as f:
        pl = [json.loads(line) for line in f]
    with open(DATASETS_DIR / "execution.jsonl") as f:
        ex = [json.loads(line) for line in f]

    # Sort new into datasets
    new_tk = [e for e in all_new if e["metadata"]["category"] in ("tool_routing_varied", "hard_negative")]
    new_pl = [e for e in all_new if e["metadata"]["category"] == "planning"]

    tk.extend(new_tk)
    pl.extend(new_pl)

    # Save
    for data, name in [(tk, "tool_knowledge"), (pl, "planning"), (ex, "execution")]:
        path = DATASETS_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} examples")

    # Combined
    combined = tk + pl + ex
    # Final dedup of combined
    seen = set()
    combined_unique = []
    for c in combined:
        h = hashlib.md5(c["messages"][0]["content"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            combined_unique.append(c)

    with open(DATASETS_DIR / "combined_all.jsonl", "w") as f:
        for c in combined_unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  SCALE-UP COMPLETE")
    print(f"{'='*60}")
    print(f"  New examples added: {len(all_new)}")
    print(f"  Tool knowledge:     {len(tk)}")
    print(f"  Planning:           {len(pl)}")
    print(f"  Execution:          {len(ex)}")
    print(f"  Combined total:     {len(combined_unique)}")

    new_sources = Counter(e["metadata"]["source"] for e in all_new)
    print(f"\n  New data breakdown:")
    for s, c in new_sources.most_common():
        print(f"    {s}: {c}")


if __name__ == "__main__":
    main()
