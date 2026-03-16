import os
import json
import time
import random
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# MODEL = "meta-llama/llama-3.1-70b-instruct"
MODEL = "meta-llama/llama-3.1-8b-instruct"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "assetops-data-generation"
    }
)

dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
scenarios = dataset["train"]

OUTPUT_FILE = "assetops_sft_dataset.jsonl"

PARAPHRASES_PER_SCENARIO = 20
TARGET_DATASET_SIZE = 20000
MAX_RETRIES = 3

ASSETS = [
    "Chiller 1","Chiller 2","Chiller 3",
    "Pump 1","Pump 2","Pump 3",
    "Cooling Tower 1","Cooling Tower 2",
    "Compressor 1","Compressor 2"
]

SENSORS = [
    "temperature","pressure","vibration",
    "flow_rate","power_usage","humidity"
]

SITES = ["MAIN","SITE_A","SITE_B","SITE_C"]

TIME_WINDOWS = [
    "last hour","last 24 hours","last week","last month"
]

existing_questions = set()

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                obj = json.loads(line)
                existing_questions.add(obj["messages"][0]["content"])
            except:
                pass

def call_llm(messages, temperature=0.3):

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature
            )

            return response

        except Exception:

            if attempt == MAX_RETRIES - 1:
                return None

            time.sleep(2 ** attempt)

def generate_plan(question):

    system_prompt = """
You are an expert AssetOpsBench planning agent.

Output format:

# Task1: description
# Agent1: agent_name
# Tool1: tool_name
# Args1: {json args}
# Dependency1: None or TaskX
"""

    response = call_llm([
        {"role":"system","content":system_prompt},
        {"role":"user","content":question}
    ], temperature=0.2)

    if response is None:
        return None,None

    content = response.choices[0].message.content

    usage = getattr(response,"usage",None)

    tokens = {
        "prompt": usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0
    }

    return content,tokens

def paraphrase_question(question):

    prompt = f"""
Rewrite this question in {PARAPHRASES_PER_SCENARIO} different ways.
Return each paraphrase on a new line.

Question:
{question}
"""

    response = call_llm([{"role":"user","content":prompt}],temperature=0.7)

    if response is None:
        return []

    text = response.choices[0].message.content

    lines=[]

    for line in text.split("\n"):

        line=line.strip("- ").strip()

        if len(line)>8:
            lines.append(line)

    return lines[:PARAPHRASES_PER_SCENARIO]

def procedural_variation(question):

    asset=random.choice(ASSETS)
    sensor=random.choice(SENSORS)
    site=random.choice(SITES)
    window=random.choice(TIME_WINDOWS)

    replacements={
        "Chiller 6":asset,
        "Chiller":asset,
        "sensor":sensor,
        "site MAIN":f"site {site}",
        "last 24 hours":window
    }

    for k,v in replacements.items():
        question=question.replace(k,v)

    return question

def validate_plan(plan):

    if plan is None:
        return False

    required=["Task","Agent","Tool","Args"]

    for field in required:
        if field not in plan:
            return False

    try:

        for line in plan.split("\n"):
            if "# Args" in line:
                json_part=line.split(":",1)[1].strip()
                json.loads(json_part)

    except:
        return False

    return True

def label_difficulty(plan):

    steps=plan.count("# Task")

    if steps==1:
        return "L1"
    if steps==2:
        return "L2"
    if steps<=4:
        return "L3"

    return "L4"

total_examples=0
prompt_tokens=0
completion_tokens=0

with open(OUTPUT_FILE,"a") as f:

    pbar=tqdm(total=TARGET_DATASET_SIZE,unit="examples")

    start_time=time.time()

    while total_examples<TARGET_DATASET_SIZE:

        example=random.choice(scenarios)

        base_question=example["text"]

        questions=paraphrase_question(base_question)

        for q in questions:

            q=procedural_variation(q)

            if q in existing_questions:
                continue

            plan,tokens=generate_plan(q)

            if not validate_plan(plan):
                continue

            difficulty=label_difficulty(plan)

            record={
                "messages":[
                    {"role":"user","content":q},
                    {"role":"assistant","content":plan}
                ],
                "difficulty":difficulty
            }

            f.write(json.dumps(record)+"\n")
            f.flush()

            existing_questions.add(q)

            total_examples+=1

            if tokens:
                prompt_tokens+=tokens["prompt"]
                completion_tokens+=tokens["completion"]

            elapsed=time.time()-start_time
            speed=total_examples/max(elapsed,1)

            pbar.set_postfix({
                "speed":f"{speed:.2f}/s",
                "tokens":prompt_tokens+completion_tokens
            })

            pbar.update(1)

            if total_examples>=TARGET_DATASET_SIZE:
                break

            time.sleep(random.uniform(0.2,0.6))

    pbar.close()

print("\nDataset saved:",OUTPUT_FILE)
print("Total examples:",total_examples)
print("Prompt tokens:",prompt_tokens)
print("Completion tokens:",completion_tokens)
print("Total tokens:",prompt_tokens+completion_tokens)