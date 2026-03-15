import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

load_dotenv()

MODEL = "meta-llama/llama-3.1-8b-instruct"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
scenarios = dataset["train"]

results = []

for example in scenarios:
    prompt = example["text"]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\n================")
    print("ID:", example["id"])
    print("PROMPT:", prompt)
    print("MODEL:", answer)

    results.append({
        "id": example["id"],
        "prompt": prompt,
        "response": answer
    })

with open("llama4_maverick_results.json", "w") as f:
    json.dump(results, f, indent=2)
