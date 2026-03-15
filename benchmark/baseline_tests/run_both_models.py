import os
import json
import time
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

MODELS = {
    "llama4_maverick": "meta-llama/llama-3.1-8b-instruct",
    "qwen_2_5_7b": "qwen/qwen-2.5-7b-instruct"
}

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
scenarios = dataset["train"]

os.makedirs(".", exist_ok=True)

for name, model in MODELS.items():

    print("\n========================")
    print("Running:", name)
    print("========================")

    results = []

    for example in tqdm(scenarios):

        prompt = example["text"]

        try:

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content

            usage = getattr(response, "usage", None)

            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            results.append({
                "id": example["id"],
                "prompt": prompt,
                "response": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            })

        except Exception as e:

            print("Error:", e)

            results.append({
                "id": example["id"],
                "prompt": prompt,
                "response": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": str(e)
            })

        time.sleep(0.3)

    filename = f"{name}_results.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", filename)

print("\nAll models finished.")