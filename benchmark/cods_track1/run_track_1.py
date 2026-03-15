import argparse
import json
import os

from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

# Load environment variables (.env or exported variables)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "qwen/qwen-2.5-7b-instruct"

RESULT_DIR = "./track1_results/"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_scenarios(utterance_ids=None):
    """
    Load AssetOpsBench scenarios from HuggingFace
    """
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    scenarios = ds["train"]

    if utterance_ids:
        scenarios = scenarios.filter(lambda x: x["id"] in utterance_ids)

    return scenarios


def query_model(prompt):
    """
    Send prompt to Qwen via OpenRouter
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content


def run(scenarios):
    results = []

    for example in scenarios:
        prompt = example["text"]
        sid = example["id"]

        print("=" * 40)
        print("ID:", sid)
        print("PROMPT:", prompt)

        answer = query_model(prompt)

        print("MODEL:", answer)

        results.append(
            {
                "id": sid,
                "prompt": prompt,
                "response": answer,
            }
        )

    output_file = os.path.join(RESULT_DIR, "qwen_track1_results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utterance_ids",
        type=str,
        default=None,
        help="Comma separated scenario IDs (e.g. 1,2,3)",
    )

    args = parser.parse_args()

    utterance_ids = None
    if args.utterance_ids:
        utterance_ids = [int(x.strip()) for x in args.utterance_ids.split(",")]

    scenarios = load_scenarios(utterance_ids)

    run(scenarios)
