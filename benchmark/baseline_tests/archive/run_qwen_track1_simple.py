import os
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "qwen/qwen-2.5-7b-instruct"

dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")

scenarios = dataset["train"]

for example in scenarios.select(range(10)):   # first 10 for testing
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
