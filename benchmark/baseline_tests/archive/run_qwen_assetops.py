import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "qwen/qwen-2.5-7b-instruct"


def ask_model(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


prompts = [
    "What is predictive maintenance?",
    "Explain asset lifecycle management.",
    "How can IoT sensors improve maintenance planning?"
]

for p in prompts:
    print("\nPROMPT:", p)
    print("MODEL:", ask_model(p))
