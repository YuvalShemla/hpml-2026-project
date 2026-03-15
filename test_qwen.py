from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

response = client.chat.completions.create(
    model="qwen/qwen-2.5-7b-instruct",
    messages=[
        {"role": "user", "content": "Explain what AssetOpsBench evaluates in one sentence."}
    ],
    extra_headers={
        "HTTP-Referer": "http://localhost", 
        "X-Title": "AssetOpsBench-Test"
    }
)

print(response.choices[0].message.content)
