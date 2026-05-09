import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Changed model to gemini-3-flash-preview (Current March 2026 standard)
response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents="Give me a 3-step plan for a Python app."
)

print(response.text)