
from google import genai

client = genai.Client(api_key="AIzaSyBokdGlV3QgehdA0jTbeaCaxKgFqSkYhwc")

# Changed model to gemini-3-flash-preview (Current March 2026 standard)
response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents="Give me a 3-step plan for a Python app."
)

print(response.text)