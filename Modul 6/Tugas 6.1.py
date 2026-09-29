import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

prompt = "Jelaskan apa itu Generative AI dengan singkat."

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

print("Model:", MODEL)
print("Response:", result["response"])