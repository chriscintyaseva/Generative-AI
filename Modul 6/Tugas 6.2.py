import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "system": "You are a concise technical writer. Answer in simple English with no jargon.",
        "prompt": "Explain what a vector database does.",
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

print(result["response"])