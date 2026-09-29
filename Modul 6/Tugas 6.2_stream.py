import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": "Sebutkan 5 kegunaan vector database.",
        "stream": True
    },
    stream=True
)

response.raise_for_status()

for line in response.iter_lines():
    if line:
        data = json.loads(line)
        print(data["response"], end="", flush=True)

print()
