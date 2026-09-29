import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Explain embeddings in 3 bullet points."
            }
        ],
        "stream": True
    },
    stream=True
)

response.raise_for_status()

for line in response.iter_lines():
    if line:
        data = line.decode("utf-8")
        import json
        result = json.loads(data)
        delta = result.get("message", {}).get("content", "")
        if delta:
            print(delta, end="", flush=True)

print()