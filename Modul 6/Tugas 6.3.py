import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise technical assistant."
            },
            {
                "role": "user",
                "content": "What is the difference between RAG and fine-tuning?"
            }
        ],
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

print("Model:", MODEL)
print("Response:")
print(result["message"]["content"])