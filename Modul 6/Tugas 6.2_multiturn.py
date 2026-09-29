import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

history = [
    {
        "role": "system",
        "content": "You are a helpful Python tutor."
    }
]

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        break

    history.append({
        "role": "user",
        "content": user_input
    })

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": history,
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()
    assistant_text = result["message"]["content"]

    history.append({
        "role": "assistant",
        "content": assistant_text
    })

    print(f"Qwen: {assistant_text}\n")