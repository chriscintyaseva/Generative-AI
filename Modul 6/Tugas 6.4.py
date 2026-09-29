import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

def get_model_info(model_name: str) -> dict:
    db = {
        "qwen2.5:3b": {
            "context_k": 32768,
            "cost_input": 0.00,
            "cost_output": 0.00
        },
        "gpt-4o": {
            "context_k": 128,
            "cost_input": 2.50,
            "cost_output": 10.00
        },
        "claude-sonnet-4-5": {
            "context_k": 200,
            "cost_input": 3.00,
            "cost_output": 15.00
        }
    }

    return db.get(
        model_name,
        {"error": f"Unknown model: {model_name}"}
    )

user_question = "How large is the context window of qwen2.5:3b?"

messages = [
    {
        "role": "system",
        "content": "If the user asks about model information, use the get_model_info tool."
    },
    {
        "role": "user",
        "content": user_question
    }
]

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
)

response.raise_for_status()

assistant_text = response.json()["message"]["content"]

print("User:", user_question)
print("Qwen:", assistant_text)

tool_result = get_model_info("qwen2.5:3b")

print("\nTool called: get_model_info('qwen2.5:3b')")
print("Tool result:", tool_result)

messages.append({
    "role": "assistant",
    "content": assistant_text
})

messages.append({
    "role": "user",
    "content": f"Tool result: {json.dumps(tool_result)}. Give the final answer based on this result."
})

final_response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
)

final_response.raise_for_status()

final_text = final_response.json()["message"]["content"]

print("\nFinal answer:")
print(final_text)