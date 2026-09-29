import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

WEAK_SYSTEM = "You are an AI assistant."

STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems.
- Suggest concrete improvements with code examples.
- Explain WHY each issue matters.

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list.
Each item: Issue → Impact → Fix.
"""

code_to_review = """Review this function:

def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()
"""

def ask_model(system_prompt, user_prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


print("=== WEAK SYSTEM PROMPT ===")
print(ask_model(WEAK_SYSTEM, code_to_review))

print("\n=== STRONG SYSTEM PROMPT ===")
print(ask_model(STRONG_SYSTEM, code_to_review))