import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

FEW_SHOT_SYSTEM = """You are a data extractor.

Given a raw AI benchmark result string, extract:
- model name
- task
- score

Return the result as a JSON object.

Examples:

Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation.
"""

test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified"
]


for text in test_inputs:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": FEW_SHOT_SYSTEM
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()
    output = result["message"]["content"]

    print(f"Input: {text}")
    print(f"Output: {output}\n")