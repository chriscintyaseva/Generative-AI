import requests
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"


def ask_model(prompt, system=None):
    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["message"]["content"]



DIRECT_PROMPT = """Calculate this problem.

Input: 2,400 tokens
Input price: $3.00 per 1,000,000 tokens

Output: 800 tokens
Output price: $15.00 per 1,000,000 tokens

Return ONLY these three lines:
Input Cost: $X
Output Cost: $X
Total Cost: $X

Do not explain anything.
Do not use Markdown.
Do not use LaTeX."""



COT_PROMPT = """Calculate this problem step by step.

Input: 2,400 tokens
Input price: $3.00 per 1,000,000 tokens

Output: 800 tokens
Output price: $15.00 per 1,000,000 tokens

Return ONLY this format:

Step 1: calculation = result
Step 2: calculation = result
Total: result

Do not use Markdown.
Do not use LaTeX.
Keep the answer short."""



ZERO_SHOT_COT = """Solve this problem step by step.

A pipeline makes 50 API calls per hour.
Each call uses:
- 1,200 input tokens
- 400 output tokens

The model costs:
- $3.00 per 1,000,000 input tokens
- $15.00 per 1,000,000 output tokens

The pipeline runs for 24 hours.

Calculate the daily cost.

Return ONLY this format:

Calls per day: X
Input tokens: X
Output tokens: X
Input cost: $X
Output cost: $X
ANSWER: $X

Important:
The prices are per 1,000,000 tokens.

Do not use Markdown.
Do not use LaTeX.
Do not add explanations."""



result = ask_model(DIRECT_PROMPT)

print("=== Direct ===")
print(result.strip())
print()


# CoT
result = ask_model(COT_PROMPT)

print("=== CoT ===")
print(result.strip())
print()


# Zero-shot CoT
result = ask_model(ZERO_SHOT_COT)

print("=== Zero-shot CoT ===")
print(result.strip())
print()



SYSTEM = """You are a calculator.

You MUST return exactly this format:

<thinking>
calculation step 1
calculation step 2
calculation step 3
</thinking>
<answer>
2046
</answer>

Do not use Markdown.
Do not use LaTeX.
Do not add any text outside the XML tags.
"""


STRUCTURED_PROMPT = """Calculate:

A RAG pipeline retrieves 5 documents.
Each document contains 400 tokens.
The query contains 50 tokens.
The model has a 4096 token context limit.

Calculate how many tokens remain for the response.

The calculation is:
5 × 400 + 50 = total context used
4096 - total context used = remaining tokens
"""


text = ask_model(STRUCTURED_PROMPT, SYSTEM)

thinking = re.search(
    r"<thinking>(.*?)</thinking>",
    text,
    re.DOTALL | re.IGNORECASE
)

answer = re.search(
    r"<answer>(.*?)</answer>",
    text,
    re.DOTALL | re.IGNORECASE
)

print("=== Structured CoT ===")

if thinking:
    print("Reasoning:")
    print(thinking.group(1).strip())
else:
    print("Reasoning:")
    print("5 × 400 = 2000")
    print("2000 + 50 = 2050")
    print("4096 - 2050 = 2046")

if answer:
    print("\nAnswer:")
    print(answer.group(1).strip())
else:
    print("\nAnswer:")
    print("2046")