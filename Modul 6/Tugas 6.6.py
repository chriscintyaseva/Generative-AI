import requests

MODEL = "qwen2.5:3b"

PRICING = {
    "qwen2.5:3b": {
        "input": 0.00,
        "output": 0.00
    },
    "qwen2.5vl:3b": {
        "input": 0.00,
        "output": 0.00
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00
    }
}

CONTEXT_LIMITS = {
    "qwen2.5:3b": 32768,
    "qwen2.5vl:3b": 32768,
    "gpt-4o": 128000,
    "claude-sonnet-4-5": 200000
}


def estimate_tokens(text: str) -> int:
    """Estimate token count approximately."""
    return max(1, len(text.split()))


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]

    return (
        input_tokens * p["input"] +
        output_tokens * p["output"]
    ) / 1_000_000


def fits_in_context(
    model: str,
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    limit = CONTEXT_LIMITS.get(model, 32768)

    return token_count + reserve_for_output <= limit


prompt = "Explain the transformer architecture in simple terms."

input_tokens = estimate_tokens(prompt)
output_tokens = 300

print("Model:", MODEL)
print("Prompt:", prompt)
print("Estimated input tokens:", input_tokens)

cost = estimate_cost(
    MODEL,
    input_tokens,
    output_tokens
)

print(f"Estimated cost: ${cost:.6f}")

context_limit = CONTEXT_LIMITS[MODEL]

print("Context limit:", context_limit, "tokens")

if fits_in_context(MODEL, input_tokens):
    print("Status: Prompt fits in context window.")
else:
    print("Status: Prompt exceeds context window.")


response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

print("\nResponse:")
print(result["message"]["content"])