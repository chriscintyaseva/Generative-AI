import requests
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]
    must_be_json: bool = False


def ask_ollama(system: str, user: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
    results = []

    for case in cases:
        text = ask_ollama(system, case.input_text)

        keyword_hit = any(
            keyword.lower() in text.lower()
            for keyword in case.expected_keywords
        )

        json_valid = True

        if case.must_be_json:
            import json

            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid

        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80]
        })

    pass_rate = sum(
        result["passed"] for result in results
    ) / len(results)

    return {
        "pass_rate": pass_rate,
        "results": results
    }


CLASSIFY_SYSTEM = """Classify the AI task as one of:
CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.

Return ONLY the category word.
Do not provide explanations."""


test_cases = [
    EvalCase(
        "Predict whether an email is spam.",
        ["CLASSIFICATION"]
    ),
    EvalCase(
        "Write a product description for headphones.",
        ["GENERATION"]
    ),
    EvalCase(
        "Find the most relevant documents for a query.",
        ["RETRIEVAL"]
    ),
    EvalCase(
        "Convert this sentence to a vector.",
        ["EMBEDDING"]
    ),
    EvalCase(
        "Label customer reviews as positive or negative.",
        ["CLASSIFICATION"]
    )
]


report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)

print(f"Pass rate: {report['pass_rate']:.0%}")

for result in report["results"]:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{status}] "
        f"{result['input']!r} -> "
        f"{result['response_preview']!r}"
    )