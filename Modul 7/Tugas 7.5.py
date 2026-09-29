import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

SYSTEM = """You are a data extractor.
Extract information from the text and return ONLY valid JSON.

Do not use markdown.
Do not use code fences.
Do not add explanations.

Use exactly this schema:
{
  "company": "string",
  "founded": 0,
  "products": ["string"],
  "headquarters": "string",
  "is_public": false
}

If information is unknown, use null.
"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment."
]

def extract_company_info(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}
            ],
            "stream": False,
            "format": "json"
        }
    )

    response.raise_for_status()

    raw = response.json()["message"]["content"].strip()

    return json.loads(raw)

for text in texts:
    try:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print()
    except json.JSONDecodeError:
        print("Error: Model menghasilkan JSON yang tidak valid.")