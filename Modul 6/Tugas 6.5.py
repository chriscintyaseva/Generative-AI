import requests
import base64
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"

def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    ext = Path(path).suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    else:
        raise ValueError("Format gambar harus JPG, JPEG, PNG, atau WEBP.")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "What is in this image?",
                    "images": [b64]
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()
    return result["message"]["content"]


image_path = "timi.jpg"

print("Model:", MODEL)
print("Image:", image_path)
print("\nResponse:")
print(describe_image_file(image_path))