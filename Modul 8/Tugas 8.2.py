import os
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

OPENAI_CODE = """
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

embeddings = embed(texts)
print(f"Shape: {embeddings.shape}")
print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")
"""

print("=== OpenAI Embeddings ===")
print("Tidak dijalankan karena OPENAI_API_KEY belum tersedia.")

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

result = vo.embed(
    ["What is RAG?", "Explain vector databases."],
    model="voyage-3",
    input_type="document"
)

embeddings = np.array(result.embeddings, dtype=np.float32)

print("\n=== Voyage AI Embeddings ===")
print(f"Shape: {embeddings.shape}")
print(f"Token usage: {result.total_tokens}")