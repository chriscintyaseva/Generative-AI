
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings
import os
import time

# ==========================================
# KONFIGURASI OLLAMA EMBEDDING
# ==========================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

def embed(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)


# ==========================================
# KONEKSI PINECONE
# ==========================================

# Masukkan API Key Pinecone di environment variable Windows
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY belum diatur.")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "ai-knowledge-base"

# Membuat index jika belum tersedia
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

# Menunggu index siap
while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(1)

index = pc.Index(INDEX_NAME)


# ==========================================
# UPSERT VECTORS
# ==========================================

documents = [
    {
        "id": "d1",
        "text": "RAG connects LLMs to external knowledge bases."
    },
    {
        "id": "d2",
        "text": "Fine-tuning updates model weights on task-specific data."
    },
    {
        "id": "d3",
        "text": "ChromaDB is an open-source local vector database."
    },
]

vectors = [
    {
        "id": doc["id"],
        "values": embed([doc["text"]])[0],
        "metadata": {
            "text": doc["text"],
            "source": "guide"
        },
    }
    for doc in documents
]

index.upsert(vectors=vectors)

print("=" * 60)
print("                 PINECONE")
print("=" * 60)

print("\nIndex Statistics:")
print(index.describe_index_stats())


# ==========================================
# QUERY
# ==========================================

query = "how to add external knowledge to an LLM?"

q_vector = embedding_model.embed_query(query)

results = index.query(
    vector=q_vector,
    top_k=3,
    include_metadata=True,
)

print("\n" + "=" * 60)
print("              SEARCH RESULTS")
print("=" * 60)

for match in results["matches"]:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Text : {match['metadata']['text']}")


# ==========================================
# METADATA FILTER
# ==========================================

filtered = index.query(
    vector=q_vector,
    top_k=3,
    filter={
        "source": {"$eq": "guide"}
    },
    include_metadata=True,
)

print("\n" + "=" * 60)
print("              METADATA FILTER")
print("=" * 60)

for match in filtered["matches"]:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Text : {match['metadata']['text']}")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
