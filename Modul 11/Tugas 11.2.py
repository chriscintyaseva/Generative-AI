
import chromadb
from langchain_ollama import OllamaEmbeddings

# ==========================================
# KONFIGURASI OLLAMA EMBEDDING
# ==========================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Fungsi untuk menghasilkan embedding
def embed(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)


# ==========================================
# PERSISTENT CHROMA CLIENT
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Membuat atau mengambil collection
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# MENAMBAHKAN DOKUMEN
# ==========================================

documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"category": "technique", "index": i}
    for i in range(len(documents))
]

embeddings = embed(documents)

# Upsert menambahkan data baru atau memperbarui ID yang sudah ada
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("=" * 60)
print("               CHROMADB")
print("=" * 60)

print(f"\nCollection size: {collection.count()}")


# ==========================================
# QUERY / PENCARIAN DOKUMEN
# ==========================================

query = "how do I add knowledge to an LLM without retraining?"

q_embed = embed([query])[0]

results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

print("\nTOP RESULTS")
print("=" * 60)

for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"\nDistance : {dist:.4f}")
    print(f"Document : {doc}")
    print(f"Category : {meta['category']}")


# ==========================================
# METADATA FILTERING
# ==========================================

results_filtered = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}},
    include=["documents", "distances"]
)

print("\n" + "=" * 60)
print("             METADATA FILTERING")
print("=" * 60)

for doc, dist in zip(
    results_filtered["documents"][0],
    results_filtered["distances"][0]
):
    print(f"\nDistance : {dist:.4f}")
    print(f"Document : {doc}")


# ==========================================
# UPDATE DOCUMENT
# ==========================================

updated_text = (
    "RAG retrieves external documents and injects "
    "them into the LLM context."
)

collection.update(
    ids=["doc_0"],
    documents=[updated_text],
    embeddings=embed([updated_text])
)

print("\nDocument doc_0 berhasil diperbarui.")


# ==========================================
# DELETE DOCUMENT
# ==========================================

collection.delete(
    ids=["doc_4"]
)

print(f"After delete, count: {collection.count()}")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
