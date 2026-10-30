
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Konfigurasi Ollama
Settings.llm = Ollama(
    model="qwen2.5:3b",
    request_timeout=120.0,
    temperature=0.1,
    context_window=4096
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout=120.0
)

# Dokumen sumber
documents = [
    Document(
        text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."
    ),
    Document(
        text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."
    ),
    Document(
        text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."
    ),
    Document(
        text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."
    ),
]

# Membuat vector index
index = VectorStoreIndex.from_documents(documents)

# Membuat retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2
)

# Pertanyaan
question = "What algorithms are used in vector search?"

print("=" * 60)
print("       RESPONSE SYNTHESIZER")
print("=" * 60)

# Menguji dua response modes
for mode in [
    ResponseMode.COMPACT,
    ResponseMode.TREE_SUMMARIZE
]:
    synthesizer = get_response_synthesizer(
        response_mode=mode
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )

    response = engine.query(question)

    print("\n" + "=" * 60)
    print(f"RESPONSE MODE: {mode}")
    print("=" * 60)
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(response)

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
