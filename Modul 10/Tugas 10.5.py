
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# 1. Configure Ollama LLM
Settings.llm = Ollama(
    model="qwen2.5:3b",
    request_timeout=120.0,
    temperature=0.1,
    context_window=4096
)


# 2. Configure embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout=120.0
)


# 3. Create documents
documents = [
    Document(
        text="ChromaDB is an open-source embedding database with a simple Python API.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="Pinecone is a managed vector database designed for production workloads.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="FAISS is a library by Meta for efficient similarity search on dense vectors.",
        metadata={"tag": "vector-search"}
    ),
    Document(
        text="pgvector adds vector similarity search to PostgreSQL.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="Weaviate is an open-source vector database with a GraphQL API.",
        metadata={"tag": "vector-db"}
    ),
]


# 4. Build vector index
index = VectorStoreIndex.from_documents(documents)


# 5. Create retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3
)


# 6. Retrieve relevant documents
query = "open source vector databases"

nodes = retriever.retrieve(query)


# 7. Display retrieved nodes
print("=" * 60)
print("          VECTOR INDEX RETRIEVER")
print("=" * 60)

print(f"\nQuery: {query}")

print("\nRetrieved Nodes:")

for i, node in enumerate(nodes, start=1):
    print(f"\nNode {i}")
    print("-" * 40)
    print(f"Score: {node.score:.4f}")
    print(f"Text : {node.text}")
    print(f"Tag  : {node.metadata.get('tag')}")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
