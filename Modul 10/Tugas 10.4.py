
from llama_index.core import VectorStoreIndex, Document, Settings
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
        text="RAG combines retrieval with generation to answer questions from external knowledge."
    ),
    Document(
        text="Fine-tuning trains a model on new data to improve performance on a specific task."
    ),
    Document(
        text="Prompt engineering optimises the input to an LLM to get better outputs without any training."
    ),
    Document(
        text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."
    ),
    Document(
        text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."
    ),
]


# 4. Build index - embeds all documents automatically
index = VectorStoreIndex.from_documents(documents)


# 5. Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=3
)


# 6. Ask question
print("=" * 60)
print("       LLAMAINDEX VECTOR STORE & QUERY ENGINE")
print("=" * 60)

question = "What is the difference between RAG and fine-tuning?"

print("\nQuestion:", question)

response = query_engine.query(question)

print("\nAnswer:")
print(response)


# 7. Display source nodes
print("\n" + "=" * 60)
print("SOURCE NODES")
print("=" * 60)

for i, node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}")
    print(f"Score: {node.score:.4f}")
    print(f"Text : {node.text[:150]}...")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
