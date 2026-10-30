
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# Configure global LLM settings
Settings.llm = Ollama(
    model="qwen2.5:3b",
    request_timeout=120.0,
    temperature=0.1,
    context_window=4096,
)

# Configure embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout=120.0,
)

# Configure text splitting settings
Settings.chunk_size = 512
Settings.chunk_overlap = 64


# Display configuration
print("=" * 50)
print("       LLAMAINDEX GLOBAL SETTINGS")
print("=" * 50)

print("\nLLM Model: qwen2.5:3b")
print("Embedding Model: nomic-embed-text")
print(f"Chunk Size: {Settings.chunk_size}")
print(f"Chunk Overlap: {Settings.chunk_overlap}")

print("\nConfiguration completed successfully!")
print("=" * 50)
