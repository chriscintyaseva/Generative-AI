
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os

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

# Lokasi penyimpanan index
PERSIST_DIR = "./llama_index_store"

# Buat folder penyimpanan jika belum ada
os.makedirs(PERSIST_DIR, exist_ok=True)

# Dokumen yang akan dimasukkan ke index
documents = [
    Document(
        text="LlamaIndex persists indexes to disk as JSON files."
    ),
    Document(
        text="Loading a persisted index avoids re-embedding on every run."
    ),
]

# Build dan simpan index
index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist(
    persist_dir=PERSIST_DIR
)

print("=" * 60)
print("       LLAMAINDEX PERSISTENCE")
print("=" * 60)

print(f"\nIndex berhasil disimpan ke: {PERSIST_DIR}")

# Load index kembali dari penyimpanan
storage_context = StorageContext.from_defaults(
    persist_dir=PERSIST_DIR
)

loaded_index = load_index_from_storage(storage_context)

# Membuat query engine
query_engine = loaded_index.as_query_engine()

# Mengajukan pertanyaan
question = "How does LlamaIndex handle persistence?"
response = query_engine.query(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response)

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
