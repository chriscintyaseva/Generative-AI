
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


# Create documents from text
documents = [
    Document(
        text="""LlamaIndex is a data framework for LLM applications.
It provides tools for ingesting, structuring, and querying data
using large language models. The core abstraction is the Index,
which organises documents into a structure optimised for retrieval.""",
        metadata={
            "source": "llamaindex_overview",
            "category": "framework"
        },
    ),

    Document(
        text="""Vector store indexes store document embeddings in a vector database.
They support fast semantic search using approximate nearest-neighbour algorithms.
LlamaIndex integrates with ChromaDB, Pinecone, Weaviate,
and many other backends.""",
        metadata={
            "source": "vector_store_guide",
            "category": "retrieval"
        },
    ),

    Document(
        text="""The query engine is the highest-level abstraction in LlamaIndex.
It takes a natural language question, retrieves relevant nodes,
and uses an LLM to synthesise a coherent answer from the retrieved context.""",
        metadata={
            "source": "query_engine_docs",
            "category": "querying"
        },
    ),
]


# Parse documents into nodes (chunks)
parser = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=20
)

nodes = parser.get_nodes_from_documents(documents)


# Display results
print("=" * 60)
print("       LLAMAINDEX DOCUMENT & NODE PARSING")
print("=" * 60)

print(f"\nDocuments: {len(documents)}")
print(f"Nodes: {len(nodes)}")

for i, n in enumerate(nodes, start=1):
    print(f"\nNode {i}")
    print("-" * 40)
    print(f"Node ID : {n.node_id[:8]}...")
    print(f"Length  : {len(n.text)} chars")
    print(f"Source  : {n.metadata.get('source')}")
    print(f"Category: {n.metadata.get('category')}")
    print(f"Content : {n.text}")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
