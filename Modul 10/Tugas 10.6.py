
from llama_index.core import SummaryIndex, Document, Settings
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


# 3. Create document sections
sections = [
    Document(
        text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."
    ),

    Document(
        text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."
    ),

    Document(
        text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."
    ),

    Document(
        text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."
    ),

    Document(
        text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."
    ),
]


# 4. Build SummaryIndex
index = SummaryIndex.from_documents(sections)


# 5. Create query engine
query_engine = index.as_query_engine()


# 6. Ask question
question = "Which model should I use for a coding project?"

response = query_engine.query(question)


# 7. Display result
print("=" * 60)
print("                SUMMARY INDEX")
print("=" * 60)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response)

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
