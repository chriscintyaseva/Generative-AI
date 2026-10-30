
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Documents
docs = [
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."
    ),
    Document(
        page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."
    ),
    Document(
        page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."
    ),
    Document(
        page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."
    ),
    Document(
        page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."
    ),
]


# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_documents(docs)


# 3. Embeddings and Chroma vector store
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 4. RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer the question using ONLY the provided context.

If the context does not contain the answer,
say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])


# 5. Initialize Ollama LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()


# Format retrieved documents
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# Create RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | parser
)


# 6. Ask questions
questions = [
    "What does RAG stand for?",
    "How is RAG different from fine-tuning?",
    "What is the capital of France?"
]

print("=" * 60)
print("          RETRIEVAL-AUGMENTED GENERATION")
print("=" * 60)

for i, q in enumerate(questions, start=1):
    print(f"\nQUESTION {i}")
    print("-" * 60)
    print(f"Q: {q}")

    answer = rag_chain.invoke(q)

    print(f"\nA: {answer}")

print("\n" + "=" * 60)
print("             PROGRAM SELESAI")
print("=" * 60)
