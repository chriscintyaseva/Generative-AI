
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv
import os

# =====================================
# KONFIGURASI GEMINI
# =====================================

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_output_tokens=1024,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

parser = StrOutputParser()


# =====================================
# 1. BASIC CHAIN: PROMPT | MODEL | PARSER
# =====================================

print("\n" + "=" * 50)
print("              1. BASIC CHAIN")
print("=" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "sentences": 2,
    "text": """
    LangChain is a framework for developing applications
    powered by large language models.
    It provides abstractions for prompts, models, memory,
    and retrieval so developers can compose complex AI
    workflows from simple building blocks without writing
    boilerplate.
    """
})

print("\nSummary:")
print(result)


# =====================================
# 2. BRANCHING WITH RUNNABLEPARALLEL
# =====================================

print("\n" + "=" * 50)
print("       2. BRANCHING WITH RUNNABLEPARALLEL")
print("=" * 50)

summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise this in one sentence."),
    ("human", "{text}"),
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 3 keywords from this text. Return them comma-separated."),
    ("human", "{text}"),
])

parallel_chain = RunnableParallel(
    summary=summarise_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)

result = parallel_chain.invoke({
    "text": """
    Retrieval-Augmented Generation (RAG) improves LLM
    answers by fetching relevant documents at inference time.
    """
})

print("\nSummary:")
print(result["summary"])

print("\nKeywords:")
print(result["keywords"])


# =====================================
# 3. CONDITIONAL ROUTING
# =====================================

print("\n" + "=" * 50)
print("          3. CONDITIONAL ROUTING")
print("=" * 50)

code_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Python expert. Provide a complete, working code example."
    ),
    ("human", "{input}"),
])

prose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clear technical writer. Explain conceptually."
    ),
    ("human", "{input}"),
])

code_chain = code_prompt | llm | parser
prose_chain = prose_prompt | llm | parser


def route(input_dict: dict) -> str:
    """Route to code or prose chain based on the question."""

    q = input_dict["input"].lower()

    if any(word in q for word in [
        "write", "code", "implement", "example", "function"
    ]):
        return "code"

    return "prose"


def run_routed(input_dict: dict) -> str:
    choice = route(input_dict)

    print(f"\nSelected route: {choice.upper()}")

    if choice == "code":
        return code_chain.invoke(input_dict)

    return prose_chain.invoke(input_dict)


router_chain = RunnableLambda(run_routed)

print("\nQuestion 1:")
result1 = router_chain.invoke({
    "input": "Write a function to chunk text with overlap."
})
print(result1)

print("\nQuestion 2:")
result2 = router_chain.invoke({
    "input": "What is attention in transformers?"
})
print(result2)


# =====================================
# PROGRAM SELESAI
# =====================================

print("\n" + "=" * 50)
print("             PROGRAM SELESAI")
print("=" * 50)
