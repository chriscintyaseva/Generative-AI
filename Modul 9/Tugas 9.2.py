
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Memuat API Key dari file .env
load_dotenv()

# Konfigurasi Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_output_tokens=1024,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Instruksi sistem
system_message = SystemMessage(
    content=(
        "You are a concise technical assistant. "
        "Answer in plain text without Markdown formatting. "
        "Use simple language and short paragraphs. "
        "Avoid long explanations."
    )
)

# Pesan pengguna
messages = [
    system_message,
    HumanMessage(
        content="What is LangChain Expression Language? "
                "Explain briefly in Indonesian."
    )
]

# =====================================
# INVOKE
# =====================================
print("=" * 50)
print("                 INVOKE")
print("=" * 50)

response = llm.invoke(messages)
print(response.text)


# =====================================
# STREAM
# =====================================
print("\n" + "=" * 50)
print("                 STREAM")
print("=" * 50)

for chunk in llm.stream(messages):
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()


# =====================================
# BATCH
# =====================================
print("\n" + "=" * 50)
print("                  BATCH")
print("=" * 50)

batch_questions = [
    "What is RAG? Explain briefly in Indonesian.",
    "What is a vector database? Explain briefly in Indonesian."
]

batch_messages = [
    [
        SystemMessage(
            content="Answer in plain text, briefly, without Markdown."
        ),
        HumanMessage(content=question)
    ]
    for question in batch_questions
]

responses = llm.batch(batch_messages)

for i, response in enumerate(responses, start=1):
    print(f"\nPertanyaan {i}:")
    print("-" * 30)
    print(response.text)


# =====================================
# SELESAI
# =====================================
print("\n" + "=" * 50)
print("             PROGRAM SELESAI")
print("=" * 50)
