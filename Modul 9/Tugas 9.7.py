
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# Initialize Ollama
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


# Create chain
chain = prompt | llm | parser


# Session store - keyed by session_id
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# Wrap chain with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# Session configuration
config = {
    "configurable": {
        "session_id": "user-123"
    }
}



# Turn 1
r1 = chain_with_history.invoke(
    {"input": "What is RAG?"},
    config=config
)

print("\n" + "=" * 60)
print("TURN 1 - PERTANYAAN PERTAMA")
print("=" * 60)
print("User: What is RAG?")
print("\nAI:")
print(r1)


# Turn 2
r2 = chain_with_history.invoke(
    {"input": "What are the main components I just asked about?"},
    config=config
)

print("\n" + "=" * 60)
print("TURN 2 - PERTANYAAN KEDUA")
print("=" * 60)
print("User: What are the main components I just asked about?")
print("\nAI:")
print(r2)


# Inspect stored history
history = get_session_history("user-123")

print("\n" + "=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)
print(f"Total messages in history: {len(history.messages)}")

print("\nProgram selesai.")
