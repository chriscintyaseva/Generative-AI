
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# =====================================
# SIMPLE TEMPLATE
# =====================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Format to messages (for inspection)
msgs = prompt.format_messages(
    role="Python tutor",
    language="English",
    question="What is a generator?",
)

print("=== SIMPLE TEMPLATE ===")

for msg in msgs:
    print(f"\nRole: {msg.type}")
    print(f"Content: {msg.content}")


# =====================================
# TEMPLATE WITH CONVERSATION HISTORY
# =====================================

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)

print("\n=== CONVERSATION HISTORY ===")
print(f"Total messages in formatted prompt: {len(formatted)}")

for msg in formatted:
    print(f"\nRole: {msg.type}")
    print(f"Content: {msg.content}")
