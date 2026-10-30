
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List


# Initialize Ollama
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)


# Define structured output schema
class ModelComparison(BaseModel):
    models: List[str] = Field(
        description="Model names compared"
    )
    best_for_cost: str = Field(
        description="Most cost-effective model"
    )
    best_for_quality: str = Field(
        description="Highest quality model"
    )
    recommendation: str = Field(
        description="Overall recommendation"
    )


# Enable structured output
structured_llm = llm.with_structured_output(
    ModelComparison,
    method="json_schema"
)


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert."),
    ("human", "Compare these models for a startup on a budget: {models}")
])


# Create chain
chain = prompt | structured_llm


# Run the program
print("=" * 50)
print("          STRUCTURED OUTPUT")
print("=" * 50)

result: ModelComparison = chain.invoke({
    "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
})


print("\nModels Compared:")
for model in result.models:
    print(f"- {model}")

print("\nBest for Cost:")
print(result.best_for_cost)

print("\nBest for Quality:")
print(result.best_for_quality)

print("\nRecommendation:")
print(result.recommendation)

print("\n" + "=" * 50)
print("             PROGRAM SELESAI")
print("=" * 50)
