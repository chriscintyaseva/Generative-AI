
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator


# ==========================================
# 1. STATE DEFINITION
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    step_count: int
    final_answer: str


# ==========================================
# 2. NODE FUNCTIONS
# ==========================================

def think_node(state: AgentState) -> dict:
    thought = (
        f"Step {state['step_count'] + 1}: "
        "analysing the problem..."
    )

    return {
        "messages": [thought],
        "step_count": state["step_count"] + 1
    }


def answer_node(state: AgentState) -> dict:
    return {
        "final_answer": "Based on my analysis, the answer is X."
    }


# ==========================================
# 3. CONDITIONAL EDGE
# ==========================================

def should_continue(state: AgentState) -> str:
    if state["step_count"] < 2:
        return "think"

    return "answer"


# ==========================================
# 4. BUILD GRAPH
# ==========================================

graph = StateGraph(AgentState)

graph.add_node("think", think_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "think")

graph.add_conditional_edges(
    "think",
    should_continue,
    {
        "think": "think",
        "answer": "answer"
    }
)

graph.add_edge("answer", END)


# ==========================================
# 5. COMPILE GRAPH
# ==========================================

app = graph.compile()


# ==========================================
# 6. RUN GRAPH
# ==========================================

result = app.invoke({
    "messages": [],
    "step_count": 0,
    "final_answer": ""
})


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("Messages:", result["messages"])
print("Answer:", result["final_answer"])
print("Total Steps:", result["step_count"])
