
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator


# Define state
class ReviewState(TypedDict):
    plan: str
    approved: bool
    result: str
    messages: Annotated[list[str], operator.add]


# Node 1: Create execution plan
def plan_node(state: ReviewState) -> dict:
    plan = (
        "Step 1: Query vector DB. "
        "Step 2: Summarise results. "
        "Step 3: Format response."
    )

    return {
        "plan": plan,
        "messages": [f"Plan created: {plan}"]
    }


# Node 2: Execute after approval
def execute_node(state: ReviewState) -> dict:
    if not state["approved"]:
        return {
            "result": "Execution skipped - plan not approved.",
            "messages": ["Skipped."]
        }

    return {
        "result": "Pipeline executed successfully.",
        "messages": ["Executed."]
    }


# Create memory checkpoint
checkpointer = MemorySaver()

# Build graph
graph = StateGraph(ReviewState)

graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

# Pause before execute for human approval
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"]
)

config = {
    "configurable": {
        "thread_id": "review-session-1"
    }
}


# Run until interrupt
print("========== GENERATING PLAN ==========")

state = app.invoke(
    {
        "plan": "",
        "approved": False,
        "result": "",
        "messages": []
    },
    config=config
)

print("Plan:", state["plan"])
print("\nWaiting for human approval...")


# Human approval
print("\n========== HUMAN REVIEW ==========")

state_update = {
    "approved": True
}

# Update approval state and resume execution
app.update_state(config, state_update)

final = app.invoke(None, config=config)

print("\n========== FINAL RESULT ==========")
print("Result:", final["result"])

print("\nMessages:")
for message in final["messages"]:
    print("-", message)
