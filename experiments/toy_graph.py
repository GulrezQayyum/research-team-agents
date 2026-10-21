import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State Schema with custom append reducer
class ToyState(TypedDict):
    counter: int
    history: Annotated[list[str], operator.add]

# 2. Plain Python Nodes
def node_a(state: ToyState) -> dict:
    return {
        "counter": state["counter"] + 1,
        "history": [f"Node A visited (count={state['counter'] + 1})"]
    }

def node_b(state: ToyState) -> dict:
    return {
        "counter": state["counter"] + 1,
        "history": [f"Node B visited (count={state['counter'] + 1})"]
    }

def node_c(state: ToyState) -> dict:
    return {
        "history": ["Node C finalized execution"]
    }

# 3. Conditional Router
def should_continue(state: ToyState) -> str:
    if state["counter"] < 3:
        return "node_b"
    return "node_c"

# 4. Build and Compile Graph
workflow = StateGraph(ToyState)

workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_node("node_c", node_c)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")

# Loop until counter limit met, then route to C
workflow.add_conditional_edges("node_b", should_continue, {
    "node_b": "node_b",
    "node_c": "node_c"
})
workflow.add_edge("node_c", END)

app = workflow.compile()

result = app.invoke({
    "counter": 0,
    "history": []
})

print(result)