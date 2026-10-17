from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from config import get_llm, tracker
from tools import web_search

# Load base model
llm = get_llm()

# =====================================================================
# STEP 1: Basic Call & Token Tracking
# =====================================================================
print("--- STEP 1: Basic Call & Token Tracking ---")

messages = [
    SystemMessage(content="You are a concise, helpful technical assistant."),
    HumanMessage(content="Explain the difference between a process and a thread in two sentences."),
]

response = llm.invoke(messages)
tracker.add_usage(response.usage_metadata)

print(f"Content:\n{response.content}\n")
print(f"Step 1 Usage: {response.usage_metadata}\n")


# =====================================================================
# STEP 2: Tool Inspection
# =====================================================================
print("--- STEP 2: Tool Inspection ---")
print(f"Tool Name       : {web_search.name}")
print(f"Tool Description: {web_search.description}\n")


# =====================================================================
# STEP 3: Robust Multi-Turn ReAct Loop
# =====================================================================
print("--- STEP 3: Manual ReAct Loop ---")

tools_by_name = {web_search.name: web_search}
llm_with_tools = llm.bind_tools([web_search])

react_messages = [
    SystemMessage(
        content=(
            "You are a helpful assistant. Use search tools when required. "
            "When tool results (or errors) are returned, synthesize them into a concise text response."
        )
    ),
    HumanMessage(content="What is the latest news about Groq AI hardware?"),
]

max_iterations = 3
final_answer = None

for iteration in range(1, max_iterations + 1):
    print(f"--- Iteration {iteration} ---")
    ai_msg = llm_with_tools.invoke(react_messages)
    tracker.add_usage(ai_msg.usage_metadata)
    react_messages.append(ai_msg)

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]

            print(f"Tool Call Requested: {t_name}(args={t_args})")

            if t_name in tools_by_name:
                tool_output = tools_by_name[t_name].invoke(t_args)
            else:
                tool_output = f"Error: Tool '{t_name}' is not recognized."

            react_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=t_id, name=t_name)
            )
    else:
        final_answer = ai_msg.content
        break

# Handle edge case where max_iterations is reached without text output
if not final_answer or not final_answer.strip():
    # Make one final invocation without tools to force text generation
    final_fallback = llm.invoke(react_messages)
    tracker.add_usage(final_fallback.usage_metadata)
    final_answer = final_fallback.content

print(f"\nFinal ReAct Output:\n{final_answer}\n")


# =====================================================================
# STEP 4: Structured Output Benchmark (10 Runs)
# =====================================================================
print("--- STEP 4: Structured Output (10 Runs) ---")

class RoutingDecision(BaseModel):
    """Supervisor node routing schema."""
    reasoning: str = Field(description="Explanation for selecting the target node.")
    next_node: Literal["search_agent", "code_agent", "final_responder"] = Field(
        description="Target worker agent node."
    )

# Use json_mode for robust structured extraction on open-source models
structured_llm = llm.with_structured_output(
    RoutingDecision, method="json_mode", include_raw=True
)

system_router_instruction = SystemMessage(
    content=(
        "You are a router that classifies input requests. Do NOT solve or answer the user request. "
        "Return ONLY a JSON object strictly matching this schema:\n"
        "{\n"
        '  "reasoning": "brief explanation",\n'
        '  "next_node": "search_agent" | "code_agent" | "final_responder"\n'
        "}"
    )
)

test_cases = [
    ("Find current market price of AAPL stock.", "search_agent"),
    ("Write a Python script to invert a binary tree.", "code_agent"),
    ("Summarize our discussion so far.", "final_responder"),
    ("Look up recent developments in quantum computing.", "search_agent"),
    ("Debug memory leak in C++ raw pointer array.", "code_agent"),
    ("Wrap up the report and state response is complete.", "final_responder"),
    ("Search for recent academic papers on LLM agents.", "search_agent"),
    ("Implement quicksort algorithm in Rust.", "code_agent"),
    ("Thank you, I have all needed answers.", "final_responder"),
    ("Lookup documentation for FastAPI request validation.", "search_agent"),
]

parsed_successes = 0
routing_accuracies = 0

for idx, (user_prompt, expected_route) in enumerate(test_cases, start=1):
    try:
        output_dict = structured_llm.invoke(
            [system_router_instruction, HumanMessage(content=user_prompt)]
        )

        raw_msg = output_dict.get("raw")
        if raw_msg and hasattr(raw_msg, "usage_metadata"):
            tracker.add_usage(raw_msg.usage_metadata)

        parsed_obj: RoutingDecision = output_dict.get("parsed")

        if parsed_obj:
            parsed_successes += 1
            is_correct = parsed_obj.next_node == expected_route
            if is_correct:
                routing_accuracies += 1

            status = "CORRECT" if is_correct else f"MISMATCH (Expected: {expected_route})"
            print(f"Run {idx:02d} | Route: [{parsed_obj.next_node}] | Status: {status}")
            print(f"       Reasoning: {parsed_obj.reasoning}")
        else:
            print(f"Run {idx:02d} | [PARSING FAILED] Could not validate schema.")

    except Exception as e:
        print(f"Run {idx:02d} | [EXCEPTION] Execution failed: {str(e)}")

print(f"\n--- Benchmark Performance Results ---")
print(f"Schema Parse Success Rate: {parsed_successes}/{len(test_cases)} ({parsed_successes * 10}%)")
print(f"Routing Accuracy Rate   : {routing_accuracies}/{len(test_cases)} ({routing_accuracies * 10}%)\n")

tracker.summary()