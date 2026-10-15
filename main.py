import hashlib
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.0)


print("--- STEP 1: Basic Call & Token Tracking ---")

messages = [
    SystemMessage(content="You are a concise, helpful technical assistant."),
    HumanMessage(
        content="Explain the difference between a process and a thread in two sentences."
    ),
]

response = llm.invoke(messages)

print(f"Content:\n{response.content}\n")
print(f"Usage Metadata: {response.usage_metadata}")

total_input_tokens = response.usage_metadata.get("input_tokens", 0)
total_output_tokens = response.usage_metadata.get("output_tokens", 0)


print("\n--- STEP 2: Defined Web Search Tool with Disk Cache ---")

CACHE_DIR = Path(".tool_cache")
CACHE_DIR.mkdir(exist_ok=True)


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for real-time information, recent news, or factual facts.

    Args:
        query: Clear, specific search terms to look up.
    """
    query_hash = hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{query_hash}.json"

    if cache_file.exists():
        print(f"[CACHE HIT] Returning cached DuckDuckGo result for: '{query}'")
        return cache_file.read_text(encoding="utf-8")

    print(f"[CACHE MISS] Fetching real DuckDuckGo search for: '{query}'")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        formatted_result = "No search results found."
    else:
        formatted_result = "\n\n".join(
            [
                f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}"
                for r in results
            ]
        )

    cache_file.write_text(formatted_result, encoding="utf-8")
    return formatted_result


print(f"Tool Name: {web_search.name}")
print(f"Tool Description (what LLM sees): {web_search.description}")


print("\n--- STEP 3: Manual ReAct Loop ---")

llm_with_tools = llm.bind_tools([web_search])

react_messages = [
    SystemMessage(
        content=(
            "You are a helpful technical assistant. "
            "When tool results are provided to you, synthesize them into a clear text answer. "
            "Do NOT issue any further tool calls once you receive the search results."
        )
    ),
    HumanMessage(content="What is the latest news about Groq's AI hardware?"),
]

step1_response = llm_with_tools.invoke(react_messages)

total_input_tokens += step1_response.usage_metadata.get("input_tokens", 0)
total_output_tokens += step1_response.usage_metadata.get("output_tokens", 0)

print("Model produced tool call requests:")
for tool_call in step1_response.tool_calls:
    print(f" - Tool Name: {tool_call['name']}")
    print(f" - Tool Call Args: {tool_call['args']}")
    print(f" - Tool Call ID: {tool_call['id']}")

react_messages.append(step1_response)

for tool_call in step1_response.tool_calls:
    if tool_call["name"] == web_search.name:
        tool_output = web_search.invoke(tool_call["args"])

        react_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

final_response = llm_with_tools.invoke(react_messages)
total_input_tokens += final_response.usage_metadata.get("input_tokens", 0)
total_output_tokens += final_response.usage_metadata.get("output_tokens", 0)

final_text = final_response.content
if not final_text and final_response.tool_calls:
    final_text = str(final_response.tool_calls[0].get("args", {}))

print(f"\nFinal Answer:\n{final_text if final_text else 'No response content generated.'}")
print("\n--- STEP 4: Structured Output (10 Runs) ---")


class RoutingDecision(BaseModel):
    """Router classification schema for supervisor nodes."""

    reasoning: str = Field(
        description="Brief explanation of why this route was selected."
    )
    next_node: Literal["search_agent", "code_agent", "final_responder"] = Field(
        description="The target worker node to route the conversation to."
    )


structured_llm = llm.with_structured_output(RoutingDecision, method="json_mode")

sample_prompts = [
    "Find the current status of AAPL stock.",
    "Write a Python script to invert a binary tree.",
    "Summarize what we have discussed so far.",
    "Search for recent papers on LLM agents.",
    "Debug this memory leak in my C++ array.",
    "Give me the final summary report.",
    "Look up recent developments in quantum computing.",
    "Implement quicksort algorithm in Rust.",
    "Thank you, I have all the answers I need.",
    "Lookup documentation for FastAPI request validation.",
]

for idx, user_prompt in enumerate(sample_prompts, start=1):
    result: RoutingDecision = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a router. Return JSON matching this schema:\n"
                    "{\n"
                    '  "reasoning": "string",\n'
                    '  "next_node": "search_agent" | "code_agent" | "final_responder"\n'
                    "}"
                )
            ),
            HumanMessage(content=user_prompt),
        ]
    )
    print(f"Run {idx:02d} | Input: '{user_prompt}'")
    print(
        f"       -> Route: [{result.next_node}] | Reason: {result.reasoning}\n"
    )


print("--- SUMMARY ---")
print(f"Cumulative Input Tokens : {total_input_tokens}")
print(f"Cumulative Output Tokens: {total_output_tokens}")
print(f"Total Tokens Used       : {total_input_tokens + total_output_tokens}")