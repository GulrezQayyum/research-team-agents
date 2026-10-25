import os
import json
import re
import time
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_llm, tracker
from tools import web_search

MAX_SEARCHES = 4

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content="You are a research assistant. Research the topic using AT MOST 4 search tool calls.\n"
    "Rules:\n"
    "1. Write a 400 to 600 word detailed report based on your research.\n"
    "2. Provide explicit URL citations (e.g., [Title](URL)) for every claim.\n"
    "3. NEVER cite a source or URL that did not appear in actual search results.\n"
    "4. If search results return empty or error, synthesize whatever partial knowledge you have or state limitations clearly."
)

llm = get_llm()
tools = [web_search]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)
llm_without_tools = llm

def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Standard ReAct reasoning node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def final_report_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Budget-exhausted synthesis node.
    
    Inforces final text generation without tool-calling capabilities.
    """
    synthesis_instruction = HumanMessage(
        content="You have reached your maximum search budget. Synthesize a 400-600 word report using ONLY the search results retrieved above. Cite all explicit URLs and state any limitations."
    )
    prompt_messages = state["messages"] + [synthesis_instruction]
    response = llm_without_tools.invoke(prompt_messages)
    return {"messages": [response]}

def route_tools(state: AgentState) -> str:
    """Routes execution based on executed tool count and tool call requests."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_messages_executed = sum(1 for msg in messages if isinstance(msg, ToolMessage))
    
    if tool_messages_executed >= MAX_SEARCHES:
        return "final_report"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
        
    return END

workflow = StateGraph(AgentState)

workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)
workflow.add_node("final_report", final_report_node)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    route_tools,
    {
        "tools": "tools",
        "final_report": "final_report",
        END: END
    }
)

workflow.add_edge("tools", "llm")
workflow.add_edge("final_report", END)

graph_app = workflow.compile()

def extract_urls_from_tool_messages(messages: List[BaseMessage]) -> List[str]:
    """Parses raw ToolMessages to extract retrieved URLs for evaluation."""
    urls = []
    url_pattern = re.compile(r'https?://[^\s><"]+|www\.[^\s><"]+')
    for msg in messages:
        if isinstance(msg, ToolMessage):
            found_urls = url_pattern.findall(str(msg.content))
            urls.extend(found_urls)
    return list(set(urls))

def run_agent_with_metrics(topic: str) -> Dict[str, Any]:
    """Runs agent using app.stream to capture metrics and context during crashes."""
    start_time = time.time()
    
    initial_user_msg = HumanMessage(
        content=(
            f"Research topic: {topic}\n"
            f"Use at most {MAX_SEARCHES} searches, write a 400-600 word report, "
            f"and include full URL citations from tool outputs."
        )
    )
    
    initial_state = {"messages": [SYSTEM_PROMPT, initial_user_msg]}
    
    error_msg = None
    accumulated_messages: List[BaseMessage] = []
    
    try:
        # Use streaming to maintain state integrity across partial crashes
        for event in graph_app.stream(initial_state, config={"recursion_limit": 20}):
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    accumulated_messages.extend(node_output["messages"])
    except Exception as e:
        error_msg = f"GraphExecutionError: {str(e)}"

    wall_time = round(time.time() - start_time, 2)
    
    llm_calls = 0
    tool_calls_count = 0
    input_tokens = 0
    output_tokens = 0
    final_report = ""

    for msg in accumulated_messages:
        if isinstance(msg, AIMessage):
            llm_calls += 1
            if msg.tool_calls:
                tool_calls_count += len(msg.tool_calls)
            
            usage = getattr(msg, "usage_metadata", None) or {}
            if usage:
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
            
            if not msg.tool_calls and msg.content:
                final_report = msg.content

    retrieved_urls = extract_urls_from_tool_messages(accumulated_messages)

    status_failed = False
    if not final_report or len(final_report.strip()) < 50:
        status_failed = True
        if not error_msg:
            error_msg = "FAILED: empty_report (No final text synthesis produced)"

    return {
        "topic": topic,
        "status_failed": status_failed,
        "wall_time_sec": wall_time,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_urls": retrieved_urls,
        "error": error_msg,
        "report": final_report
    }

if __name__ == "__main__":
    test_topics = [
        "Recent breakthroughs in solid-state battery commercialization",
        "Developments in AI hardware inference accelerators"
    ]
    
    print("=== RUNNING REFACTORED BASELINE BENCHMARK ===\n")
    for topic in test_topics:
        res = run_agent_with_metrics(topic)
        print("=" * 60)
        print(f"TOPIC           : {res['topic']}")
        print(f"FAILED STATUS   : {res['status_failed']}")
        print(f"WALL TIME       : {res['wall_time_sec']}s")
        print(f"LLM / TOOL CALLS: {res['llm_calls']} LLM calls | {res['tool_calls']} Tool calls")
        print(f"TOKENS CONSUMED : {res['input_tokens']} in / {res['output_tokens']} out")
        print(f"RETRIEVED URLS  : {len(res['retrieved_urls'])} URLs captured")
        print(f"ERROR           : {res['error']}")
        print("-" * 60)
        print("REPORT PREVIEW:")
        print(res['report'][:500] + "...\n" if len(res['report']) > 500 else res['report'])
        print("=" * 60 + "\n")