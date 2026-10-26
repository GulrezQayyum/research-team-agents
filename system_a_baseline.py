import os
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_llm, tracker
from tools import web_search

MAX_SEARCHES = 4
RESULTS_DIR = Path(".results")
RESULTS_DIR.mkdir(exist_ok=True)

CURRENT_DATE_STR = datetime.now().strftime("%B %d, %Y")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content=(
        f"Today is {CURRENT_DATE_STR}.\n"
        f"You are a research assistant. Research the topic using AT MOST {MAX_SEARCHES} search tool calls.\n"
        "Rules:\n"
        "1. Write a 400 to 600 word detailed report based strictly on retrieved research.\n"
        "2. Provide explicit URL citations (e.g., [Title](URL)) for every claim made.\n"
        "3. NEVER cite a source or URL that did not explicitly appear in actual search results.\n"
        "4. If search results are incomplete or fail, state these limitations clearly. Do NOT invent claims or use uncited parametric knowledge."
    )
)

llm = get_llm()
tools = [web_search]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)
llm_without_tools = llm

def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Standard ReAct reasoning node with tool binding."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def final_report_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Budget-exhausted synthesis node (unbound LLM call)."""
    synthesis_instruction = HumanMessage(
        content=(
            f"Current Date: {CURRENT_DATE_STR}. "
            "You have reached your maximum search budget. Synthesize a 400-600 word report using ONLY the search results retrieved above. "
            "Cite explicit URLs from tool outputs and state any search limitations clearly."
        )
    )
    prompt_messages = state["messages"] + [synthesis_instruction]
    response = llm_without_tools.invoke(prompt_messages)
    return {"messages": [response]}

def route_tools(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Count executed ToolMessage payloads
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
    urls = []
    url_pattern = re.compile(r'https?://[^\s><"\'\]\)]+')
    for msg in messages:
        if isinstance(msg, ToolMessage):
            found_urls = url_pattern.findall(str(msg.content))
            urls.extend(found_urls)
    return sorted(list(set(urls)))

def evaluate_citation_honesty(report_text: str, retrieved_urls: List[str]) -> Dict[str, Any]:
    url_pattern = re.compile(r'https?://[^\s><"\'\]\)]+')
    cited_urls = list(set(url_pattern.findall(report_text)))
    
    valid_citations = [url for url in cited_urls if url in retrieved_urls]
    hallucinated_citations = [url for url in cited_urls if url not in retrieved_urls]
    
    honesty_ratio = len(valid_citations) / len(cited_urls) if cited_urls else 1.0
    
    return {
        "cited_urls": cited_urls,
        "valid_citations": valid_citations,
        "hallucinated_citations": hallucinated_citations,
        "honesty_ratio": round(honesty_ratio, 2)
    }


def run_agent_with_metrics(topic: str, run_id: int) -> Dict[str, Any]:
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
    final_messages: List[BaseMessage] = []
    
    try:
        final_state = graph_app.invoke(initial_state, config={"recursion_limit": 20})
        if final_state and "messages" in final_state:
            final_messages = final_state["messages"]
    except Exception as e:
        error_msg = str(e)

    wall_time = round(time.time() - start_time, 2)
    
    llm_calls = 0
    tool_calls_count = 0
    input_tokens = 0
    output_tokens = 0
    final_report = ""

    for msg in final_messages:
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

    retrieved_urls = extract_urls_from_tool_messages(final_messages)
    citation_eval = evaluate_citation_honesty(final_report, retrieved_urls)

    status_failed = False
    failure_reason = None

    if error_msg:
        status_failed = True
        if "recursion_limit" in error_msg.lower():
            failure_reason = "recursion_limit"
        else:
            failure_reason = "graph_exception"
    elif not final_report or len(final_report.strip()) < 50:
        status_failed = True
        failure_reason = "empty_report"
    elif any(isinstance(m, AIMessage) and getattr(m, "invalid_tool_calls", None) for m in final_messages):
        status_failed = True
        failure_reason = "tool_call_invalid"

    run_payload = {
        "run_id": run_id,
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "status_failed": status_failed,
        "failure_reason": failure_reason,
        "wall_time_sec": wall_time,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retrieved_urls": retrieved_urls,
        "citation_eval": citation_eval,
        "error_message": error_msg,
        "report": final_report
    }

    safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic[:25])
    out_file = RESULTS_DIR / f"run_{safe_topic}_run{run_id}_{int(time.time())}.json"
    out_file.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    return run_payload


if __name__ == "__main__":
    test_topics = [
        "Recent breakthroughs in solid-state battery commercialization",
        "Developments in AI hardware inference accelerators"
    ]
    
    print(f"=== SYSTEM A BENCHMARK EXECUTION ({CURRENT_DATE_STR}) ===\n")
    
    summary_results = []
    
    for topic in test_topics:
        print(f"\n--- Benchmark Topic: '{topic}' ---")
        for run_idx in range(1, 4):
            print(f"Executing Run {run_idx}/3...")
            res = run_agent_with_metrics(topic, run_idx)
            summary_results.append(res)
            print(
                f"Run {run_idx} Complete | Time: {res['wall_time_sec']}s | "
                f"Failed: {res['status_failed']} ({res['failure_reason'] or 'None'}) | "
                f"Honesty Ratio: {res['citation_eval']['honesty_ratio']}"
            )
            time.sleep(2)  